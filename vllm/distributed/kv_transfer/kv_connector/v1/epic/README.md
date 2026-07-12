# EPIC Connector — Non-contiguous KV Cache Reuse for vLLM V1

EPIC(Efficient Position-Independent Caching, [arXiv:2410.15332](https://arxiv.org/pdf/2410.15332))의
non-contiguous KV cache reuse를 vLLM v0.22.1(V1 엔진)에 **KV connector로 정식 통합**한 구현.
원본([DerekHJH/epic](https://github.com/DerekHJH/epic), vLLM 0.7.0 fork)은 단일요청·dense·모델패치
데모였고, 본 구현은 그 알고리즘 본체(PIC, selective recompute, fusion attention)를
`KVConnectorBase_V1` + FlexAttention 위에 재설계한 것이다.

## 1. 해결하는 문제 (infix 시나리오)

prompt = **A + B + C + D + F + G** (A: 캐시된 prefix, C·F: 별도 워밍된 코드 파일
"fileKV", B·D·G: 새 텍스트):

- vLLM native prefix cache는 **A까지만** 재사용 가능 (블록 해시가 prefix-chain이라
  중간에 박힌 C·F는 영원히 매칭 불가).
- infix reuse: A는 native/EPIC prefix로, C·F는 fileKV에서 로드하고 두 보정을 적용:
  - **PIC (위치 보정)**: 캐시된 K에는 옛 위치의 RoPE가 박혀 있음 → delta 재회전.
  - **selective recompute (의미 보정)**: C·F는 A·B를 "본 적 없는" KV → 경계 link
    토큰만 재계산해 문맥을 부분 주입 (LegoLink).
- 계산 절감의 본질: **M = B∪D∪G ∪ link(C)∪link(F) ∪ {N-1}만 1-step forward**
  (sparse-forward). full-forward + mask 방식은 절감이 없어 기각됨.
- causal 정합: FlexAttention `logical_q_positions`로 M 각 행이 자기 논리 위치
  기준 causal attention (B 토큰은 자기 앞만 봄).

수학적 증명: `test_infix_scenario_equivalence.py` — 실제 PICRotator +
LegoLinkRecompute를 float64 토이 트랜스포머로 구동, HF-spec 절차(절대위치
isolated + Phase-B 결합 forward)와 atol 1e-10 완전 동치. B 행들은 dense
prefill과 **완전 일치**(세그먼트별 causality의 행동적 증명).

## 2. 모듈 구성

```
epic/
├── epic_connector.py    EpicConnector(KVConnectorBase_V1) — scheduler/worker 양 role
├── chunk_store.py       EpicChunkStore(CPU LRU) + EpicSchedulerIndex(mirror)
│                        + 내용해시/문맥체인(ChainHasher) 프리미티브
├── pic.py               PICRotator — delta 재회전 R(p_new−p_old), K 전용, cos/sin memo
├── reuse_strategy.py    4-전략 추상화(ABC) + EPIC 구현체 (chain-strict selection,
│                        LegoLink per-chunk/per-run link)
├── fusion_mask.py       FlexAttention mask_mod — 고정 텐서 lookup (재컴파일 회피)
├── runner_sparse.py     runner용 sparse row edit 헬퍼 (순수 로직)
├── metadata.py          scheduler→worker 직렬화 경계 (loads/saves/sparse/prefetch)
├── prefetch.py          EpicGpuStagingStore — GPU staging (side-stream H2D)
├── prefetch_parser.py   tool-call 파싱 + FileKVPrefetcher (클라이언트 측)
├── prefetch_service.py  ZMQ listener/client + DynamoPrefetchBridge (외부 주입)
├── staging_worker.py    rank당 전용 staging 프로세스 (CUDA IPC; MIG 검토·기각)
├── filekv_catalog.py    파일 버전 장부 — 수정 정합성 + 부분 읽기 정규화
├── DESIGN.md            공통 추상 인터페이스 설계 (EPIC + CacheBlend)
├── PHASE2.md            sparse 구현 상태 + core 패치 anchor + GPU 검증 절차
├── PREFETCH.md          prefetch/dynamo/staging/정합성 상세
└── README.md            (이 문서)
```

등록: `factory.py`에 `"EpicConnector"` (기존 파일 수정 목록은 §4 표 참조).

## 3. 단계별 구현 내역

### Phase 1 — 위치독립 prefix 재사용 (비침습)
- chunk(256tok)를 **내용 단독 해시**로 저장/매칭; load 시 PIC delta rotation.
- 수명주기: `get_num_new_matched_tokens` → `update_state_after_alloc` →
  `build_connector_meta` → `start_load_kv`(scatter) / `save_kv_layer`(harvest).

### Phase 2a — FlexAttention partial-mask 배선 (core 무수정)
- custom mask × paged KV를 지원하는 V1 backend는 **FlexAttention 유일**.
- `fusion_mask.py`: 고정 텐서 in-place 갱신 + 단일 mask_mod 객체 → 재컴파일 회피.

### Phase 2b — sparse-forward (core 패치, 전부 default-off 가드)
| 패치 | 파일 | 내용 |
|---|---|---|
| 회계 | `v1/core/sched/scheduler.py` | external=\|A\|+\|B\| 보고, `num_scheduled→\|M\|`, `num_computed→N` 수렴+길이 불변식 ValueError, `delay_cache_blocks`, 단독 배치 게이트, **truncation 방어 defer** |
| 출력 | `v1/core/sched/output.py` | `epic_sparse_positions`/`epic_seq_len`/`epic_computed_advance` |
| positions | `v1/worker/gpu_model_runner.py` | M의 논리 위치로 positions 덮어쓰기(→ token gather/slot_mapping 자동 정합), `seq_lens=N` |
| logical q | `v1/attention/backends/flex_attention.py` | `logical_q_positions` (연속 케이스 항등) |
| 플래그 | `v1/attention/backend.py` | `CommonAttentionMetadata.epic_sparse_logical_q` |

안전 게이팅: sparse on이면 FlexAttention + enforce_eager 검증(fail-closed).

### Hardening (infix 시나리오 평가에서 나온 실버그 수정)
- **emit/등록 일관성**: `_emit_sparse`는 match 시 등록된 요청만 방출
  (아니면 advance=N 오폭 → 길이 불변식 크래시였음).
- **budget guard**: 1-step prefill이 토큰 버짓/long-prefill 임계에 안 맞으면
  sparse 거절 → prefix-only 강등 (절단되면 블록이 [0,N) 미커버 → OOB였음).
- **native 영역 보호**: `num_computed` 이하(공유 native 블록, 정확 KV)에는
  근사 store KV를 절대 scatter하지 않음 (`src_offset` head-trim). effective
  prefix = max(store prefix, native extent) → A가 native에만 있어도 sparse 동작.
- **문맥 체인 (context-sound fold)**: content hash는 "같은 바이트"만 증명 —
  저장 시 `ChainHasher` digest(chain_start/chain_end)를 기록하고, sparse 모드의
  exact-prefix fold는 **체인 일치 시에만** 허용. 독립 워밍된 파일이 prefix에
  인접해도 스티치 없이 흡수되지 않음 (non-prefix 강등 → link 재계산).
- **per-run link** (`epic_link_per_run`, 기본 off=EPIC 원본 per-chunk):
  prompt 인접 + 체인 연속(같은 warm 증명)인 run은 head chunk에만 link.

### Prefetch / dynamo / staging (상세: PREFETCH.md)
- **GPU staging**: 이전 turn의 tool-call을 파싱해 다음 turn의 fileKV를 지정
  worker의 GPU에 선적재 (side-stream H2D → 로드 시 H2D 생략).
- **외부 명령 주입**: `epic_prefetch_endpoint`(ZMQ REP) ←
  `EpicPrefetchClient`/`DynamoPrefetchBridge` (dynamo frontend가 배치 결정
  `dst_worker`와 함께 주입; dropped 응답 → `warm_fn` 재워밍).
- **전용 staging worker** (`epic_staging_mode:"external"`): rank당 별도
  프로세스(같은 GPU, 자체 CUDA context, torch mp CUDA-IPC zero-copy 매핑).
  **MIG는 검토 후 기각** — 인스턴스 간 메모리 공유(P2P/IPC) 불가.
  TP: rank별 store/staging으로 shard 격리, hit/miss skew 무해(무collective).
- **파일 수정 정합성**: content-addressing이 정확성 보장(수정 바이트는 해시
  불일치, v1/v2 공존 안전). `FileKVCatalog`(에이전트 공유 버전 장부) +
  `evict_hashes` 디렉티브(구버전 staged 즉시 회수; evict→stage 순서) +
  `on_file_modified` 재워밍 + torn-render 감지.
- **부분 읽기(줄 범위)**: sub-chunk(<256tok) 렌더는 캐시 안 함(재계산이 더 쌈);
  `snap_lines`로 겹치는 범위를 canonical 렌더 하나로 통일 가능.

### 성능 최적화 (CPU 실측, 32k-token prompt)
| 경로 | 개선 | 방법 |
|---|---|---|
| 분할+이중해싱 | ~4ms → 0.66ms | 프롬프트 1회 인코딩 + memoryview 윈도우 (digest 불변) |
| M 유도 | ~6-8ms → 0.70ms | numpy bool 마스크 (60-layout fuzz로 set 레퍼런스 고정) |
| slot id | 1.54 → 0.62ms | per-block range-extend (numpy 시도는 4.48ms로 기각) |
| match+save | 2회 해싱 → 1회 | per-step split 캐시; budget-guard plan 재사용 |
| PIC trig | 레이어당 → 청크당 1회 | cos/sin identity-memo + positions/slots hoist |
| identity delta | 회전 생략 | R(0)=I (prefix 재로드) |
| save 경로 | host copy 2회→1회, async | GPU→pinned 직행 D2H + `wait_for_save` fence |

## 4. 사용법

```bash
# Phase 1 (위치독립 prefix 재사용, 어떤 backend든)
--kv-transfer-config '{"kv_connector":"EpicConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"epic_chunk_size":256}}'

# sparse infix (FlexAttention + eager 필수)
... --attention-backend FLEX_ATTENTION --enforce-eager \
--kv-transfer-config '{"kv_connector":"EpicConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"epic_chunk_size":256,
    "epic_sparse_forward":true,"epic_fusion_mask":true,"epic_link_tokens":8}}'
```

| extra_config | 기본 | 의미 |
|---|---|---|
| `epic_chunk_size` | 256 | chunk 토큰 수 (block_size 배수 강제) |
| `epic_cpu_bytes` | 8GiB | CPU store LRU 예산 (world_size로 분할) |
| `epic_sparse_forward` | false | sparse-forward (M만 forward) |
| `epic_fusion_mask` | false | FlexAttention mask_mod 주입 |
| `epic_link_tokens` | 8 | non-prefix chunk당 재계산 경계 토큰 수 |
| `epic_link_per_run` | false | per-run(파일) link — 체인 연속 증명 시 head만 |
| `epic_prefetch_gpu_bytes` | 0 (off) | GPU staging 예산 (0=prefetch 완전 비활성) |
| `epic_worker_id` | -1 | frontend가 아는 이 replica의 id (디렉티브 필터) |
| `epic_prefetch_endpoint` | "" | ZMQ 명령 주입 endpoint (scheduler role) |
| `epic_staging_mode` | inprocess | "external"=rank당 전용 staging 프로세스 |
| `epic_debug_counters` | false | in-band 카운터 (sparse_match/prefetch_hit 등) |
| `epic_debug_check_load` | false | scatter read-back 검사 |

## 5. 검증 상태

- **CPU (전부 통과)**: main **311** / feature-prefetch **363** passed.
  ```bash
  /tmp/epic-test-venv/bin/python -m pytest tests/v1/kv_connector/unit/epic/ -q
  ```
  주요 스위트: `test_infix_scenario_equivalence`(A+B+C+D+F+G 동치 증명),
  `test_infix_corner_cases`(native-covers/fold/중복/비정렬/straddle/체인 19케이스),
  `test_spec_procedure_equivalence`, `test_sparse_*`(회계/가드/불변식),
  `test_pipeline_optimizations`(최적화 적용+동일성 이중 검증),
  `test_prefetch*`/`test_staging_worker`(실 프로세스 spawn + ZMQ 왕복, TP 격리),
  `test_filekv_consistency`(v1/v2 공존, evict 수명주기, catalog, 부분 읽기).
- **GPU (미실행)**: `gpu_smoke.py` 5단계 체크리스트(PHASE2.md),
  `benchmarks/epic_reuse/`(musique/accuracy/perf). CUDA IPC staging·async D2H
  실측·Issue 1(link<chunk 붕괴, per-run 비교) 대기.

## 6. 남은 작업

- GPU end-to-end (PHASE2.md 체크리스트) + Issue 1 link 실험 (per-chunk vs per-run).
- PIECEWISE cudagraph 완화 (현재 eager 강제); sparse 혼합 배치 (현재 단독 배치).
- dynamo→engine transport의 배포 통합 (endpoint는 준비됨; RPC 배선은 배포별).
- CacheBlend 전략 구현 (인터페이스는 `reuse_strategy.py`에 준비됨).
- rope_scaling 확장 (linear·llama3 지원; YaRN/longrope 미지원 시 로드만 비활성).

## 7. 원본 EPIC과의 대응표

| EPIC 원본 (vLLM 0.7, V0) | 본 구현 (V1) |
|---|---|
| `cache_fuse_metadata` dict를 forward 시그니처 관통 | `EpicConnectorMetadata` + forward_context |
| dense `old_kvs`를 모델 객체에 부착 | `EpicChunkStore`(CPU) → paged block scatter (+GPU staging) |
| `fake_q` 더미로 재-rotary | `PICRotator` delta rotation (K 전용, memo) |
| forward 중간 토큰 슬라이싱 (status 1→2) | **M pre-forward 동결** (1급 불변식) |
| xformers attn_bias | FlexAttention `mask_mod` + `logical_q_positions` |
| `model_runner`의 selected_token_indices 보정 | M 마지막=N-1 불변식으로 자동 정합 |
| 벤치마크가 엔진 내부 직접 조작 | connector 수명주기 (scheduler 통합) + 외부 주입 API |
| 내용만 보고 재사용 (문맥 무시) | 저장 시 문맥 체인 기록 → exact fold는 체인 일치 시에만 |
| custom op 비활성화 (eager 전용) | 기본 경로 무수정, sparse만 eager (완화 TODO) |
