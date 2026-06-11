# EPIC Connector — Non-contiguous KV Cache Reuse for vLLM V1

EPIC(Efficient Position-Independent Caching, [arXiv:2410.15332](https://arxiv.org/pdf/2410.15332))의
non-contiguous KV cache reuse를 vLLM v0.22.1(V1 엔진)에 **KV connector로 정식 통합**한 구현.
원본([DerekHJH/epic](https://github.com/DerekHJH/epic), vLLM 0.7.0 fork)은 단일요청·dense·모델패치
데모였고, 본 구현은 그 알고리즘 본체(PIC, selective recompute, fusion attention)를
`KVConnectorBase_V1` + FlexAttention 위에 재설계한 것이다.

## 1. 해결하는 문제

prompt = **A + C + B** (A, B는 과거 요청에서 KV가 캐시된 chunk, C는 새 토큰):

- vLLM native prefix cache는 **A까지만** 재사용 가능 (블록 해시가 prefix-chain이라
  중간에 박힌 B는 영원히 매칭 불가).
- B를 재사용하려면 두 보정이 필요:
  - **PIC (위치 보정)**: 캐시된 K에는 옛 위치의 RoPE가 박혀 있음 → 새 위치로 재회전.
  - **selective recompute (의미 보정)**: B는 A·C를 "본 적 없는" KV → 경계 link 토큰만
    재계산해 문맥을 부분 주입 (LegoLink).
- 계산 절감의 본질: **M = C ∪ link ∪ {마지막 토큰}만 forward** (sparse-forward).
  full-forward + mask 방식은 절감이 없어 기각됨.

## 2. 모듈 구성

```
epic/
├── epic_connector.py   EpicConnector(KVConnectorBase_V1) — scheduler/worker 양 role
├── chunk_store.py      EpicChunkStore — 내용해시(위치독립) chunk store, CPU LRU
├── pic.py              PICRotator — delta 재회전 R(p_new − p_old), K 전용
├── reuse_strategy.py   4-전략 추상화(ABC) + EPIC 구현체 (LegoLink)
├── fusion_mask.py      FlexAttention mask_mod — 고정 텐서 lookup (재컴파일 회피)
├── runner_sparse.py    runner용 sparse row edit 헬퍼 (순수 로직)
├── metadata.py         EpicConnectorMetadata — scheduler→worker 직렬화 경계
├── DESIGN.md           공통 추상 인터페이스 설계 (EPIC + CacheBlend)
├── PHASE2.md           구현 상태 + core 패치 anchor + GPU 검증 절차
└── README.md           (이 문서)
```

등록: `factory.py`에 `"EpicConnector"` (기존 파일 수정은 이 1곳뿐).

## 3. 단계별 구현 내역

### Phase 1 — 위치독립 prefix 재사용 (비침습)
- chunk를 **내용 단독 해시**(prefix-chain 아님)로 저장/매칭 → 과거에 어떤 위치에
  있었든 새 요청의 prefix 구간이면 재사용.
- load 시 `PICRotator`가 K에 delta rotation 적용 (RoPE 회전군 성질:
  R(Δ)·R(p_old) = R(p_new); 원본의 fake_q 핵 제거).
- core 무수정. 수명주기: `get_num_new_matched_tokens` → `update_state_after_alloc`
  → `build_connector_meta` → `start_load_kv`(scatter) / `save_kv_layer`(harvest).

### Phase 2a — FlexAttention partial-mask 배선 (core 무수정)
- V1에서 custom mask × paged KV를 지원하는 backend는 **FlexAttention 유일**
  (`mask_mod`/`score_mod`; flash_attn/flashinfer/triton/mla 전부 불가 — 조사로 확정).
- `fusion_mask.py`: `max_model_len` 크기 고정 텐서(`recompute_flag`/`kv_live`/`gate`)를
  1회 할당, 요청마다 내용만 in-place 갱신. mask_mod는 **항상 같은 함수 객체**
  → FlexAttention identity 체크 1회만 발화 → 재컴파일 회피.
- 주입: worker가 `forward_context.no_compile_layers`로 attention layer에
  `layer.logical_mask_mod`만 세팅 → flex가 자동 픽업 (기존 hook, core 무수정).

### Phase 2b — sparse-forward (core 패치, 전부 default-off 가드)
V1의 두 가정 — `num_computed_tokens`=연속 prefix 스칼라, runner positions=연속 구간 —
을 최소 침습으로 우회:

| 패치 | 파일 | 내용 |
|---|---|---|
| 회계 | `v1/core/sched/scheduler.py` | external=\|A\|+\|B\| 보고(블록이 N 커버), `num_scheduled→\|M\|` 오버라이드, `num_computed→N` 수렴, sparse는 `delay_cache_blocks`(근사 KV의 native cache 오염 차단), **sparse 요청 단독 배치** 게이트 |
| 출력 | `v1/core/sched/output.py` | `epic_sparse_positions`/`epic_seq_len`/`epic_computed_advance` (기본 empty) |
| positions | `v1/worker/gpu_model_runner.py` | M의 진짜 logical 위치로 positions 덮어쓰기 → **token gather와 slot_mapping은 positions 파생이라 자동 정합**; `seq_lens=N` 오버라이드 |
| logical q | `v1/attention/backends/flex_attention.py` | `logical_q_positions` 필드 — 기존 `decode_offset+local` 식은 "query=시퀀스 꼬리" 가정이라 흩어진 M에 부정확; runner가 쓴 positions를 그대로 사용 (연속 케이스 항등) |
| 플래그 | `v1/attention/backend.py` | `CommonAttentionMetadata.epic_sparse_logical_q` |

안전 게이팅: `epic_sparse_forward=True`이면 FlexAttention + enforce_eager를
**검증하고 아니면 즉시 ValueError**(fail-closed). 마지막 prompt 토큰은 항상 M에
포함(불변식) → `logits_indices=qsl[1:]-1`로 sampling 자동 정합.

## 4. A+C+B 동작 흐름 (sparse on)

```
scheduler : A+C+B 매칭 → external=|A|+|B| → 블록 N개 할당 → num_scheduled=|M| → 단독 배치
worker    : start_load_kv — B의 KV를 PIC 재회전해 B 위치 블록에 scatter (A는 native)
runner    : positions = M의 실제 logical 위치 (C 구간 + B의 link 위치 + N-1)
            → input_ids gather / slot_mapping 자동 정합, seq_lens=N
forward   : M개 행만 QKV/MLP/attention (계산 ∝ |M|)
flex      : logical_q_positions로 M의 진짜 위치 기준 causal attention
            (모든 KV 위치가 적재/계산되어 있어 causal로 충분)
sampling  : 마지막 M 행 = 위치 N-1 → 정상 decode 진입 (num_computed=N 수렴)
```

## 5. 사용법

```bash
# Phase 1 (위치독립 prefix 재사용, 어떤 backend든)
--kv-transfer-config '{"kv_connector":"EpicConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"epic_chunk_size":256}}'

# Phase 2b (sparse-forward; FlexAttention + eager 필수)
# v0.22에서 VLLM_ATTENTION_BACKEND env var는 제거됨 -> --attention-backend 사용.
... --attention-backend FLEX_ATTENTION --enforce-eager \
--kv-transfer-config '{"kv_connector":"EpicConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"epic_chunk_size":256,
    "epic_sparse_forward":true,"epic_fusion_mask":true,"epic_link_tokens":8}}'
```

| extra_config | 기본 | 의미 |
|---|---|---|
| `epic_chunk_size` | 256 | chunk 토큰 수 (block_size 배수 강제) |
| `epic_store_max_bytes` | (store 기본) | CPU store LRU 예산 |
| `epic_fusion_mask` | false | FlexAttention mask_mod 주입 |
| `epic_sparse_forward` | false | sparse-forward (M만 forward) |
| `epic_link_tokens` | 8 | non-prefix chunk당 재계산할 경계 토큰 수 |

## 6. 검증 상태

- **CPU (전부 통과)**: epic unit/functional 테스트 + vanilla 회귀
  (`tests/v1/core/test_scheduler.py` 97, `test_prefix_caching.py` 61).
  핵심 수학(PIC delta = direct RoPE 동치), 회계(스텝 후 num_computed=N 정확),
  플래그 off 무흔적(실 Scheduler 인스턴스로 확인) 포함.
  ```bash
  .venv/bin/python -m pytest tests/v1/kv_connector/unit/epic/ tests/v1/core/test_scheduler.py -q
  ```
- **GPU (미실행)**: 수치 정확성·재컴파일 무발화는 GPU 검증 필요.
  절차는 `PHASE2.md`의 4단계 체크리스트(무흔적 → fail-closed ×2 → 실 sparse 런)와
  `tests/v1/kv_connector/unit/epic/gpu_smoke.py` 참조.

## 7. 남은 작업

- GPU end-to-end (PHASE2.md 체크리스트) → MITIGATED 위험들의 RESOLVED 승격.
- PIECEWISE cudagraph 완화 (현재 eager 강제).
- sparse 혼합 배치 (현재 단독 배치; fusion_mask 2-D 확장 필요).
- CacheBlend 전략 구현 (`IdentityAlignment` + 동적 recompute의 2-phase pre-pass —
  인터페이스는 `reuse_strategy.py`에 준비됨).
- rope_scaling 확장 (현재 linear만; llama3/YaRN 미지원 시 로드만 비활성).

## 8. 원본 EPIC과의 대응표

| EPIC 원본 (vLLM 0.7, V0) | 본 구현 (V1) |
|---|---|
| `cache_fuse_metadata` dict를 forward 시그니처 관통 | `EpicConnectorMetadata` + forward_context |
| dense `old_kvs`를 모델 객체에 부착 | `EpicChunkStore`(CPU) → paged block scatter |
| `fake_q` 더미로 재-rotary | `PICRotator` delta rotation (K 전용) |
| forward 중간 토큰 슬라이싱 (status 1→2) | **M pre-forward 동결** (1급 불변식) |
| xformers attn_bias | FlexAttention `mask_mod` (텐서 lookup) |
| `model_runner`의 selected_token_indices 보정 | M 마지막=N-1 불변식으로 자동 정합 |
| 벤치마크가 엔진 내부 직접 조작 | connector 수명주기 (scheduler 통합) |
| custom op 비활성화 (eager 전용) | 기본 경로 무수정, sparse만 eager (완화 TODO) |
