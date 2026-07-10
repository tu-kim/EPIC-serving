# fileKV Prefetch (feature/prefetch)

Agentic-serving latency hiding: stage the next turn's fileKV chunks on the
**designated worker's GPU** before the turn starts, so the prefill's chunk
loads skip the CPU→GPU copy.

## The loop

```
turn t decode ──► LLM emits tool calls ──► parse AT EMIT TIME
                    <tool_call><function=read>
                    <parameter=filePath>src/foo.py</parameter>...

frontend (dynamo scheduler)
  │  decides which worker/replica serves turn t+1  ──►  dst_worker
  ▼
FileKVPrefetcher.prefetch_for_output(llm_output, dst_worker)
  │  render (byte-faithful!) → tokenize → chunk-grid pad → content hashes
  ▼
EpicConnector.enqueue_prefetch(chunk_hashes=..., dst_worker=...)   [scheduler role]
  │  filters to store-known hashes, queues (thread-safe)
  ▼
build_connector_meta            drains queue → EpicReqPrefetch on the metadata
  │                             (rides an ONGOING decode step; no extra step)
  ▼
worker start_load_kv → _consume_prefetches
  │  dst_worker ∈ {-1, my epic_worker_id} → EpicGpuStagingStore.stage()
  │  async H2D on a DEDICATED side stream + ready event
  ▼
turn t+1 prefill → _load_chunk
     staging hit  → GPU tensors direct (wait_event only; H2D already paid)
     staging miss → CPU fileKV store, exactly as before
```

## Configuration (`kv_connector_extra_config`)

| key | default | meaning |
|---|---|---|
| `epic_prefetch_gpu_bytes` | `0` (off) | GPU staging budget per worker; `0` keeps every prefetch path inert |
| `epic_worker_id` | `-1` | this worker/replica's identity as the frontend scheduler knows it; directives with `dst_worker == -1` match everyone |

## Contracts / invariants

* **Prefetch is a hint.** Every failure mode (unknown hash, store eviction,
  staging-budget eviction, wrong worker) degrades to the pre-existing CPU
  load path — never an error on the serving path.
* **Byte fidelity is the client's job** (`render_fn`): the rendered text must
  equal the future tool-result text byte-for-byte (line-number prefixes,
  wrappers, line ranges) and be tokenized/padded on the same 256-token chunk
  grid, or the content hashes miss (see DESIGN.md chunk-grid convention).
* **Only stored chunks can be staged**: `enqueue_prefetch` drops hashes the
  scheduler index does not know. Warm the files first (a normal prefill
  request over the rendered file text) — prefetch then hides the *reload*
  latency of every subsequent turn.
* **Correctness-neutral**: staged tensors are the same bytes as the CPU
  store's; PIC rotation and scatter are unchanged. `stage()` copies on a side
  stream and `get()` orders the consumer stream on the copy event
  (device-side dependency, no host sync).
* **Multi-worker**: "worker" here is the frontend's placement unit (an engine
  replica). Within one TP engine every rank stages its own KV shard from its
  own store; the `dst_worker` filter selects the replica, not the TP rank.
* Staging survives CPU-store eviction: `start_load_kv` falls back to the
  staged copy when the CPU store no longer holds a directive'd chunk
  (`StagedChunk` is duck-compatible with `StoredChunk`).

## Out of scope (deliberate)

* The dynamo→engine transport for `enqueue_prefetch` (deployment-specific
  RPC; the method is thread-safe so any frontend thread may call it).
* Pre-rotation at stage time: PIC needs the chunk's position in the *next*
  prompt, which is unknown until the request arrives; rotation stays at load
  time (it is GPU-side and cheap relative to H2D).
* Cross-engine staging transfer / eviction coordination.
