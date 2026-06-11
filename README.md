<!-- markdownlint-disable MD001 MD041 -->

# EPIC-serving — Non-contiguous KV Cache Reuse on vLLM

This repository is **vLLM v0.22.1 + EPIC** ([arXiv:2410.15332](https://arxiv.org/pdf/2410.15332)):
a migration of the EPIC position-independent KV caching prototype
([DerekHJH/epic](https://github.com/DerekHJH/epic), vLLM 0.7.0 fork) into the
vLLM V1 engine as a proper `KVConnectorBase_V1` integration. Given a prompt
`A + C + B` where chunks `A`/`B` are cached from earlier requests, it reuses
both — including the **non-prefix** chunk `B` — re-rotating cached keys to
their new positions (PIC) and forwarding only the new/link tokens
(sparse-forward).

- **Implementation & docs**: [`vllm/distributed/kv_transfer/kv_connector/v1/epic/`](vllm/distributed/kv_transfer/kv_connector/v1/epic/) — see its [README](vllm/distributed/kv_transfer/kv_connector/v1/epic/README.md), [DESIGN](vllm/distributed/kv_transfer/kv_connector/v1/epic/DESIGN.md), [PHASE2](vllm/distributed/kv_transfer/kv_connector/v1/epic/PHASE2.md)
- **Tests**: [`tests/v1/kv_connector/unit/epic/`](tests/v1/kv_connector/unit/epic/) (80 CPU tests + `gpu_smoke.py`)
- **Benchmarks**: [`benchmarks/epic_reuse/`](benchmarks/epic_reuse/) — full / prefix-only / reuse-only / epic@k over |A|,|C|,|B| sweeps, synthetic-needle + SQuAD/HotpotQA accuracy
- **All EPIC changes vs vanilla**: `git diff <baseline-commit> main` (baseline = first commit)

## Build & Run

### 1. Install (build)

All EPIC changes are **Python-only**, so you can reuse the official v0.22.1
compiled kernels instead of a full CUDA build:

```bash
git clone https://github.com/tu-kim/EPIC-serving.git && cd EPIC-serving
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

If the precompiled-wheel lookup fails (this repo's commits are not upstream
vLLM commits), point it at the official v0.22.1 wheel explicitly:

```bash
uv pip download vllm==0.22.1 --no-deps -d /tmp/vllm-wheel
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=$(ls /tmp/vllm-wheel/vllm-*.whl) \
uv pip install -e . --torch-backend=auto
```

### 2. CPU tests (no GPU needed)

The EPIC unit/functional tests run on CPU. They need `pytest` plus `tblib`
(used by the repo-wide `tests/conftest.py`); `torch` / `transformers` / `numpy`
already come with the install in step 1:

```bash
uv pip install pytest tblib

.venv/bin/python -m pytest tests/v1/kv_connector/unit/epic/ -q   # 80 EPIC tests
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py -q    # vanilla regression
```

> The `SwigPyObject`/`torch.jit.script_method`/`BPE.__init__` lines in the
> output are harmless third-party `DeprecationWarning`s, not test failures.
> Silence them with `-p no:cacheprovider -W ignore::DeprecationWarning` if
> you want quiet output.

### 3. GPU functional verification (run this first on a CUDA box)

```bash
VLLM_ATTENTION_BACKEND=FLEX_ATTENTION .venv/bin/python \
  tests/v1/kv_connector/unit/epic/gpu_smoke.py --model meta-llama/Llama-3.2-1B-Instruct
```

Steps: 1) no-trace (connector on, sparse off ≡ baseline) → 2,3) fail-closed
config gates → 4) real sparse run vs dense comparison.

### 4. Serving with EPIC

```bash
# Phase 1 only: position-independent prefix reuse (any backend)
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --kv-transfer-config '{"kv_connector":"EpicConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"epic_chunk_size":256}}'

# Full EPIC: non-prefix reuse + sparse-forward (FlexAttention + eager required)
VLLM_ATTENTION_BACKEND=FLEX_ATTENTION .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B-Instruct --enforce-eager \
  --kv-transfer-config '{"kv_connector":"EpicConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"epic_chunk_size":256,"epic_sparse_forward":true,
      "epic_fusion_mask":true,"epic_link_tokens":8}}'
```

### 5. Benchmarks (accuracy + TTFT, |A|/|C|/|B|/k sweeps)

```bash
# data: synthetic needle (offline) or --data-mode hf --hf-dataset squad
.venv/bin/python -m benchmarks.epic_reuse.data_prep \
  --model meta-llama/Llama-3.2-1B-Instruct --out epic_bench.jsonl
# perf: full / prefix / reuse-only(no recompute) / epic@k
VLLM_LOGGING_LEVEL=WARNING .venv/bin/python -m benchmarks.epic_reuse.bench_perf \
  --data epic_bench.jsonl --model meta-llama/Llama-3.2-1B-Instruct \
  --modes full,prefix,reuse-only,epic --link-sweep 0,8,32 --out perf.csv
# accuracy vs dense reference
.venv/bin/python -m benchmarks.epic_reuse.bench_accuracy \
  --data epic_bench.jsonl --model meta-llama/Llama-3.2-1B-Instruct \
  --modes reuse-only,epic --link-sweep 0,8,32 --out acc.csv
# plots (TTFT/speedup vs |B|, accuracy-vs-k tradeoff, ...)
.venv/bin/python -m benchmarks.epic_reuse.plot_results \
  --perf perf.csv --acc acc.csv --fix-a 0 --fix-c 256 --fix-b 512 --outdir plots/
```

See [`benchmarks/epic_reuse/README.md`](benchmarks/epic_reuse/README.md) for
grid options and result interpretation.

The original vLLM README follows below.

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
