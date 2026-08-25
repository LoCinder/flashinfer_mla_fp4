# flashinfer_mla_fp4

A standalone FlashInfer sidecar providing an FP4 MLA decode kernel adapted
from FlashInfer's modular FP8 `cutedsl_mla` implementation. Its KV cache stores
a 512-dimensional E2M1 latent vector with one FP32 scale per token and a
separate 64-dimensional E4M3 RoPE vector. This is a custom cache format, not
the standard NVFP4 block-scaled format. The package does not patch or replace
the installed FlashInfer package.

## Requirements

- Python 3.10
- `flashinfer-python==0.6.17`
- A tcgen05-capable NVIDIA Blackwell GPU; validated on B300 (SM103)
- A working CUDA, PyTorch, and NVIDIA CuTe DSL environment

The FlashInfer version is checked at import time because this package inherits
internal FlashInfer CuTe DSL classes.

## Install

```bash
git clone https://github.com/LoCinder/flashinfer_mla_fp4.git
cd flashinfer_mla_fp4
pip install -e .
```

## Cache format

The currently supported configuration is:

- Latent dimension: 512 packed E2M1 values
- RoPE dimension: 64 E4M3 values
- Scale: one FP32 value per token
- Query length: 1; longer queries can be added via `fold_sq`
- Query: E4M3 with shape `[batch, 1, heads, 576]`
- Output: BF16 by default with shape `[batch, 1, heads, 512]`

The packed cache has shape `[pages, page_size, 320]` and dtype `torch.uint8`:
the first 256 bytes contain the packed latent values and the final 64 bytes
contain the E4M3 RoPE values. The scale tensor has shape
`[pages, page_size, 1]` and dtype `torch.float32`.

## Usage

Assuming the query, packed cache, scale tensor, page table, and sequence
lengths have already been prepared:

```python
import torch

from flashinfer_mla_fp4 import BatchMLADecodeCuteDSLFP4Wrapper

workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
wrapper = BatchMLADecodeCuteDSLFP4Wrapper(workspace)
wrapper.plan(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    num_heads=q.shape[2],
    page_size=64,
    q_dtype=torch.float8_e4m3fn,
    out_dtype=torch.bfloat16,
    is_var_seq=True,
    kv_dtype=torch.uint8,
)

output = wrapper.run(
    q=q,
    kv_cache=kv_cache,
    kv_cache_sf=kv_cache_sf,
    block_tables=block_tables,
    seq_lens=seq_lens,
    max_seq_len=int(seq_lens.max().item()),
    softmax_scale=1.0 / (512**0.5),
)
```

The first `plan` call compiles the required specialization. Later wrappers in
the same process reuse the compilation cache.

This package contains only the FP4 MLA decode path. KV cache quantization,
cache writes, prefill handling, and SGLang backend routing remain the caller's
responsibility.

## Test

```bash
pytest -q -s tests/test_cute_dsl_mla_fp4.py
```

The test suite compares FP4 output against the corresponding materialized FP8
cache across multiple shapes and reports one small warm-L2 performance point.

## Results

The FP8 baseline in this section uses FlashInfer's `cutedsl_mla` decode path.

### Kernel accuracy

Measured on Kimi-K2.6 with KV length 1024 and query length 1. The metrics are
averaged across all MLA layers relative to an FP8 baseline. This implementation
has lower attention-output error than dequantizing NVFP4 to FP8, with slightly
higher KV write error. MXFP4-to-FP8 has the largest output error.

| Path | Mean cosine ↑ | Mean L2 error ↓ | Mean KV L2 error ↓ |
|---|---:|---:|---:|
| Per-token-scaled FP4 (this implementation) | **0.986437** | **0.157104** | 0.114018 |
| NVFP4 → FP8 | 0.985614 | 0.175141 | **0.095618** |
| MXFP4 → FP8 | 0.975980 | 0.233607 | 0.116849 |

### Kernel latency

Measured on one B300 GPU with 64 query heads, page size 64, and cold L2. Each
cell reports FP8 → FP4 latency in microseconds and the FP4 latency change;
negative values mean FP4 is faster.

| Batch \ Seq | 4096 | 8192 | 16384 | 32768 |
|---:|---:|---:|---:|---:|
| 4 | 14.278→13.461 (-5.7%) | 16.317→16.110 (-1.3%) | 19.598→19.810 (+1.1%) | 26.789→26.347 (-1.6%) |
| 8 | 15.914→15.909 (-0.0%) | 19.091→19.392 (+1.6%) | 26.149→25.744 (-1.5%) | 38.854→37.726 (-2.9%) |

### End-to-end validation

We integrated this kernel into our private SGLang fork and validated FP4
KV cache serving on Kimi-K2.6 (DPA=8) and Kimi-K3 (DCP=8). Single-request
latency stayed within ±3% of the FP8 baseline.

GPQA-Diamond accuracy averaged over eight runs:

| Configuration | FP8 baseline | FP4 | Change |
|---|---:|---:|---:|
| Kimi-K2.6 | 90.5% | 89.7% | -0.8 pp |
| Kimi-K3 | 93.5% | 93.1% | -0.4 pp |

## License

Kernel sources are BSD-3-Clause; the FlashInfer-derived test is Apache-2.0.
