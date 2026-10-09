# FlashInfer MLA FP4 Fused Dequantization

An experimental CuTe DSL MLA decode package adapted from FlashInfer's modular
FP8 implementation. QK uses native block-scaled NVFP4 MMA. For PV, the same
NVFP4 latent is dequantized into FP8 shared memory inside the decode kernel,
then consumed by FP8 × FP8 MMA. No second KV payload or global FP8 temporary
cache is required; RoPE remains E4M3.

This is a research baseline, separate from the per-token-scaled implementation
on `main` and the shared-payload/dual-scale implementation on
`native_nvfp4_qk`. It does not patch FlashInfer or integrate with SGLang.
The three Python packages can coexist.

## Install

In an existing Linux CUDA environment with FlashInfer 0.6.17, PyTorch, Triton
and CuTe DSL installed:

```bash
git clone --branch fused_dequant_pv --single-branch \
  https://github.com/LoCinder/flashinfer_mla_fp4.git flashinfer_mla_fp4_fused_dequant
cd flashinfer_mla_fp4_fused_dequant
pip install -e .
```

FlashInfer 0.6.17 is checked at import time. Installation does not install,
upgrade or downgrade runtime dependencies. The optimization measurements used
B300 (SM103), CUTLASS DSL 4.6.2 and PyTorch 2.13 with CUDA 13.0.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python examples/decode.py
```

The example quantizes Q and KV before constructing the runner:

```python
from flashinfer_mla_fp4_fused_dequant import Runner

runner = Runner(q, q_scales, q_rope, payload, kv_scales, kv_rope,
                lengths, splits=8)
output = runner()
```

Inputs are contiguous CUDA tensors:

- Q: `[B, H, 256]` uint8 packed E2M1; Q scales: `[B, H, 32]` E4M3.
- KV payload: `[B, S, 256]` uint8; KV scales: `[B, S, 32]` E4M3.
- Q/KV RoPE: `[B, H, 64]` / `[B, S, 64]` E4M3.
- Valid lengths: `[B]` int32, between 1 and S.
- Output: `[B, H, 512]` BF16, backed by a reused output buffer.

Each latent block of 16 values has an E4M3 scale. Global scales are fixed to 1.
The helper `quantization.quantize.quantize` prepares this format from
FP32/BF16/FP8 inputs. Scales are linear by default; prepacked page64 KV scales
can be passed with `input_sf_packed=True`.

The constructor compiles, packs scales, allocates workspace/output and creates
a sequential page table. Construct it outside CUDA Graph capture or timing;
repeated `runner()` calls reuse these buffers. This is an operator-only
research interface, not a serving `plan()/run()` wrapper.

Scope: Q length 1, latent 512, RoPE 64, H=8/16/32/64/128, page size 64.
Storage must be page-padded; valid lengths mask padded tokens. Validation and
performance measurements target B300; no SM120 support is claimed.

## Tests

With pytest installed, select one idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_cute_dsl_mla_fp4.py
```

The suite covers quantization for three input dtypes, 16 decode/reference cases,
page permutations, ragged lengths, empty splits, H=8/16/32/64/128, 30 CUDA Graph
replays and updates to captured input buffers. An exhaustive packed-conversion
test checks all E2M1 bytes against all nonnegative finite E4M3 scales.

The GPU guard refuses an occupied GPU and stops only its own process if another
workload appears.

## Performance

The optimized V loader preserves vectorized shared-memory stores and uses
packed half2 conversion/multiplication. QK uses paged payload/scale TMA;
PV dequantization loads that same packed cache into registers and writes FP8 V
to shared memory. The original split-K heuristic and PDL reduction are retained.

Q/KV preparation, scale packing and compilation are excluded from decode
timings; online P conversion, V dequantization, attention and reduction are
included. Measurements and validation details are in [PERFORMANCE.md](PERFORMANCE.md).

## Layout

```text
flashinfer_mla_fp4_fused_dequant/
  __init__.py
  mla_config_fp4.py
  collective_builder_fp4.py
  mla_decode_fp4.py
  roles/             # Q/K load, V dequantization, MMA, compute, correction
  wrappers/
    batch_mla_fp4.py  # Runner
  quantization/      # Q/KV quantization and scale packing
tests/               # numerical references and correctness tests
examples/            # small runnable example
```

Only the optimized fused-dequantization route is included. Historical experiment
scripts, alternative variants and raw profiling artifacts are not installed.
