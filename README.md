# FlashInfer MLA FP4 Native

An experimental CuTe DSL MLA decode package adapted from FlashInfer's modular
FP8 implementation. QK uses native block-scaled FP4 MMA; PV retains the
per-token-scaled FP4 path. K and V share one E2M1 payload, with fitted E4M3
block16 scales for K and one FP32 scale per token for V. RoPE remains E4M3.
This dual-scale cache is not the standard NVFP4 KV-cache format.

This package uses the latest optimized research kernel through its `Runner`
interface, with the retained optimizations enabled directly. It does not patch
FlashInfer or integrate with SGLang. It can coexist
with the original `flashinfer_mla_fp4` package.

## Install

In an existing Linux CUDA environment with FlashInfer 0.6.17, PyTorch, Triton,
and CuTe DSL installed:

```bash
git clone --branch native_nvfp4_qk --single-branch \
  https://github.com/LoCinder/flashinfer_mla_fp4.git flashinfer_mla_fp4_native
cd flashinfer_mla_fp4_native
pip install -e .
```

FlashInfer 0.6.17 is checked at import time. Installation does not install,
upgrade, or downgrade runtime dependencies. The tested optimization environment
used B300 (SM103), CUTLASS DSL 4.6.2, and PyTorch 2.13 with CUDA 13.0.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python examples/decode.py
```

The example creates small synthetic inputs, prepares Q and KV before constructing
the runner, and executes the optimized page64 path:

```python
from flashinfer_mla_fp4_native import Runner

runner = Runner(q, q_scales, q_rope, payload, k_scales, kv_rope,
                token_scales, lengths, splits=8, input_sf_packed=True)
output = runner()
```

Inputs are contiguous CUDA tensors. Q is `[B, H, 256]` uint8 packed E2M1;
Q scales are `[B, H, 32]` E4M3 in linear layout. KV payload is `[B, S, 256]`
uint8, token scales are `[B, S, 1]` FP32, and K scales have logical shape
`[B, S, 32]` with the page64 byte permutation when `input_sf_packed=True`.
Q/KV RoPE tensors are `[B, H, 64]` / `[B, S, 64]` E4M3. Lengths are `[B]`
int32. Output is `[B, H, 512]` BF16, backed by a reused output buffer.

The constructor compiles, packs scales as needed, allocates workspace/output,
and creates a sequential page table. Construct it outside CUDA Graph capture
or timing; repeated `runner()` calls reuse these buffers. This is the research
interface, not a serving `plan()/run()` wrapper.

Scope: Q length 1, latent 512, RoPE 64, H=8/16/32/64/128, page size 64.
Q/K block-scale TMA, the independent scale producer, four-stage token-scale
queue, and fused QK/RoPE path are enabled directly. Split-K also enables
programmatic dependent launch (PDL) for the reduction; no optimization flags
are needed. Global scales default to 1, and their product must be 1 for fused
latent/RoPE accumulation. The quantization helpers use the same default.
The input sequence storage must be page-padded; lengths mask padded tokens.
SM120 is not supported; no SM100 performance claim is made.

## Tests

With pytest installed, select one idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_cute_dsl_mla_fp4.py
```

The 19 cases cover quantization, independent numerical references,
head counts, split-K, shuffled pages, ragged lengths, and CUDA Graph replay.
Only the current optimized configuration is tested. The test GPU guard refuses
an occupied GPU and stops its own process if another workload appears.
All 19 cases passed on B300 using an isolated installation of the built wheel.

The current optimized core also passed all 19 cases under memcheck with CUDA
API-probe reporting disabled; no device-memory errors were reported. See the
performance report for the API-probe warnings and test details. Earlier
synccheck barrier errors also reproduced on the frozen native and official FP8
controls remain unresolved. These results do not establish production readiness.

## Performance

The current kernel uses a four-stage K pipeline, rebalanced warp register
budgets, independent K-scale TMEM slots, and early release of consumed
token-scale pipeline slots. No alternative execution versions are included.

A full B300 comparison against FlashInfer 0.6.17's FP8 `cutedsl_mla` covered
B=2/4/8/16/32/64/128, S=16K/32K/64K/128K/256K, and H=64/128, with CUDA Graph
and both warm/cold L2. Both kernels used the same official FP8 split-K heuristic.
Native FP4 was faster at every tested B>=16 point; small-batch/short-context
cases could be slower. Across the grid, geometric-mean latency was about 11%
lower warm and 13–14% lower cold. Some largest-workload points showed substantial
timing variability; independent rechecks retained the performance advantage.

Q quantization, KV writing, scale packing and compilation are excluded from
decode timings. No end-to-end accuracy claim is made for this version.
Measurement details and optimization history are in [PERFORMANCE.md](PERFORMANCE.md).

## Layout

```text
flashinfer_mla_fp4_native/
  __init__.py
  mla_config_fp4.py
  collective_builder_fp4.py
  mla_decode_fp4.py
  roles/             # load, MMA, compute, correction
  wrappers/
    batch_mla_fp4.py  # Runner (same interface)
  quantization/      # Q/KV quantization and scale packing
tests/               # correctness suite and numerical references
examples/            # small runnable example
```

The frozen baselines, split-search scripts and raw performance records remain
in the separate research checkout; they are not installed with this package.
