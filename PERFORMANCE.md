# Fused NVFP4-to-FP8 PV baseline

## Configuration

B300 / SM103, FlashInfer 0.6.17, CUTLASS DSL 4.6.2,
PyTorch 2.13.0+cu130. H=128, Q=1, page=64, latent=512,
RoPE=64 and BF16 output. Q/KV quantization and preparation are excluded;
online P conversion, V dequantization, decode and split-K reduction are included.
Both versions use the original FP8 split-K heuristic and identical inputs.

Measurements below were collected on 2026-09-30 with CUDA Graph + CUPTI.
Method/cache-mode order alternates between rounds. Each result is the median
of seven per-round medians, with 40 samples per round. Cold L2 uses
FlashInfer's cold-cache timing mode; there is no silent CUDA-event fallback.

## Retained optimization

- Preserve the 16-byte alignment of expanded V shared-memory stores and the
  four-byte alignment of packed scale loads.
- Convert packed E2M1 through half2 multiplication into packed E4M3. Products
  are exact in FP16, retaining the original final FP8 rounding.

Only the V loader changed. MMA types, KV format, warp counts, lane mapping,
stage counts, split-K and reduction are unchanged. An alignment-only profile
at 16×16384 reduced shared-store bank conflicts from 14.67 million to
1.35 million. TMA staging/in-place expansion, lane remapping and static loader
unrolling did not give a stable additional improvement and are not included.

Each cell is unoptimized → optimized latency in µs (latency change).

| Batch × Seq | Splits | Warm L2 | Cold L2 |
|---|---:|---:|---:|
| 4 × 4096 | 18 | 23.680 → 20.064 (-15.3%) | 25.937 → 22.961 (-11.5%) |
| 4 × 65536 | 18 | 112.961 → 67.793 (-40.0%) | 114.066 → 70.193 (-38.5%) |
| 16 × 16384 | 4 | 119.906 → 69.473 (-42.1%) | 120.945 → 71.648 (-40.8%) |
| 32 × 65536 | 2 | 844.710 → 441.939 (-47.7%) | 845.159 → 443.172 (-47.6%) |
| 64 × 65536 | 1 | 1652.623 → 855.384 (-48.2%) | 1653.147 → 857.415 (-48.1%) |

These are before/after results for this fused baseline, not speedups over
official FP8 or the native single-payload method. They do not establish a
theoretical performance limit or end-to-end serving performance.

## Prior validation of the optimized core

- Original and optimized decode outputs match bit-for-bit.
- Sixteen independent-reference cases cover head counts, page permutations,
  ragged lengths, empty splits, sharper scores and mixed KV scales.
  Maximum relative L2 error against the online-softmax/FP8-P reference was
  0.00168207; this checks implementation correctness, not full-precision
  attention's quantization error.
- CUDA Graph replay and updates to captured input buffers were exact.
- Exhaustive packed conversion covered 260,096 values, including signed zero
  and saturation. Output bytes matched the independent reference exactly.
- Device memcheck passed three tail/empty-split cases with zero errors.
  CUDA API capability-probe reporting alone was disabled.

The published package keeps the same compiled kernel bodies. Package-relative
imports, public input preparation and documentation are separated from the
core to keep this baseline reproducible without old scratch paths.

## Independent package verification (2026-10-09)

The wheel was built with `pip wheel --no-deps --no-build-isolation` and installed
into an isolated target directory, without replacing runtime dependencies.
All 20 tests passed on a second B300 host, including all 16 decode cases and
exhaustive packed conversion. The example produced finite BF16 output of shape
`[1, 64, 512]`. Missing/wrong FlashInfer versions were rejected before CUDA
imports; the wheel contains no runtime dependency requirements.

The installed wheel was paired with the previous optimized implementation using
identical inputs and splits. Q/K quantization and attention output were bit-exact.
Five rounds × 40 CUPTI + CUDA Graph samples per cache mode, alternating method
and mode order, produced the following latency pairs (old optimized → package,
µs):

| Batch × Seq | Warm L2 | Cold L2 |
|---|---:|---:|
| 4 × 4096 | 20.128 → 20.144 | 22.880 → 22.944 |
| 4 × 65536 | 67.520 → 67.537 | 70.064 → 69.936 |
| 16 × 16384 | 69.217 → 69.296 | 71.377 → 71.680 |
| 64 × 65536 | 854.182 → 855.862 | 856.087 → 857.526 |

Differences ranged from -0.18% to +0.43%; no material packaging regression was
observed. These small differences are not claimed as improvements.

The installed wheel also passed device memcheck on the 192-token tail,
single-valid-token/empty-split and 4160-token ragged cases, with zero errors.
Only CUDA API capability-probe reporting was disabled; device checks remained
enabled. The container's existing runtime packages were not replaced.
