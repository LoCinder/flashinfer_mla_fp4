# Performance measurements

## Current version vs FP8: full CUDA Graph grid (2026-09-29)

B300 GPU 6, FlashInfer 0.6.17, CUTLASS DSL 4.6.2, PyTorch 2.13.0+cu130,
CUPTI 13.0.1. Q=1, page=64, latent=512, RoPE=64, BF16 output.
The grid covers B=2/4/8/16/32/64/128, S=16384/32768/65536/131072/262144,
and H=64/128: 70 shapes, each measured with warm and cold L2.

Each implementation/cache mode used five rounds of 50 CUDA Graph replays,
alternating implementation and cache-mode order. Values are medians of round
medians. CUPTI measures the full decode kernel span, including reduction.
Cold L2 flushes 2x the 126 MiB L2 before each replay, outside the timed span.
Warm L2 reuses addresses without flushing; large KV inputs need not fit in L2.
Q/KV preparation, compilation, allocation and planning are excluded.

FP8 uses its official split heuristic, with the identical count explicitly
passed to the native Runner: 32/18/9/4/2/1/1 for the seven batch sizes above.
PDL is enabled on both for split>1. No new split tuning was applied.

| Heads | Cache | Geometric-mean latency change vs FP8 | Native faster |
|---:|:---|---:|---:|
| 64 | Warm L2 | -10.7% | 29/35 |
| 64 | Cold L2 | -14.0% | 30/35 |
| 128 | Warm L2 | -10.8% | 28/35 |
| 128 | Cold L2 | -13.3% | 30/35 |

Every B>=16 point was faster, with initial latency reductions of roughly
8–29%. Small-batch/short-context points regressed by up to 37%.
B64/S262144 and B128/S131072 or S262144 showed >5% round-to-round variation
in at least one measurement (maximum initial round spread 22.3%). An
independent recheck of these six shapes plus two stable controls still favored
native in every pair, but exact percentages are not stable to 1–2%.

All 70 shapes passed finite-output and bitwise eager-vs-graph checks for each
implementation. This was not a new FP4-vs-FP8 accuracy evaluation. Peak allocated
memory was 29.43 GiB. The runtime hash matches the accepted round-4 version below.
The complete tables, 70,000 raw samples, rechecks and benchmark script are kept
in the separate local evidence directory
`flashinfer_mla_fp4_native_perf_full_20260929/`; they are not part of this package.

## Optimization round 4: medium/large loads (2026-09-29)

Retain one small scheduling change in `roles/mla_compute_fp4.py`: release the
token-scale pipeline slot immediately after its synchronous SMEM-to-register
copy, instead of holding it until after the P store. Softmax and P scaling use
the register copy. This moves one line and adds one comment; the arithmetic,
cache format, API, stage counts, register budgets, splits and PDL are unchanged.

Against the accepted round-3 version below, two independent comparisons used
H=64, Q=1, page=64 on an idle B300, CUPTI + CUDA Graph, warm/cold L2, fixed
splits and rotating profile order. Confirmation used 7 rounds x 60 samples;
the recheck reversed starting orders and used 7 rounds x 80. Both included an
identical-source A/A control. Q and KV preparation remain excluded.

| Batch x Seq | Split | Warm latency reduction across both passes | Cold latency reduction |
|---|---:|---:|---:|
| 16 x 32768 | 4 | 0.71–1.01% | 0.31–0.88% |
| 16 x 65536 | 4 | 0.51–0.94% | 0.71–0.73% |
| 32 x 16384 | 2 | 0.45–0.72% | 0.18–0.66% |
| 32 x 32768 | 2 | 0.56–0.89% | 0.63–0.72% |

B=64/S=32768 and 65536 were effectively flat, overlapping A/A variation.
B=16/S=16384 was noisy, with warm changes from +0.68% to -2.60% and A/A
changes from +1.22% to -1.65%; no 2.6% gain is claimed. Across all seven shapes,
geometric-mean latency reductions were 0.37%/0.76% warm and 0.44%/0.49% cold
in the two passes. Small B=4/8 guardrails initially showed 1–2% regressions that
did not repeat consistently with reversed ordering. This is a modest
medium-load improvement, not a universal speedup or a large-load breakthrough.

Page-index reuse, in-place P scaling, delayed P acquisition/scale loading,
deeper P buffering and reordered scale TMA did not yield a better repeatable
tradeoff and were not retained. No alternate paths or switches are added.

Instruction-level profiling refined the bottleneck assessment: many hot
long-scoreboard PCs are pipeline barrier waits, not direct global scale reads.
The baseline large-load diagnostic showed 58.67% maximum-bandwidth utilization,
73.24% SM busy and no spills, with one CTA per SM. This motivates investigating
compute/MMA/correction balance rather than assuming scale bandwidth is the
dominant bottleneck; sampled wait percentages are not recoverable wall time.

All paired outputs were bitwise equal, including CUDA Graph replay. Both the
candidate and an isolated installation of the final wheel passed all 19
correctness cases under memcheck with zero device-memory errors, using
`--report-api-errors no`. Earlier API-probe/synccheck caveats below remain.
Local and installed runtime SHA256:
`5cbba1527de1c8c9ca8bb36387af3b4e953a7036068b8ef51a77a35e24cae76b`.

Full tables, controls, rejected experiments, profiler exports and source
snapshots are in the sibling
`flashinfer_mla_fp4_native_perf_round4_20260929/REPORT.md` and accompanying files.
Experiments and the final package are isolated in `/tmp/native_round4_20260929`
in the container. No commit or push was made.

## Optimization round 3: CUDA Graph (2026-09-29)

This round compares against the cleanup-recovery version below, with PDL already
enabled for split-K. The retained changes affect two runtime files, adding nine
net source lines. The Runner API, quantizers, cache format, arithmetic, split
counts and PDL behavior are unchanged. No experimental switches are added.

- Increase the K pipeline from three to four stages, leaving V at two stages.
- Change the softmax/correction/other warp register budgets from 216/208/80 to
  208/224/64. The warp assignment and warp count stay the same.
- Copy the four K block-scale slices into disjoint TMEM slots before the first
  latent QK MMA, instead of overwriting one slot between QK stages. This changes
  scheduling, not the number of copies or the transferred bytes. It uses space
  inside the existing TMEM allocation.

Measurements: B300 GPU 6, H=64, Q=1, page=64, CUPTI, one full decode per CUDA
Graph, fixed split counts, shared inputs, and rotating timing order. Warm L2
means replay without an explicit cache flush; it does not imply that large
inputs fit in L2. Cold L2 explicitly flushes between measured calls. Q
quantization, KV writing, scale packing, compilation and allocation are excluded.
Clocks and power limits were not changed. Software versions are unchanged from
round 2 below.

After exploratory ablations, a four-profile confirmation used 7 rounds x 60
samples per shape and mode. An independent baseline/final recheck used 7 rounds
x 80 samples, starting with the opposite profile order. Each value below is
the median of the recheck's round medians; negative changes mean lower latency.

| Batch x Seq | Split | Warm L2: before -> after (us) | Change | Cold L2: before -> after (us) | Change |
|---|---:|---:|---:|---:|---:|
| 4 x 4096 | 8 | 14.753 -> 14.528 | -1.5% | 16.992 -> 16.800 | -1.1% |
| 4 x 32768 | 16 | 24.800 -> 24.320 | -1.9% | 28.960 -> 28.496 | -1.6% |
| 8 x 8192 | 8 | 18.816 -> 18.496 | -1.7% | 21.600 -> 21.344 | -1.2% |
| 16 x 8192 | 4 | 24.832 -> 24.160 | -2.7% | 28.097 -> 27.648 | -1.6% |
| 16 x 32768 | 4 | 60.417 -> 58.257 | -3.6% | 64.705 -> 61.889 | -4.4% |
| 32 x 16384 | 2 | 61.248 -> 59.520 | -2.8% | 65.584 -> 63.009 | -3.9% |
| 64 x 4096 | 1 | 28.944 -> 29.168 | +0.8% | 35.520 -> 34.192 | -3.7% |
| 64 x 32768 | 1 | 176.817 -> 175.538 | -0.7% | 179.025 -> 176.483 | -1.4% |
| 64 x 65536 | 1 | 344.867 -> 342.371 | -0.7% | 344.659 -> 341.012 | -1.1% |

Across nine shapes, geometric-mean latency decreased by **1.67% warm / 2.23%
cold** in this recheck, versus **0.74% warm / 2.11% cold** in the preceding
confirmation. The small-load warm results varied between passes and are not
claimed as robust gains. More reproducible improvements were B=16, S=32768
(2.5–3.6% warm, 4.1–4.4% cold), B=32, S=16384 (2.8–3.1% warm, 3.9% cold),
and B=64, S=4096 (3.7–3.9% cold, with a 0.6–0.8% warm regression).

The resource tradeoff is more shared memory: an Nsight Compute diagnostic on
B=16, S=32768 reported 185.34 -> 209.92 KB of dynamic shared memory per CTA.
Both versions remain limited to one resident CTA per SM at this specialization.
Registers per thread were 168 in both reports, and both had zero measured
local-memory load/store sectors. These measurements do not support attributing
the gain to eliminated register spills. Nsight's profiled durations are not
mixed into the timing table above.

Other experiments were not retained: fewer K/V stages regressed long-context
performance; more V stages and other register allocations did not offer a better
overall tradeoff. Combining four K stages with three V stages failed to launch
and was rejected. A split-only sweep over 1/2/4/8/12/16/24/32 on six reference
shapes found no reason to change the existing benchmark split choices. No new
shape-specific split heuristic was introduced.

Validation: all paired outputs were bitwise identical before/after, including
graph replay. All 19 correctness cases passed, covering H=8/16/32/64/128,
shuffled pages, ragged tails, split/no-split and changing graph inputs. Memcheck
ran all 19 cases with `--report-api-errors no` and reported zero device-memory
errors. The wheel was built without installing dependencies, installed into an
isolated target, and passed the same 19 tests. Its runtime source hash matches
the benchmark candidate and the local package. The earlier API-probe and
synccheck caveats below remain; this round is
not a new end-to-end accuracy or production-readiness certification.

Raw samples, scripts, the frozen baseline and final candidate are in the sibling
`flashinfer_mla_fp4_native_perf_round3_20260929/` directory. Remote experiments
are isolated in `/tmp/native_round3_20260929`. The runtime SHA256 is
`1f1f117c4f591798919391ea415b461cced76e33cdf9ac625dc74c02010fed98`,
versus `ce7f8c10a3b6fbbfda571025147aab0bd672cd17da7eb8a3b6006a4560b5fe0d`
for this round's baseline, using the path/digest convention below.

To repeat the final paired comparison in the same software environment with
CUPTI available, from the evidence directory on an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=candidate:. python bench_round3.py \
  --profiles candidate,base \
  --shapes 4:4096:8,4:32768:16,8:8192:8,16:8192:4,16:32768:4,32:16384:2,64:4096:1,64:32768:1,64:65536:1 \
  --rounds 7 --repeats 80 --output repeat.json
```

## Cleanup follow-up: CUDA Graph (2026-09-29)

The cleanup moved score-consumer advancement from immediately after the score
wait to after softmax. Restore the original position with `wait_and_advance()`
and return the updated consumer explicitly from softmax, so CuTe tracks its
loop-carried state. All other cleanup and split-K PDL changes remain. This
changes one runtime file; no alternate implementation or tuning switch is added.

Code-generation checks on the no-split H=64 specialization confirmed that the
cleanup changed the generated instructions. The restored version has 4,296
static SASS instructions versus 4,304 after cleanup, matching the pre-cleanup
instruction count (not byte-identical code: register allocation differs).
Restoring the entire old compute role as a diagnostic reproduced the original
instruction bytes; that redundant role was not retained.

Primary comparison: B300 GPU 6, H=64, Q=1, page=64, fixed split counts, CUPTI,
one full decode per CUDA Graph, warm and cold L2 measured separately. Q/KV
preparation and compilation are excluded. Versions share inputs and rotate
measurement order. Six shapes used 7 rounds x 60 samples; four points were
rechecked with 12 rounds x 80 samples and an identical-source A/A control.

At the reported heavy-load point, B=64, S=32768, split=1, the final recheck was:

| Mode | Before cleanup (us) | Cleaned (us) | Restored (us) | Restored vs before |
|---|---:|---:|---:|---:|
| CUDA Graph, warm L2 | 177.370 | 177.481 | 177.553 | +0.10% |
| CUDA Graph, cold L2 | 179.914 | 180.201 | 179.770 | -0.08% |

This is recovery to the previous performance level, not a new speedup claim.
The initially observed ~0.5% regression was not fully reproducible. The A/A
control alone differed by 0.14% warm and 0.06% cold at this point. Relative to
the cleaned version, the restoration was essentially flat warm and 0.24%
faster cold in the final pass. It is not universally faster: B=4, S=32768 was
0.9% slower warm and 1.2% slower cold than the cleaned version, while remaining
within 0.8% of the pre-cleanup baseline. Do not treat sub-percent differences as
robust gains.

All paired outputs were bitwise equal, all 19 correctness cases passed, and
memcheck reported zero device-memory errors with `--report-api-errors no`.
The same earlier API-probe/synccheck caveats below still apply. Raw samples,
the full comparison and replay scripts are in the sibling
`flashinfer_mla_fp4_native_cleanup_recovery_20260929/` directory. Container
experiments are isolated in `/tmp/native_recovery_20260929`.

The cleanup-recovery runtime source SHA256 was
`ce7f8c10a3b6fbbfda571025147aab0bd672cd17da7eb8a3b6006a4560b5fe0d`.

## Previous optimization round 2 (before this follow-up)

Date: 2026-09-29. All numbers below compare the package at the start of this
round against the retained changes, not against an older unoptimized prototype
or a newly measured FP8 baseline. Negative changes mean lower latency.

## Retained changes

- Enable programmatic dependent launch (PDL) when split-K is used. This lets
  the reduction prologue overlap with the main-kernel tail. Existing dependency
  waits still protect consumption of the split outputs. No-split launches do
  not enable PDL.
- Remove disabled non-TMA scale loaders, separate QK/RoPE score paths,
  debug switches and unused quantization/reference experiments. Keep just the
  current native block-scaled QK plus per-token-scaled PV path.
- Centralize the already-used stage/register configuration. Preserve the
  score wait after token-scale loads; advance the consumer in the outer loop
  so the CuTe compiler tracks loop-carried pipeline state explicitly.

Runtime source shrank by 349 lines across eight files; numerical references
and their callers shrank by another 20 lines. These are net source-line changes,
excluding this report and README updates. The public Runner interface and cache
layout are unchanged.

## Measurement

- GPU: B300 (SM103), one idle GPU per process; no clock or power settings changed.
- FlashInfer 0.6.17; CUTLASS DSL 4.6.2; PyTorch 2.13.0+cu130;
  Triton 3.7.1; CUDA bindings 13.3.1.
- H=64, page size=64, Q length=1; latent 512, RoPE 64; BF16 output.
- Q is prepared beforehand. Q quantization, KV writing, scale packing,
  allocation and compilation are excluded.
- Both versions run in the same process on the same inputs and fixed split-K
  settings, alternating timing order between rounds.
- Primary measurement: CUPTI without CUDA Graph, with cold L2, 5 warmups and 60 measured calls per
  version per round, 5 rounds. Report the median of round medians.
- CUPTI measures the span from the first kernel start to the last kernel end,
  including the gap between main and reduction, rather than summing overlapping
  kernel durations.
- A separate first pass used 3 rounds × 40 samples. Its geometric-mean latency
  reduction was 5.53%, versus 5.49% in the confirmation pass below.
- Every one of the 20 paired shapes produced bitwise-identical BF16 output.

## Cold-L2 results

| Batch × Seq | Split-K | Before (µs) | After (µs) | Change |
|---|---:|---:|---:|---:|
| 4 × 4096 | 8 | 18.001 | 16.320 | -9.3% |
| 4 × 8192 | 16 | 21.632 | 19.056 | -11.9% |
| 4 × 16384 | 16 | 24.048 | 21.792 | -9.4% |
| 4 × 32768 | 16 | 30.529 | 28.096 | -8.0% |
| 8 × 4096 | 8 | 20.048 | 17.920 | -10.6% |
| 8 × 8192 | 8 | 23.472 | 21.088 | -10.2% |
| 8 × 16384 | 8 | 29.696 | 27.488 | -7.4% |
| 8 × 32768 | 8 | 43.873 | 41.329 | -5.8% |
| 16 × 4096 | 4 | 22.784 | 20.704 | -9.1% |
| 16 × 8192 | 4 | 29.968 | 27.424 | -8.5% |
| 16 × 16384 | 4 | 43.505 | 41.456 | -4.7% |
| 16 × 32768 | 4 | 66.560 | 64.001 | -3.8% |
| 32 × 4096 | 1 | 28.656 | 28.944 | +1.0% |
| 32 × 8192 | 2 | 44.417 | 42.193 | -5.0% |
| 32 × 16384 | 2 | 66.961 | 65.088 | -2.8% |
| 32 × 32768 | 2 | 108.881 | 106.817 | -1.9% |
| 64 × 4096 | 1 | 35.745 | 35.520 | -0.6% |
| 64 × 8192 | 1 | 57.969 | 57.648 | -0.6% |
| 64 × 16384 | 1 | 98.288 | 98.705 | +0.4% |
| 64 × 32768 | 1 | 179.538 | 179.793 | +0.1% |

Across all 20 shapes, geometric-mean latency decreased by **5.49%**. Across the
15 split-K shapes, it decreased by **7.28%**. Across the five non-split shapes,
the geometric change was **+0.08%** (individual changes -0.63% to +1.01%).
The small non-split differences are not claimed as gains.

## CUDA Graph replay spot checks

These are separate repeated-buffer measurements without explicit L2 flushing;
larger inputs need not fit in L2. CUDA events measure 20 calls per graph replay,
with 5 rounds × 60 samples. Each pair uses the same GPU and alternates order.

| Batch × Seq | Before (µs) | After (µs) | Change |
|---|---:|---:|---:|
| 4 × 4096 | 15.030 | 14.309 | -4.8% |
| 4 × 32768 | 24.792 | 24.134 | -2.7% |
| 8 × 32768 | 36.013 | 35.698 | -0.9% |
| 32 × 4096 | 27.718 | 27.810 | +0.3% |
| 64 × 32768 | 215.212 | 215.718 | +0.2% |

The largest observed graph-mode reduction was 4.80%. Do not translate the
cold-L2 improvement directly into an end-to-end service improvement.

## Experiments not retained

- Whole-page K-scale TMA: a contiguous-layout attempt was incorrect and rejected.
  A stride-preserving implementation was bitwise correct, but the five sampled
  shapes ranged from essentially flat to 2.8% slower.
- Moving descriptor prefetch to loader warps: sampled changes were approximately
  -1.0% to +0.6%; no convincing gain, so it was not retained.
- Moving the score wait ahead of token-scale loads during cleanup caused small
  regressions. The retained implementation preserves the late wait.

No alternative execution versions or experimental toggles were added to the
runtime package.

## Validation and limits

- All 19 correctness cases passed on the cleaned, PDL-enabled core, including
  shuffled pages, short/non-aligned tails, split/no-split, H=8/16/32/64/128,
  and CUDA Graph replay with changing sequence lengths.
- Built the wheel without installing dependencies, installed it into an
  isolated target directory, and reran all 19 cases: **19 passed**. Five paired
  benchmark spot checks on that installed wheel were also bitwise identical
  and reproduced the observed timing behavior.
- Memcheck ran all 19 cases: **0 device-memory errors**, with
  `--report-api-errors no`.
- The default API-reporting mode emitted 25 `cuGetProcAddress_v2` invalid-value
  reports during CUDA capability/API probing, also reproduced in the control
  without cleanup. These are not reported here as a clean CUDA API check.
- Earlier synccheck barrier reports reproduced on native and official FP8
  controls remain unresolved; this round does not claim synccheck certification.
- No new end-to-end accuracy or service-latency measurements were made.

Raw samples, the frozen baseline, and the replay scripts are kept outside the
runtime package in `flashinfer_mla_fp4_native_perf_round2_20260929.tar.gz`,
beside the local project directory. The container experiment root
is `/tmp/native_perf_r2_20260929`; the final source is `release/`, with an
isolated wheel installation in `installed/`.

For provenance, the SHA256 of the sorted relative Python paths and per-file
SHA256 digests is `dc1ce0d0604bcdcdbf90254730b2c967bdfdba28387c23bb72f793ee0e53d350`
for the frozen baseline and
`1731e2391fa53bfe542efab7d8c2360e73e0bc096f3d3765c5133e798098ad77`
for the initial round-2 runtime package, before the follow-up above. Its isolated
installed wheel matches the latter.

After extracting the archive in the tested environment (with CUPTI available),
the cold-L2 comparison can be reproduced on an idle GPU with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=release:frozen:release/tests \
  python evidence/bench.py --full --rounds 5 --repeats 60 --output repeat.json
```
