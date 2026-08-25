# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for per-token-scaled FP4 CuTe DSL MLA decode."""

from dataclasses import dataclass
from statistics import median

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.cute_dsl.attention import BatchMLADecodeCuteDSLWrapper
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import is_sm100a_supported, is_sm110a_supported
from flashinfer_mla_fp4 import BatchMLADecodeCuteDSLFP4Wrapper

LATENT_DIM, ROPE_DIM = 512, 64


@dataclass
class _Case:
    query: torch.Tensor
    fp4_cache: torch.Tensor
    token_scale: torch.Tensor
    fp8_cache: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    workspace: torch.Tensor
    page_size: int
    max_seq_len: int
    is_var_seq: bool


def _skip_if_unsupported() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if not (is_sm100a_supported(device) or is_sm110a_supported(device)):
        pytest.skip("Per-token-scaled FP4 MLA requires SM100-SM110 (tcgen05)")
    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL is not available")


@torch.no_grad()
def _make_w512_cache(
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack latent E2M1 and return a materialized E4M3 reference cache."""

    num_pages, page_size, _ = source.shape
    token_ids = torch.arange(
        num_pages * page_size,
        dtype=torch.int64,
        device=source.device,
    ).view(num_pages, page_size)
    token_scale = torch.where(token_ids % 2 == 0, 0.5, 1.0)[..., None].float()

    normalized = source[..., :LATENT_DIM].float() / token_scale
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32,
        device=source.device,
    )
    magnitude_codes = torch.bucketize(normalized.abs(), thresholds)
    nibbles = magnitude_codes | ((normalized < 0).to(torch.int64) << 3)
    packed = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8)

    magnitude_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=source.device,
    )
    latent = (
        magnitude_lut[magnitude_codes]
        * torch.where(normalized < 0, -1.0, 1.0)
        * token_scale
    )
    rope = source[..., LATENT_DIM:].to(torch.float8_e4m3fn)
    fp4_cache = torch.cat((packed, rope.view(torch.uint8)), dim=-1).contiguous()
    fp8_cache = torch.cat((latent, rope.float()), dim=-1).to(torch.float8_e4m3fn)
    return fp4_cache, token_scale, fp8_cache


def _make_case(
    batch_size: int,
    seq_len: int,
    page_size: int,
    is_var_seq: bool,
    num_heads: int = 128,
) -> _Case:
    device = torch.device("cuda")
    logical_dim = LATENT_DIM + ROPE_DIM
    pages_per_batch = (seq_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_batch + 2

    query = (
        torch.randn(
            batch_size,
            1,
            num_heads,
            logical_dim,
            dtype=torch.float16,
            device=device,
        )
        * 0.1
    ).to(torch.float8_e4m3fn)
    source = torch.randn(
        total_pages,
        page_size,
        logical_dim,
        dtype=torch.float16,
        device=device,
    )
    fp4_cache, token_scale, fp8_cache = _make_w512_cache(source)

    expected_bytes = total_pages * page_size * (LATENT_DIM // 2 + ROPE_DIM + 4)
    assert fp4_cache.nbytes + token_scale.nbytes == expected_bytes
    assert expected_bytes / (total_pages * page_size * logical_dim) < 0.565

    block_tables = torch.arange(
        batch_size * pages_per_batch,
        dtype=torch.int32,
        device=device,
    ).view(batch_size, pages_per_batch)
    if is_var_seq:
        seq_lens = torch.tensor(
            [max(1, seq_len - 1 - i * 17) for i in range(batch_size)],
            dtype=torch.int32,
            device=device,
        )
        fp4_cache = fp4_cache.unsqueeze(1)
        token_scale = token_scale.unsqueeze(1)
        fp8_cache = fp8_cache.unsqueeze(1)
    else:
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    return _Case(
        query=query,
        fp4_cache=fp4_cache,
        token_scale=token_scale,
        fp8_cache=fp8_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        workspace=torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device),
        page_size=page_size,
        max_seq_len=int(seq_lens.max().item()),
        is_var_seq=is_var_seq,
    )


def _plan(case: _Case):
    common = dict(
        kv_lora_rank=LATENT_DIM,
        qk_rope_head_dim=ROPE_DIM,
        num_heads=case.query.size(2),
        page_size=case.page_size,
        q_dtype=case.query.dtype,
        out_dtype=torch.bfloat16,
        is_var_seq=case.is_var_seq,
        enable_pdl=False,
    )
    fp4_wrapper = BatchMLADecodeCuteDSLFP4Wrapper(case.workspace)
    fp4_wrapper.plan(**common, kv_dtype=torch.uint8)
    fp8_wrapper = BatchMLADecodeCuteDSLWrapper(case.workspace)
    fp8_wrapper.plan(**common)
    return fp4_wrapper, fp8_wrapper


def _run(wrapper, case: _Case, cache, token_scale=None, out=None):
    if out is None:
        out = torch.empty(
            case.query.size(0),
            1,
            case.query.size(2),
            LATENT_DIM,
            dtype=torch.bfloat16,
            device=case.query.device,
        )
    scale_kwargs = {} if token_scale is None else {"kv_cache_sf": token_scale}
    wrapper.run(
        q=case.query,
        kv_cache=cache,
        block_tables=case.block_tables,
        seq_lens=case.seq_lens,
        max_seq_len=case.max_seq_len,
        softmax_scale=1.0 / (LATENT_DIM**0.5),
        out=out,
        **scale_kwargs,
    )
    return out


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [128, 512, 2048])
@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("is_var_seq", [False, True])
def test_cute_dsl_mla_decode_fp4_accuracy(
    batch_size: int,
    seq_len: int,
    page_size: int,
    is_var_seq: bool,
) -> None:
    """Compare per-token-scaled FP4 against the same cache materialized as E4M3."""

    _skip_if_unsupported()
    torch.manual_seed(42)
    case = _make_case(batch_size, seq_len, page_size, is_var_seq)
    fp4_wrapper, fp8_wrapper = _plan(case)
    fp4_output = _run(fp4_wrapper, case, case.fp4_cache, case.token_scale)
    fp8_output = _run(fp8_wrapper, case, case.fp8_cache)

    assert fp4_output.shape == (batch_size, 1, 128, LATENT_DIM)
    assert fp4_output.dtype == torch.bfloat16
    assert torch.isfinite(fp4_output).all()
    torch.testing.assert_close(
        fp4_output.float(),
        fp8_output.float(),
        atol=1.5e-3,
        rtol=0,
    )


def test_cute_dsl_mla_decode_fp4_performance() -> None:
    """Report one warm-L2 FP4-vs-FP8 latency point and catch gross regressions."""

    _skip_if_unsupported()
    torch.manual_seed(42)
    batch_size, seq_len, page_size, num_heads = 4, 4096, 64, 64
    case = _make_case(batch_size, seq_len, page_size, False, num_heads)
    fp4_wrapper, fp8_wrapper = _plan(case)
    fp4_output = _run(fp4_wrapper, case, case.fp4_cache, case.token_scale)
    fp8_output = _run(fp8_wrapper, case, case.fp8_cache)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        fp4_output.float(), fp8_output.float(), atol=1.5e-3, rtol=0
    )

    def measure(fn) -> float:
        return float(
            median(
                bench_gpu_time(
                    fn,
                    dry_run_iters=5,
                    repeat_iters=20,
                    use_cuda_graph=True,
                    cold_l2_cache=False,
                )
            )
        )

    fp8_ms = measure(lambda: _run(fp8_wrapper, case, case.fp8_cache, out=fp8_output))
    fp4_ms = measure(
        lambda: _run(
            fp4_wrapper,
            case,
            case.fp4_cache,
            case.token_scale,
            fp4_output,
        )
    )
    ratio = fp4_ms / fp8_ms
    print(
        f"warm-L2 B={batch_size} S={seq_len} H={num_heads} page={page_size}: "
        f"FP8={fp8_ms * 1000:.3f} us, FP4={fp4_ms * 1000:.3f} us, "
        f"speedup={fp8_ms / fp4_ms:.3f}x"
    )
    assert fp8_ms > 0 and fp4_ms > 0
    assert ratio < 2.0, f"FP4 warm-L2 latency regressed: FP4/FP8={ratio:.3f}x"
