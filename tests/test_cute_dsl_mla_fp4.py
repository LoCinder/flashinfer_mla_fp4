"""Quantization, decode boundaries and packed V conversion correctness."""

import cutlass
import cutlass.cute as cute
import pytest
import torch

from flashinfer_mla_fp4_fused_dequant import Runner
from flashinfer_mla_fp4_fused_dequant.quantization.quantize import quantize
from flashinfer_mla_fp4_fused_dequant.roles.mla_dequant_v import _dequantize_e2m1x8
from gpu_guard import check, start
from reference import nvfp4, reference, unpack


@pytest.fixture(scope="module", autouse=True)
def exclusive_gpu():
    start()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float8_e4m3fn])
def test_quantization(dtype):
    check()
    torch.manual_seed(391)
    x = torch.randn(2, 192, 512, device="cuda").to(dtype)
    x[:, 0] = 0
    payload, scales = quantize(x)
    expected = nvfp4(x)
    assert torch.equal(payload, expected.packed)
    assert torch.equal(scales.view(torch.uint8), expected.sf.view(torch.uint8))
    allocated = (torch.empty_like(payload), torch.empty_like(scales))
    reused = quantize(x, output=allocated)
    assert reused[0] is allocated[0] and reused[1] is allocated[1]
    assert torch.equal(reused[0], payload)
    assert torch.equal(reused[1].view(torch.uint8), scales.view(torch.uint8))


@pytest.mark.parametrize("batch,seq,heads,splits,valid,permuted,kind", [
    (1, 64, 128, 1, None, False, "normal"),
    (2, 192, 128, 4, None, False, "normal"),
    (3, 512, 128, 4, [1, 129, 511], True, "normal"),
    (4, 4096, 128, 8, [127, 2048, 4095, 4096], True, "normal"),
    (1, 16384, 128, 1, None, False, "normal"),
    (4, 4096, 128, 16, None, True, "normal"),
    (1, 64, 128, 32, [1], True, "normal"),
    (2, 256, 128, 3, [16, 65], True, "normal"),
    (3, 320, 128, 8, [17, 63, 257], True, "normal"),
    (4, 4160, 128, 12, [4097, 15, 128, 4160], True, "normal"),
    *[(2, 192, h, s, [63, 129], True, "normal")
      for h, s in ((8, 1), (16, 3), (32, 4), (64, 8))],
    (2, 4096, 128, 4, None, True, "sharp"),
    (2, 4096, 128, 4, None, True, "mixed"),
])
def test_decode(batch, seq, heads, splits, valid, permuted, kind):
    check()
    torch.manual_seed(9371)
    x = torch.randn(batch, seq, 512, device="cuda")
    query = torch.randn(batch, heads, 512, device="cuda")
    if kind == "sharp":
        query *= 3
    if kind == "mixed":
        x *= torch.exp2(torch.linspace(-5, 1, seq, device="cuda"))[None, :, None]
    x[:, 0] = 0
    x, query = x.to(torch.float8_e4m3fn), query.to(torch.float8_e4m3fn)
    qr = torch.randn(batch, heads, 64, device="cuda").to(torch.float8_e4m3fn)
    rope = torch.randn(batch, seq, 64, device="cuda").to(torch.float8_e4m3fn)
    q, qs = quantize(query)
    k, ks = quantize(x)
    lengths = torch.tensor(valid or [seq] * batch, device="cuda", dtype=torch.int32)
    expected = reference(q, qs, qr, k, ks, rope, lengths, splits)
    runner = Runner(q, qs, qr, k, ks, rope, lengths, splits=splits)
    if permuted:
        order = torch.randperm(runner.k.shape[0], device="cuda")
        runner.k, runner.r, runner.ks = [value[order] for value in
                                        (runner.k, runner.r, runner.ks)]
        runner.tables = order.argsort()[runner.tables.long()].to(torch.int32)
    actual = runner().clone()
    error = (actual.float() - expected).norm() / expected.norm().clamp_min(1e-20)
    assert torch.isfinite(actual).all() and error < .006, error.item()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = runner()
    for _ in range(30):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, actual, rtol=0, atol=0)
    runner.qr.copy_((runner.qr.float() * .5).to(torch.float8_e4m3fn))
    updated = runner().clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, updated, rtol=0, atol=0)


@cute.kernel
def _convert(codes, scales, output):
    index = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
    if index < codes.shape[0]:
        output[index] = _dequantize_e2m1x8(codes[index], scales[index])


@cute.jit
def _launch_convert(codes, scales, output, stream):
    _convert(codes, scales, output).launch(
        grid=(cute.ceil_div(codes.shape[0], 128), 1, 1),
        block=(128, 1, 1), stream=stream,
    )


def test_packed_conversion():
    check()
    # Every E2M1 byte, crossed with every nonnegative finite E4M3 scale.
    byte = torch.arange(256, device="cuda", dtype=torch.int64).repeat(127)
    scales = torch.arange(127, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
    scales = scales.float().repeat_interleave(256)
    codes = (byte * 0x01010101).to(torch.uint32)
    output = torch.empty(codes.shape[0], device="cuda", dtype=torch.uint64)
    count = cute.sym_int()

    def fake(dtype):
        return cute.runtime.make_fake_compact_tensor(dtype, (count,), assumed_align=16)

    compiled = cute.compile(
        _launch_convert, fake(cutlass.Uint32), fake(cutlass.Float32),
        fake(cutlass.Uint64),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 2",
    )
    compiled(codes, scales, output)
    expected = (unpack(codes.view(torch.uint8).reshape(-1, 4)) * scales[:, None])
    expected = expected.clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.cuda.synchronize()
    assert torch.equal(output.view(torch.uint8).reshape(-1, 8), expected.view(torch.uint8))
