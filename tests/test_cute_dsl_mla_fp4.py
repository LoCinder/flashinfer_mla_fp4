"""The shared payload/PV must stay exact; execution error is tested separately."""

import pytest
import torch

from flashinfer_mla_fp4_native.quantization.pack_scales import pack_k_scales
from flashinfer_mla_fp4_native.quantization.quantize import quantize
from flashinfer_mla_fp4_native.quantization.kv_cache import write_shared
from flashinfer_mla_fp4_native import Runner
from gpu_guard import start, check
from reference import nvfp4, unpack, split_reference, fit_reference


@pytest.fixture(scope="module", autouse=True)
def exclusive_gpu():
    start()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("warps", [4, 8])
def test_writer(dtype, warps):
    check()
    torch.manual_seed(439)
    x = torch.randn(2, 128, 512, device="cuda").to(dtype)
    x[:1, :1] = 0
    x[:1, 1:2] = x[:1, 1:2].float().mul(1e-6).to(dtype)
    old, a = quantize(x, native=False)
    payload, token, sf = write_shared(x, warps=warps)
    assert torch.equal(payload, old)
    assert torch.equal(token, a)
    expected = fit_reference(x, old)
    torch.testing.assert_close(sf.float(), expected.float(), rtol=0, atol=0)
    same, a_same, sf_only = write_shared(x, payload=old, token_scale=a, fused=False, warps=warps)
    assert same.data_ptr() == old.data_ptr() and a_same.data_ptr() == a.data_ptr()
    assert torch.equal(sf_only.view(torch.uint8), sf.view(torch.uint8))
    _, _, packed = write_shared(x, packed_sf=True, warps=warps)
    assert torch.equal(packed.view(torch.uint8), pack_k_scales(sf.reshape(-1, 64, 32)).view_as(packed).view(torch.uint8))
    z = unpack(payload).reshape(2, 128, 32, 16)
    target = x.float().reshape_as(z)
    # Within the E4M3 candidate set, LS rounding is no worse than repeated token SF.
    repeated = a.clamp(1 / 512, 448).to(torch.float8_e4m3fn).float()
    fit_error = ((target - z * sf.float()[..., None]) ** 2).sum(-1)
    old_error = ((target - z * repeated[..., None]) ** 2).sum(-1)
    assert (fit_error <= old_error + 1e-5).all()


@pytest.mark.parametrize("batch,seq,heads,page,splits", [
    (1, 128, 8, 64, 1), (1, 256, 16, 64, 1), (2, 512, 64, 64, 3),
    (2, 2048, 128, 64, 8), (4, 4096, 64, 64, 16),
    (1, 8192, 64, 64, 8), (2, 256, 64, 64, 32),
    (32, 1024, 64, 64, 32),
])
def test_decode(batch, seq, heads, page, splits):
    check()
    torch.manual_seed(2891)
    x = torch.randn(batch, seq, 512, device="cuda") * torch.linspace(.3, 2.3, 512, device="cuda")
    x[:, 0] = 0
    g = 1.
    payload, a, ks = write_shared(x, global_scale=g)
    z = unpack(payload)
    q = nvfp4(torch.randn(batch, heads, 512, device="cuda"), g)
    qr = torch.randn(batch, heads, 64, device="cuda").to(torch.float8_e4m3fn)
    kr = torch.randn(batch, seq, 64, device="cuda").to(torch.float8_e4m3fn)
    lengths = seq - torch.arange(batch, device="cuda", dtype=torch.int32) * 17 - 13
    runner = Runner(q.packed, q.sf, qr, payload, ks, kr, a, lengths,
                    splits=splits, page=page)
    order = torch.randperm(runner.k.shape[0], device="cuda")
    runner.k, runner.r = runner.k[order], runner.r[order]
    runner.a, runner.ks = runner.a[order], runner.ks[order]
    runner.tables = order.argsort()[runner.tables.long()].to(torch.int32)
    k_recon = (z.reshape(batch, seq, 32, 16) * ks.float()[..., None] * g).flatten(-2)
    scores = (q.reconstructed @ k_recon.transpose(-1, -2)
              + qr.float() @ kr.float().transpose(-1, -2)) / 512**.5
    expected = split_reference(scores, z, a.reshape(batch, -1, 128),
                               lengths, splits)
    actual = runner().float()
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    assert ((actual - expected).norm() / expected.norm()).item() < .005
    # Native block-scaled MMA and the materialized FP32 reference associate
    # products differently; rare P values can land on opposite E4M3 rounding
    # boundaries. Keep the strict aggregate bound plus an absolute tail bound.
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.005)


@pytest.mark.parametrize("heads,splits", [
    (8, 1), (16, 8), (32, 3), (64, 8), (128, 4),
])
def test_tma_tails_graph(heads, splits):
    """Ragged physical pages, both CTA head halves, and changing graph inputs."""
    check()
    torch.manual_seed(739)
    batch, seq = 8, 384
    g = 1.
    x = torch.randn(batch, seq, 512, device="cuda")
    payload, a, ks = write_shared(x, global_scale=g)
    z = unpack(payload)
    q = nvfp4(torch.randn(batch, heads, 512, device="cuda"), g)
    qr = torch.randn(batch, heads, 64, device="cuda").to(torch.float8_e4m3fn)
    kr = torch.randn(batch, seq, 64, device="cuda").to(torch.float8_e4m3fn)
    lengths = torch.tensor([1, 31, 63, 64, 65, 127, 128, 257], device="cuda", dtype=torch.int32)
    runner = Runner(q.packed, q.sf, qr, payload, ks, kr, a, lengths,
                    splits=splits)
    order = torch.randperm(runner.k.shape[0], device="cuda")
    runner.k, runner.r, runner.a, runner.ks = [v[order] for v in
                                             (runner.k, runner.r, runner.a, runner.ks)]
    runner.tables = order.argsort()[runner.tables.long()].to(torch.int32)
    k = (z.reshape(batch, seq, 32, 16) * ks.float()[..., None] * g).flatten(-2)
    scores = (q.reconstructed @ k.transpose(-1, -2)
              + qr.float() @ kr.float().transpose(-1, -2)) / 512**.5
    runner()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner()
    for change in (False, True):
        if change:
            lengths.copy_(lengths.flip(0))
        graph.replay()
        actual = runner.out[:, 0].float()
        expected = split_reference(scores, z, a.reshape(batch, -1, 128),
                                   lengths, splits)
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all()
        assert ((actual - expected).norm() / expected.norm()).item() < .005
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.003)
