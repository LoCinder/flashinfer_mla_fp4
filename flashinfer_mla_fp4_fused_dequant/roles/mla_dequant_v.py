"""Register dequantization of the same QK NVFP4 cache into FP8 V SMEM."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from flashinfer.cute_dsl.attention.roles.mla_loader_fp8 import MLAFP8LoaderVRole
from flashinfer.cute_dsl.attention.scheduler.mla_persistent import (
    create_mla_static_tile_scheduler,
)


@dsl_user_op
def _dequantize_e2m1x8(codes, scale, *, loc=None, ip=None):
    """Expand eight packed FP4 values into eight FP8 bytes."""
    # E2M1 * E4M3 products are exact in FP16. Keep the conversion and multiply
    # packed as half2, with just the original final rounding to E4M3.
    return cutlass.Uint64(
        llvm.inline_asm(
            T.i64(),
            [codes.ir_value(loc=loc, ip=ip), scale.ir_value(loc=loc, ip=ip)],
            """{
            .reg .b8 b0, b1, b2, b3;
            .reg .b16 s, f0, f1, f2, f3;
            .reg .b32 ss, h0, h1, h2, h3, o0, o1;
            mov.b32 {b0,b1,b2,b3}, $1;
            cvt.rn.f16.f32 s, $2;
            mov.b32 ss, {s,s};
            cvt.rn.f16x2.e2m1x2 h0, b0;
            cvt.rn.f16x2.e2m1x2 h1, b1;
            cvt.rn.f16x2.e2m1x2 h2, b2;
            cvt.rn.f16x2.e2m1x2 h3, b3;
            mul.f16x2 h0, h0, ss;
            mul.f16x2 h1, h1, ss;
            mul.f16x2 h2, h2, ss;
            mul.f16x2 h3, h3, ss;
            cvt.rn.satfinite.e4m3x2.f16x2 f0, h0;
            cvt.rn.satfinite.e4m3x2.f16x2 f1, h1;
            cvt.rn.satfinite.e4m3x2.f16x2 f2, h2;
            cvt.rn.satfinite.e4m3x2.f16x2 f3, h3;
            mov.b32 o0, {f0,f1};
            mov.b32 o1, {f2,f3};
            mov.b64 $0, {o0,o1};
            }""",
            "=l,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class MLAFP4LoaderVRole(MLAFP8LoaderVRole):
    @cute.jit
    def _expand(self, p, table, k_index, stage, valid_length):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = cutlass.Uint32(tid - 320)
        rank = bid % 2
        for it in cutlass.range(3, unroll=1):
            vec = lane + it * 192
            if vec < 512:
                token = vec // 4
                local_dim = (vec % 4) * 64
                acc_stage = local_dim // 128
                dim = local_dim % 128
                channel = acc_stage * 256 + rank * 128 + dim
                logical_token = k_index * 128 + token
                row = token % 64
                regs = cute.make_rmem_tensor(
                    cute.make_layout(64), cutlass.Float4E2M1FN
                )
                byte_regs = cute.make_tensor(
                    cute.recast_ptr(regs.iterator, dtype=cutlass.Uint8),
                    cute.make_layout(32),
                )
                scales = cute.make_rmem_tensor(
                    cute.make_layout(4), cutlass.Float8E4M3FN
                )
                if logical_token < valid_length:
                    page = table[logical_token // 64]
                    ptr = (
                        p.raw_latent.iterator
                        + row * 512
                        + channel
                        + page * 64 * 512
                    )
                    src = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.Uint8,
                            ptr.toint(),
                            p.raw_latent.memspace,
                            assumed_align=32,
                        ),
                        cute.make_layout(32),
                    )
                    cute.autovec_copy(src, byte_regs)
                    block = channel // 16
                    off = (
                        page * 2048
                        + (block // 8) * 512
                        + (row % 32) * 16
                        + ((block % 8) // 4) * 8
                        + (row // 32) * 4
                    )
                    # The packed layout gives each four-scale group a
                    # four-byte-aligned address; preserve vectorized loads.
                    ssrc = cute.make_tensor(
                        p.packed_scales.iterator + cute.assume(off, 4),
                        cute.make_layout(4),
                    )
                    cute.autovec_copy(ssrc, scales)
                else:
                    byte_regs.fill(0)
                    scales.fill(0)
                words = cute.recast_tensor(byte_regs, cutlass.Uint32)
                for block in cutlass.range_constexpr(4):
                    out = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Uint64)
                    for pair in cutlass.range_constexpr(2):
                        out[pair] = _dequantize_e2m1x8(
                            words[block * 2 + pair], cutlass.Float32(scales[block])
                        )
                    coord = (
                        (dim + block * 16, token % 32),
                        0,
                        (token % 64) // 32,
                        ((acc_stage, token // 64), stage),
                    )
                    offset = cute.crd2idx(coord, p.sVC.layout)
                    # Each block starts at 16 FP8 values; the SMEM swizzle
                    # preserves that alignment. Without the hint, autovec_copy
                    # emits scalar stores with heavy shared-bank conflicts.
                    offset = cute.assume(offset, 16)
                    dst = cute.make_tensor(
                        p.sVC.iterator + offset, cute.make_layout(16)
                    )
                    cute.autovec_copy(
                        cute.recast_tensor(out, cutlass.Float8E4M3FN), dst
                    )

    @cute.jit
    def run(
        self,
        common,
        params,
        split_kv,
        cache_seqs,
        block_split_kvs,
        producer,
        sched_params,
    ):
        sched = create_mla_static_tile_scheduler(
            sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work = sched.initial_work_tile_info()
        while work.is_valid_tile:
            coord = work.tile_idx
            index, count, _ = self._get_k_tile_count(
                split_kv, cache_seqs, block_split_kvs, coord
            )
            table = common.mPT[None, coord[2]]
            while count > 0:
                handle = producer.acquire_and_advance()
                self._expand(
                    params,
                    table,
                    index,
                    handle.index,
                    cache_seqs[coord[2]],
                )
                cute.arch.fence_view_async_shared()
                handle.commit()
                index += 1
                count -= 1
            sched.advance_to_next_work()
            work = sched.get_current_work()
        producer.tail()
