# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Native block-scaled E2M1 QK, E4M3 RoPE, and mixed E4M3/E2M1 PV."""

from types import SimpleNamespace
from typing import Type

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.pipeline import PipelineConsumer, PipelineProducer

from flashinfer.cute_dsl.attention.mainloop_spec import MLAMainloopSpec
from flashinfer.cute_dsl.attention.roles.mla_mma_fp8 import MLAMmaFP8Role
from flashinfer.cute_dsl.attention.scheduler.mla_persistent import (
    MLAStaticTileSchedulerParams,
    create_mla_static_tile_scheduler,
)

from ..mla_config_fp4 import MLAConfigFP4


class MLAMmaFP4Role(MLAMmaFP8Role):
    """Reuse FP8 MMA helpers while specializing mixed-dtype QK/PV ordering."""

    def __init__(self, config: MLAConfigFP4, mainloop: MLAMainloopSpec):
        super().__init__(config, mainloop)
        self.num_scale_groups = config.num_scale_groups
        self.qk_stages_per_scale_group = (
            config.scale_group_width // config.mma_qk_tiler[2]
        )

    def set_dtypes(
        self,
        q_dtype: Type[cutlass.Numeric],
        k_dtype: Type[cutlass.Numeric],
        v_dtype: Type[cutlass.Numeric],
    ) -> None:
        self.q_dtype = q_dtype
        self.k_dtype = k_dtype
        self.v_dtype = v_dtype

    @cute.jit
    def _make_local_qk_mma(self) -> cute.TiledMma:
        cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        return sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float4E2M1FN,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            cutlass.Float8E4M3FN,
            16,
            cta_group,
            self.mma_qk_tiler[:2],
        )

    @cute.jit
    def _s2t_sf(self, s_sf, t_sf, is_b: bool = False):
        if cutlass.const_expr(is_b):
            # M=128, CTA_GROUP=2 uses the SFB 2x2 TMEM datapath. Arrange
            # the two K atoms within each 16-byte row, not as two 512B atoms.
            layout = cute.make_layout(
                ((((32, 2, 2), 1), (16, 4)), 1, 2, cute.size(s_sf.shape[3])),
                stride=((((16, 4, 512), 0), (0, 1)), 0, 8, 1024),
            )
            s_sf = cute.make_tensor(s_sf.iterator, layout)
        source, target = cute.filter_zeros(s_sf), cute.filter_zeros(t_sf)
        op = (tcgen05.Cp2x64x128b0123Op(tcgen05.CtaGroup.TWO)
              if cutlass.const_expr(is_b) else
              tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO))
        atom = cute.make_copy_atom(op,
                                   cutlass.Float8E4M3FN)
        copy = tcgen05.make_s2t_copy(atom, target)
        thread = copy.get_slice(0)
        desc = tcgen05.get_s2t_smem_desc_tensor(copy, thread.partition_S(source))
        return copy, desc, thread.partition_D(target)

    @cute.jit
    def _gemm_qk_latent_one_stage(self, qk_params, s_stage_index, kv_stage_index,
                                  q_stage: int, accumulate: bool):
        mma = self._make_local_qk_mma()
        acc = qk_params.tStS_staged[None, None, None, s_stage_index]
        q = qk_params.tSrQ[None, None, None, (q_stage, 0)]
        k = qk_params.tSrKC[None, None, None, (q_stage, kv_stage_index)]
        if cutlass.const_expr(q_stage == 0):
            # Keep all four K-scale slices in disjoint TMEM columns. Issuing
            # their copies up front avoids repeatedly overwriting the SFB
            # operand between dependent QK MMAs.
            for stage in cutlass.range_constexpr(self.iterations_qk_latent):
                target = cute.make_tensor(
                    qk_params.ksf_dst.iterator + stage * 32,
                    qk_params.ksf_dst.layout,
                )
                cute.copy(qk_params.ksf_copy,
                          qk_params.ksf_src[None, None, None, None,
                                           stage + self.iterations_qk_latent * kv_stage_index],
                          target)
        for phase in cutlass.range_constexpr(cute.size(q, mode=[2])):
            mma.set(tcgen05.Field.SFA, qk_params.tQSF[None, None, phase].iterator + q_stage * 32)
            mma.set(tcgen05.Field.SFB, qk_params.tKSF[None, None, phase].iterator + q_stage * 32)
            mma.set(tcgen05.Field.ACCUMULATE, accumulate or phase != 0)
            cute.gemm(mma, acc, q[None, None, phase], k[None, None, phase], acc)

    @cute.jit
    def _make_local_qk_rope_mma(self) -> cute.TiledMma:
        cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        return sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.mma_qk_tiler[:2],
        )

    @cute.jit
    def _make_local_pv_mma(self) -> cute.TiledMma:
        cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        return sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.v_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.mma_pv_tiler[:2],
        )

    @cute.jit
    def _gemm_qk_rope_one_stage(
        self,
        qk_params: SimpleNamespace,
        s_stage_index: cutlass.Int32,
        kv_stage_index: cutlass.Int32,
        q_stage: int,
        accumulate: bool,
    ):
        local_mma = self._make_local_qk_rope_mma()
        t_scores = qk_params.tStS_staged[None, None, None, s_stage_index]
        for k_block in cutlass.range_constexpr(self.rope_dim // local_mma.shape_mnk[2]):
            local_mma.set(tcgen05.Field.ACCUMULATE, k_block != 0 or accumulate)
            cute.gemm(
                local_mma,
                t_scores,
                qk_params.tSrQ_rope[None, None, k_block, q_stage],
                qk_params.tSrKC_rope[None, None, k_block, kv_stage_index],
                t_scores,
            )

    @cute.jit
    def run(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        load_q_consumer: PipelineConsumer,
        load_k_consumer: PipelineConsumer,
        load_v_consumer: PipelineConsumer,
        mma_s_producer: PipelineProducer,
        p_mma_consumer: PipelineConsumer,
        mma_o_producer: PipelineProducer,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        tile_sched_params: MLAStaticTileSchedulerParams,
        sQ: cute.Tensor,
        sQ_rope: cute.Tensor,
        sKC: cute.Tensor,
        sKC_rope: cute.Tensor,
        sP: cute.Tensor,
        sVC: cute.Tensor,
        tmem_ptr: cute.Tensor,
        is_leader_cta: cutlass.Boolean,
        L: cutlass.Int32,
        sQSF: cute.Tensor,
        sKSF: cute.Tensor,
    ):
        """Publish native latent/rope scores, then consume scaled P for PV."""

        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrKC = tiled_mma_qk.make_fragment_B(sKC)
        rope_mma = self._make_local_qk_rope_mma()
        tSrQ_rope = rope_mma.make_fragment_A(sQ_rope)
        tSrKC_rope = rope_mma.make_fragment_B(sKC_rope)
        tOrP = tiled_mma_pv.make_fragment_A(sP)
        tOrVC = tiled_mma_pv.make_fragment_B(sVC)

        score_shape = tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        score_staged_fake = tiled_mma_qk.make_fragment_C(
            cute.append(score_shape, self.mma_s_stage)
        )
        score_staged = cute.make_tensor(tmem_ptr, score_staged_fake.layout)
        output_shape = tiled_mma_pv.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        output_fragment = tiled_mma_pv.make_fragment_C(output_shape)
        output_layout = cute.append(
            output_fragment.layout,
            cute.make_layout(
                L // self.mma_pv_tiler[1],
                stride=self.mma_pv_tiler[1] // self.warps_in_n,
            ),
        )
        output_staged = cute.make_tensor(
            score_staged.iterator + self.tmem_o_offset,
            output_layout,
        )
        qsf_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma_qk, self.mma_qk_tiler, 16,
            cute.slice_(sQSF.layout, (None, None, None, 0)),
        )
        ksf_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma_qk, self.mma_qk_tiler, 16,
            cute.slice_(sKSF.layout, (None, None, None, 0)),
        )
        tQSF = cute.make_tensor(cute.recast_ptr(tmem_ptr + 416, dtype=cutlass.Float8E4M3FN), qsf_layout)
        qcols = (self.mma_qk_tiler[0] // 32) * (self.mma_qk_tiler[2] // tiled_mma_qk.shape_mnk[2])
        tKSF = cute.make_tensor(cute.recast_ptr(tmem_ptr + 416 + qcols * 4, dtype=cutlass.Float8E4M3FN), ksf_layout)
        qsf_copy, qsf_src, qsf_dst = self._s2t_sf(sQSF, tQSF)
        ksf_copy, ksf_src, ksf_dst = self._s2t_sf(sKSF, tKSF, True)

        tile_sched = create_mla_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        while work_tile.is_valid_tile:
            tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, False)
            blk_coord = work_tile.tile_idx
            _, k_tile_count, _ = self._get_k_tile_count(
                split_kv, cache_seqs, block_split_kvs, blk_coord
            )
            if k_tile_count > 0:
                qk_params = SimpleNamespace(
                    sQ=sQ,
                    sQ_rope=sQ_rope,
                    sKC=sKC,
                    sKC_rope=sKC_rope,
                    tSrQ=tSrQ,
                    tSrQ_rope=tSrQ_rope,
                    tSrKC=tSrKC,
                    tSrKC_rope=tSrKC_rope,
                    tStS_staged=score_staged,
                    qsf_copy=qsf_copy, qsf_src=qsf_src, qsf_dst=qsf_dst,
                    ksf_copy=ksf_copy, ksf_src=ksf_src, ksf_dst=ksf_dst,
                    tQSF=tQSF, tKSF=tKSF,
                )
                pv_params = SimpleNamespace(
                    sP=sP,
                    sVC=sVC,
                    tOrP=tOrP,
                    tOrVC=tOrVC,
                    tOtO_staged=output_staged,
                )

                if is_leader_cta:
                    q_handle = load_q_consumer.wait_and_advance()
                    for qstage in cutlass.range_constexpr(4):
                        qsf_stage_dst = cute.make_tensor(
                            qsf_dst.iterator + qstage * 32, qsf_dst.layout,
                        )
                        cute.copy(qsf_copy, qsf_src[None, None, None, None, qstage], qsf_stage_dst)
                    kv_handle = load_k_consumer.wait_and_advance()
                    for group in cutlass.range_constexpr(self.num_scale_groups):
                        score_handle = mma_s_producer.acquire_and_advance()
                        for inner in cutlass.range_constexpr(
                            self.qk_stages_per_scale_group
                        ):
                            q_stage = group * self.qk_stages_per_scale_group + inner
                            self._gemm_qk_latent_one_stage(
                                qk_params,
                                score_handle.index,
                                kv_handle.index,
                                q_stage,
                                accumulate=(inner > 0),
                            )
                        for rope_stage in cutlass.range_constexpr(self.iterations_qk_rope):
                            self._gemm_qk_rope_one_stage(
                                qk_params, score_handle.index, kv_handle.index,
                                rope_stage, accumulate=True,
                            )
                        score_handle.commit()
                    kv_handle.release()
                    k_tile_count -= 1

                    while k_tile_count > 0:
                        kv_handle = load_k_consumer.wait_and_advance()
                        for group in cutlass.range_constexpr(self.num_scale_groups):
                            score_handle = mma_s_producer.acquire_and_advance()
                            for inner in cutlass.range_constexpr(
                                self.qk_stages_per_scale_group
                            ):
                                q_stage = group * self.qk_stages_per_scale_group + inner
                                self._gemm_qk_latent_one_stage(
                                    qk_params,
                                    score_handle.index,
                                    kv_handle.index,
                                    q_stage,
                                    accumulate=(inner > 0),
                                )
                            for rope_stage in cutlass.range_constexpr(self.iterations_qk_rope):
                                self._gemm_qk_rope_one_stage(
                                    qk_params, score_handle.index, kv_handle.index,
                                    rope_stage, accumulate=True,
                                )
                            score_handle.commit()
                        kv_handle.release()
                        k_tile_count -= 1

                        p_handle = p_mma_consumer.wait_and_advance()
                        v_handle = load_v_consumer.wait_and_advance()
                        pv_acc = tiled_mma_pv.get(tcgen05.Field.ACCUMULATE)
                        for acc_stage in cutlass.range_constexpr(self.iterations_pv_n):
                            output_handle = mma_o_producer.acquire_and_advance()
                            for p_stage in range(self.iterations_pv_k):
                                self._gemm_pv_one_stage(
                                    pv_params,
                                    p_handle.index,
                                    v_handle.index,
                                    p_stage,
                                    acc_stage,
                                    accumulate=(pv_acc or p_stage > 0),
                                )
                            output_handle.commit()
                        p_handle.release()
                        v_handle.release()
                        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)

                    q_handle.release()

                    v_handle = load_v_consumer.wait_and_advance()
                    pv_acc = tiled_mma_pv.get(tcgen05.Field.ACCUMULATE)
                    final_p_handle = p_mma_consumer.wait_and_advance()
                    for acc_stage in cutlass.range_constexpr(self.iterations_pv_n):
                        output_handle = mma_o_producer.acquire_and_advance()
                        for p_stage in range(self.iterations_pv_k):
                            self._gemm_pv_one_stage(
                                pv_params,
                                final_p_handle.index,
                                v_handle.index,
                                p_stage,
                                acc_stage,
                                accumulate=(pv_acc or p_stage > 0),
                            )
                        output_handle.commit()
                    final_p_handle.release()
                    v_handle.release()

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        mma_s_producer.tail()
        mma_o_producer.tail()


__all__ = ["MLAMmaFP4Role"]
