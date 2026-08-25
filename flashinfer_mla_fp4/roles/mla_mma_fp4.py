# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Mixed E4M3/E2M1 MMA role for per-token-scaled FP4 MLA decode."""

from types import SimpleNamespace
from typing import Type

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
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
        return sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.k_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.mma_qk_tiler[:2],
        )

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
    ):
        """Run per-token-scaled QK, then consume one P pipeline item for PV."""

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
                        score_handle.commit()
                    rope_handle = mma_s_producer.acquire_and_advance()
                    for rope_stage in cutlass.range_constexpr(self.iterations_qk_rope):
                        self._gemm_qk_rope_one_stage(
                            qk_params,
                            rope_handle.index,
                            kv_handle.index,
                            rope_stage,
                            accumulate=(rope_stage > 0),
                        )
                    rope_handle.commit()
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
                            score_handle.commit()
                        rope_handle = mma_s_producer.acquire_and_advance()
                        for rope_stage in cutlass.range_constexpr(
                            self.iterations_qk_rope
                        ):
                            self._gemm_qk_rope_one_stage(
                                qk_params,
                                rope_handle.index,
                                kv_handle.index,
                                rope_stage,
                                accumulate=(rope_stage > 0),
                            )
                        rope_handle.commit()
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
