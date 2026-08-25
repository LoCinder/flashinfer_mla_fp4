# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Loader roles for per-token-scaled FP4 MLA decode."""

from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.pipeline import PipelineProducer

from flashinfer.cute_dsl.attention.roles.mla_loader_fp8 import (
    MLAFP8LoaderKRole,
    MLAFP8LoaderVRole,
)
from flashinfer.cute_dsl.attention.scheduler.mla_persistent import (
    MLAStaticTileSchedulerParams,
    create_mla_static_tile_scheduler,
)

from ..mla_config_fp4 import MLAConfigFP4


class MLAFP4LoaderKRole(MLAFP8LoaderKRole):
    """Reuse the FP8 Q/K TMA path and stage one FP32 scale per token."""

    def __init__(self, config: MLAConfigFP4):
        super().__init__(config)
        self.num_scale_groups = config.num_scale_groups

    @cute.jit
    def _load_scale_one_tile(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        k_index: cutlass.Int32,
        scale_handle,
    ):
        """Load effective FP32 token scales into the compute pipeline."""

        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = tidx % 32
        values_per_lane = self.mma_qk_tiler[1] // 32
        local_n = lane_idx * values_per_lane
        kv_idx = k_index * self.mma_qk_tiler[1] + local_n

        scale_layout = cute.make_layout(
            (values_per_lane, self.num_scale_groups),
            stride=(self.num_scale_groups, 1),
        )
        scale_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        scale_stage_offset = (
            local_n * self.num_scale_groups
            + scale_handle.index * self.mma_qk_tiler[1] * self.num_scale_groups
        )
        scale_stage_offset = cute.assume(scale_stage_offset, values_per_lane)
        s_scale = cute.make_tensor(
            qk_params.sTokenScale.iterator + scale_stage_offset,
            scale_layout,
        )

        # A page boundary cannot split this aligned per-lane vector, so full
        # lanes can copy directly to shared memory without register staging.
        can_vector_copy = (
            cute.elem_less(kv_idx + values_per_lane - 1, common_params.K)
            if cutlass.const_expr(self.page_size % values_per_lane == 0)
            else False
        )
        if can_vector_copy:
            page_slot = kv_idx // self.page_size
            token_in_page = kv_idx - page_slot * self.page_size
            physical_page = common_params.mPT[page_slot]
            src_offset = (
                token_in_page * qk_params.mTokenSF.stride[0]
                + physical_page * qk_params.mTokenSF.stride[2]
            )
            src_offset = cute.assume(src_offset, values_per_lane)
            g_scale = cute.make_tensor(
                qk_params.mTokenSF.iterator + src_offset,
                scale_layout,
            )
            cute.copy(scale_copy_atom, g_scale, s_scale)
        else:
            for i in cutlass.range_constexpr(values_per_lane):
                token_idx = kv_idx + i
                for group in cutlass.range_constexpr(self.num_scale_groups):
                    s_scale[i, group] = cutlass.Float32(0.0)
                    if cute.elem_less(token_idx, common_params.K):
                        token_page_slot = token_idx // self.page_size
                        token_in_physical_page = (
                            token_idx - token_page_slot * self.page_size
                        )
                        token_physical_page = common_params.mPT[token_page_slot]
                        s_scale[i, group] = qk_params.mTokenSF[
                            token_in_physical_page,
                            group,
                            token_physical_page,
                        ]
        scale_handle.commit()

    @cute.jit
    def run(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        load_q_producer: PipelineProducer,
        load_k_producer: PipelineProducer,
        load_scale_producer: PipelineProducer,
        tile_sched_params: MLAStaticTileSchedulerParams,
    ):
        """Run the inherited Q/K TMA loop with an aligned scale stage."""

        tile_sched = create_mla_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        while work_tile.is_valid_tile:
            blk_coord = work_tile.tile_idx
            k_index, k_tile_count, local_split_kv = self._get_k_tile_count(
                split_kv,
                cache_seqs,
                block_split_kvs,
                blk_coord,
            )
            if k_tile_count > 0:
                tile_params = SimpleNamespace(
                    blk_coord=blk_coord,
                    local_split_kv=local_split_kv,
                    mPT=common_params.mPT,
                    K=cache_seqs[blk_coord[2]],
                )
                self._setup_tma_partitions(tile_params, qk_params)

                k_tile_count_init = k_tile_count
                while k_tile_count > 0:
                    if k_tile_count_init == k_tile_count:
                        q_handle = load_q_producer.acquire_and_advance()
                        self._load_q_tma(qk_params, q_handle.barrier)

                    k_handle = load_k_producer.acquire_and_advance()
                    self._load_k_one_tile(
                        tile_params,
                        qk_params,
                        k_index,
                        k_handle.barrier,
                        k_handle.index,
                    )
                    scale_handle = load_scale_producer.acquire_and_advance()
                    self._load_scale_one_tile(
                        tile_params,
                        qk_params,
                        k_index,
                        scale_handle,
                    )
                    k_index += 1
                    k_tile_count -= 1

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        load_q_producer.tail()
        load_k_producer.tail()
        load_scale_producer.tail()


# V uses the same paged TMA movement as FP8; only its MMA consumer differs.
MLAFP4LoaderVRole = MLAFP8LoaderVRole


__all__ = ["MLAFP4LoaderKRole", "MLAFP4LoaderVRole"]
