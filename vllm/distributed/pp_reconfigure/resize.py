# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler half of PipeLive resizing; live logical block IDs stay stable."""

from vllm.distributed.pp_reconfigure.errors import ReconfigurationValidationError
from vllm.v1.core.kv_cache_utils import KVCacheBlock


def plan_resize(pool, capacity):
    if type(capacity) is not int or capacity < 2:
        raise ReconfigurationValidationError(
            "KV capacity must be an integer >= 2 (including null block)"
        )
    disabled: set[int] = getattr(pool, "_pipelive_disabled", set())
    delta = capacity - pool.num_gpu_blocks
    if delta < 0:
        free = [
            block.block_id
            for block in reversed(pool.blocks)
            if block.block_id not in disabled
            and not block.is_null
            and block.ref_cnt == 0
        ]
        if len(free) < -delta:
            raise ReconfigurationValidationError(
                "KV shrink would release active request blocks"
            )
        return {"capacity": capacity, "remove": free[:-delta], "add": []}
    restored = sorted(disabled)[:delta]
    new_ids = list(range(len(pool.blocks), len(pool.blocks) + delta - len(restored)))
    return {"capacity": capacity, "remove": [], "add": restored + new_ids}


def commit_pool_resize(pool, plan):
    disabled: set[int] = getattr(pool, "_pipelive_disabled", set())
    for block_id in plan["remove"]:
        block = pool.blocks[block_id]
        assert block.ref_cnt == 0 and not block.is_null
        pool._maybe_evict_cached_block(block)
        pool.free_block_queue.remove(block)
        disabled.add(block_id)
    for block_id in plan["add"]:
        if block_id == len(pool.blocks):
            pool.blocks.append(KVCacheBlock(block_id))
        else:
            disabled.remove(block_id)
        pool.free_block_queue.append(pool.blocks[block_id])
    pool._pipelive_disabled = disabled
    pool.num_gpu_blocks = plan["capacity"]
