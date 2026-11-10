# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side KV resizing with stable logical block IDs."""

from vllm.distributed.pp_reconfigure.errors import ReconfigurationValidationError
from vllm.v1.core.kv_cache_utils import KVCacheBlock


def required_migration_blocks(pool):
    """Migrate referenced KV; evict idle prefixes before omitting their blocks."""
    disabled: set[int] = getattr(pool, "_pp_disabled_blocks", set())
    required = []
    for block in pool.blocks:
        if block.block_id in disabled or block.is_null:
            continue
        if block.ref_cnt > 0:
            required.append(block.block_id)
        elif block.block_hash is not None or pool.cached_block_hashes_by_block.get(
            block.block_id
        ):
            # A later prefix hit must not expose KV omitted from the destination.
            pool._maybe_evict_cached_block(block)
    return required


def plan_resize(pool, capacity):
    if type(capacity) is not int or capacity < 2:
        raise ReconfigurationValidationError(
            "KV capacity must be an integer >= 2 (including null block)"
        )
    disabled: set[int] = getattr(pool, "_pp_disabled_blocks", set())
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
    disabled: set[int] = getattr(pool, "_pp_disabled_blocks", set())
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
    pool._pp_disabled_blocks = disabled
    pool.num_gpu_blocks = plan["capacity"]
