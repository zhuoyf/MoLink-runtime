# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PipeLive §5: independently allocated, layer-stacked KV blocks."""

import math
import os

import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec


def enabled():
    return os.environ.get("HETVLLM_PIPELIVE") == "1"


class StackedKVCache:
    """Stable logical IDs address independently releasable physical stacks.

    Each allocation holds k layers at the same logical block index. Pointer
    tensors are the runner's cache handles, never dense views of the KV data.
    """

    def __init__(self, names, spec, num_blocks, device, stack_factor=2):
        if type(spec) is not FullAttentionSpec or spec.kv_quant_mode != 0:
            raise ValueError("PipeLive requires uniform, unquantized full attention")
        indices = [int(name.split(".")[2]) for name in names]
        if (
            stack_factor < 1
            or len(names) % stack_factor
            or indices != list(range(indices[0], indices[0] + len(indices)))
            or indices[0] % stack_factor
        ):
            raise ValueError("PipeLive layer ranges must align to the stacking factor")
        self.names = names
        self.locations = {name: index for index, name in enumerate(names)}
        self.group_ids = set(range(len(names) // stack_factor))
        self.spec = spec
        self.device = device
        self.stack_factor = stack_factor
        self.shape = (spec.num_kv_heads, spec.block_size, 2 * spec.head_size)
        self.page_elements = math.prod(self.shape)
        self.allocation_bytes = math.ceil(
            stack_factor * spec.page_size_bytes / (2 << 20)
        ) * (2 << 20)
        self.stacks: dict[int, dict[int, torch.Tensor]] = {}
        self.handles = {}
        for name in names:
            handle = torch.zeros(num_blocks, dtype=torch.int64, device=device)
            handle._pipelive = (self, name)
            self.handles[name] = handle
        self.add_blocks(range(num_blocks))

    def add_blocks(self, block_ids):
        block_ids = list(block_ids)
        if any(block_id in self.stacks for block_id in block_ids):
            raise ValueError("KV block already allocated")
        allocated = {}
        for block_id in block_ids:
            stacks = {
                group_id: torch.zeros(
                    self.allocation_bytes, dtype=torch.uint8, device=self.device
                )
                for group_id in self.group_ids
            }
            allocated[block_id] = stacks
        self.stacks.update(allocated)
        self._refresh_addresses()

    def remove_blocks(self, block_ids):
        block_ids = list(block_ids)
        if 0 in block_ids or not set(block_ids) <= self.stacks.keys():
            raise ValueError("cannot release null or absent KV blocks")
        # Caller has quiesced readers and excluded these IDs from scheduler use.
        for block_id in block_ids:
            del self.stacks[block_id]
        self._refresh_addresses()

    def block(self, name, block_id):
        index = self.locations[name]
        raw = self.stacks[block_id][index // self.stack_factor]
        offset = index % self.stack_factor * self.page_elements
        return raw.view(self.spec.dtype)[offset : offset + self.page_elements].view(
            self.shape
        )

    def _refresh_addresses(self):
        capacity = max(self.stacks, default=0) + 1
        for name in self.names:
            handle = self.handles[name]
            if handle.numel() < capacity:
                handle.resize_(capacity)
            addresses = [0] * handle.numel()
            for block_id in self.stacks:
                addresses[block_id] = self.block(name, block_id).data_ptr()
            handle.copy_(torch.tensor(addresses, dtype=torch.int64, device=self.device))

    def key_value(self, name):
        # Only strides/shape and the base pointers are used in address mode.
        view = self.block(name, 0).unsqueeze(0).transpose(1, 2)
        return view.split(self.spec.head_size, dim=-1)

    @property
    def allocated_bytes(self):
        return len(self.stacks) * len(self.group_ids) * self.allocation_bytes

    def retire_layers(self, names):
        names = set(names) & self.handles.keys()
        groups = {self.locations[name] // self.stack_factor for name in names}
        if any(
            name not in names and self.locations[name] // self.stack_factor in groups
            for name in self.names
        ):
            raise ValueError("cannot release part of a layer stack")
        for stacks in self.stacks.values():
            for group_id in groups:
                del stacks[group_id]
        self.group_ids -= groups
        self.names = [name for name in self.names if name not in names]
        for name in names:
            del self.handles[name]


def allocate_stacked_cache(config, device, kernel_block_sizes):
    if len(config.kv_cache_groups) != 1:
        raise ValueError("PipeLive requires a single full-attention KV group")
    group = config.kv_cache_groups[0]
    if kernel_block_sizes != [group.kv_cache_spec.block_size]:
        raise ValueError("PipeLive requires matching kernel and scheduler block sizes")
    pool = StackedKVCache(
        group.layer_names,
        group.kv_cache_spec,
        config.num_blocks,
        device,
        int(os.environ.get("HETVLLM_PIPELIVE_STACK", "2")),
    )
    return pool.handles
