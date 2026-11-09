# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PipeLive §6.1 dirty-slot draining and immutable bounded KV patches."""

import functools
import threading

import numpy as np
import torch

from vllm.triton_utils import tl, triton


def live_step(operation):
    @functools.wraps(operation)
    def call(worker, *args, **kwargs):
        live = getattr(worker, "_pipelive", None)
        if live is None or not live.active:
            return operation(worker, *args, **kwargs)
        with live.cache_lock:
            output = args[0] if args else kwargs["scheduler_output"]
            live.before_step(output)
            if live.gathered is not None:
                torch.cuda.current_stream(worker.device).wait_event(live.gathered)
            result = operation(worker, *args, **kwargs)
            live.after_step(output)
            return result

    return call


def live_sampling(operation):
    @functools.wraps(operation)
    def call(worker, *args, **kwargs):
        live = getattr(worker, "_pipelive", None)
        if live is None:
            return operation(worker, *args, **kwargs)
        with live.channel.mutex:
            group = live.channel.group
            if pending := getattr(group, "pipelive_pending", None):
                pending.synchronize()
            result = operation(worker, *args, **kwargs)
            if handler := worker.model_runner.pp_handler:
                group.pipelive_pending = handler.broadcast_stream.record_event()
            return result

    return call


class DirtyBitmap:
    def __init__(self, num_slots):
        self.bits = np.zeros(num_slots, dtype=np.bool_)
        self.lock = threading.Lock()
        self.scheduled = 0

    def mark(self, slots, advance=True):
        slots = np.asarray(slots, dtype=np.int64)
        if slots.size and (slots.min() < 0 or slots.max() >= len(self.bits)):
            raise ValueError("dirty slot outside the KV allocation")
        with self.lock:
            self.bits[slots] = True
            if advance:
                self.scheduled += len(slots)

    def drain(self):
        with self.lock:
            slots = np.flatnonzero(self.bits).tolist()
            self.bits.fill(False)
            return slots, self.scheduled


class PatchLifecycle:
    """Monotone allocation generations and FIFO patch versions per transaction."""

    def __init__(self, generations):
        self.generations = list(generations)
        self.version = 0
        self.lock = threading.RLock()

    def observe(self, generations):
        if len(generations) != len(self.generations):
            raise ValueError("KV capacity changed during migration")
        # Monotone generations let unchanged Decode bypass an in-flight apply.
        if all(
            generation <= current
            for generation, current in zip(generations, self.generations)
        ):
            return []
        with self.lock:
            changed = [
                block
                for block, generation in enumerate(generations)
                if generation > self.generations[block]
            ]
            for block in changed:
                self.generations[block] = generations[block]
            return changed

    def accept(self, version, generations):
        if version != self.version + 1:
            raise ValueError("out-of-order or replayed KV patch")
        valid = set()
        for block, generation in generations:
            if not 0 <= block < len(self.generations):
                raise ValueError("patch block outside KV allocation")
            if generation >= self.generations[block]:
                valid.add(block)
                self.generations[block] = generation
        self.version = version
        return valid


@triton.jit
def _copy_slots(
    addresses,
    slots,
    payload,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
    SCATTER: tl.constexpr,
):
    index = tl.program_id(0)
    slot = tl.load(slots + index)
    address = tl.load(addresses + slot // BLOCK)
    base = address.to(tl.pointer_type(payload.dtype.element_ty))
    offsets = tl.arange(0, TILE)
    cache_offsets = offsets // DIM * BLOCK * DIM + slot % BLOCK * DIM + offsets % DIM
    if SCATTER:
        data = tl.load(
            payload + index * HEADS * DIM + offsets, offsets < HEADS * DIM, 0
        )
        tl.store(base + cache_offsets, data, offsets < HEADS * DIM)
    else:
        data = tl.load(base + cache_offsets, offsets < HEADS * DIM, 0)
        tl.store(payload + index * HEADS * DIM + offsets, data, offsets < HEADS * DIM)


def copy_slots(handle, slots, payload=None):
    pool, _ = handle._pipelive
    spec = pool.spec
    indices = torch.tensor(slots, dtype=torch.int64, device=pool.device)
    scatter = payload is not None
    if payload is None:
        payload = torch.empty(
            len(slots),
            spec.num_kv_heads,
            2 * spec.head_size,
            device=pool.device,
            dtype=spec.dtype,
        )
    if slots:
        _copy_slots[(len(slots),)](
            handle,
            indices,
            payload,
            spec.num_kv_heads,
            2 * spec.head_size,
            spec.block_size,
            triton.next_power_of_2(spec.num_kv_heads * 2 * spec.head_size),
            scatter,
        )
    return payload
