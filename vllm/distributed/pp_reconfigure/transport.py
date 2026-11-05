# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import uuid
from typing import Any

import torch

from vllm.distributed.parallel_state import get_pp_group


class PPTransport:
    """Lossless bounded-buffer transfer on vLLM's PP device group."""

    def copy(self, source, destination, source_rank):
        group = get_pp_group()
        local = source if group.rank_in_group == source_rank else destination
        local = local.view(1) if local.ndim == 0 else local
        rows = max(1, (16 << 20) // (local[0].numel() * local.element_size()))
        for offset in range(0, local.shape[0], rows):
            part = local[offset : offset + rows]
            if source is not None:
                group.send(part.contiguous(), dst=1 - source_rank)
            else:
                buffer = group.recv(part.shape, part.dtype, src=source_rank)
                part.copy_(buffer)
        return local.numel() * local.element_size()

    def connect(self):
        pass

    def close(self):
        pass


class NixlTransport:
    """NIXL UCX READs through bounded pinned-host staging buffers.

    shortcut: CPU staging supports ordinary Ethernet and strided KV layouts;
    register GPU block descriptors when optimizing migration bandwidth.
    """

    def __init__(self):
        from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

        if NixlWrapper is None:
            raise ValueError("NIXL backend requires nixl on both workers")
        self.agent = NixlWrapper(
            f"molink-{uuid.uuid4()}", nixl_agent_config(backends=["UCX"])
        )
        self.buffer = torch.empty(16 << 20, dtype=torch.uint8, pin_memory=True)
        self.registration = self.agent.register_memory(self.buffer)
        self.remote = None
        self.remote_ptr = None

    def connect(self):
        group = get_pp_group()
        peer = 1 - group.rank_in_group
        metadata = (self.agent.get_agent_metadata(), self.buffer.data_ptr())
        descriptions: list[Any] = [None, None]
        torch.distributed.all_gather_object(
            descriptions, metadata, group=group.cpu_group
        )
        self.remote = self.agent.add_remote_agent(descriptions[peer][0])
        self.remote_ptr = descriptions[peer][1]

    def copy(self, source, destination, source_rank):
        group = get_pp_group()
        local = source if group.rank_in_group == source_rank else destination
        local = local.view(1) if local.ndim == 0 else local
        rows = max(1, self.buffer.numel() // (local[0].numel() * local.element_size()))
        for offset in range(0, local.shape[0], rows):
            part = local[offset : offset + rows]
            nbytes = part.numel() * part.element_size()
            if nbytes > self.buffer.numel():
                raise ValueError("one KV block exceeds NIXL staging buffer")
            staging = self.buffer[:nbytes].view(part.dtype).view(part.shape)
            if source is not None:
                staging.copy_(part)
            # D2H copy has completed before the receiver can issue a READ.
            group.broadcast_object(True, src=source_rank)
            status: Any = None
            if destination is not None:
                handle = None
                try:
                    local_desc = self.agent.get_xfer_descs(
                        [(self.buffer.data_ptr(), nbytes, 0)], "DRAM"
                    )
                    remote_desc = self.agent.get_xfer_descs(
                        [(self.remote_ptr, nbytes, 0)], "DRAM"
                    )
                    handle = self.agent.initialize_xfer(
                        "READ", local_desc, remote_desc, self.remote
                    )
                    state = self.agent.transfer(handle)
                    deadline = time.monotonic() + 60
                    while state not in ("DONE", "ERR"):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("NIXL READ exceeded 60 seconds")
                        time.sleep(0.001)
                        state = self.agent.check_xfer_state(handle)
                    if state != "DONE":
                        raise RuntimeError("NIXL READ failed")
                    part.copy_(staging)
                    torch.cuda.current_stream(part.device).synchronize()
                    status = {"ok": True}
                except Exception as exc:
                    status = {"ok": False, "error": str(exc)}
                finally:
                    if handle is not None:
                        self.agent.release_xfer_handle(handle)
            # The source may reuse staging only after destination H2D completes.
            status = group.broadcast_object(status, src=1 - source_rank)
            if not status["ok"]:
                raise RuntimeError(status["error"])
        return local.numel() * local.element_size()

    def close(self):
        if self.remote is not None:
            self.agent.remove_remote_agent(self.remote)
            self.remote = None
        if self.registration is not None:
            self.agent.deregister_memory(self.registration)
            self.registration = None
