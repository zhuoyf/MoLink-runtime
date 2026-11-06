# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PipeLive §6.1 asymmetric TCP entry protocol for overlapping NCCL groups."""

import functools
import json
import socket
import threading
import time
from typing import Any

import torch


def serialize_inference_nccl(operation):
    @functools.wraps(operation)
    def call(group, *args, **kwargs):
        mutex = getattr(group, "pipelive_mutex", None)
        if mutex is None:
            return operation(group, *args, **kwargs)
        with mutex:
            if pending := getattr(group, "pipelive_pending", None):
                pending.synchronize()
            result = operation(group, *args, **kwargs)
            handles = []
            if operation.__name__ == "isend_tensor_dict":
                handles = result
            elif operation.__name__ == "irecv_tensor_dict" and result is not None:
                handles = result[1]
            for handle in handles:
                handle.wait()
            # Reserve GPU entry until completion while allowing the worker to
            # enqueue computation. Every inference/migration entry checks it.
            group.pipelive_pending = torch.cuda.current_stream(
                group.device
            ).record_event()
            return result

    return call


class MigrationChannel:
    """One bounded NCCL transfer per successful ACK/ACCEPT handshake."""

    def __init__(self, group, device):
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        self.group = group
        self.device = device
        self.mutex = threading.RLock()
        group.pipelive_mutex = self.mutex
        self.stream = torch.cuda.Stream(device=device, priority=0)
        self.comm = PyNcclCommunicator(group.cpu_group, device)
        if not self.comm.available or self.comm.disabled:
            raise ValueError("PipeLive requires an available PyNccl communicator")
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", 0))
        listener.listen(1)
        listener.settimeout(60)
        # Same-node prototype; the existing worker supports only two local GPUs.
        descriptions: list[Any] = [None, None]
        torch.distributed.all_gather_object(
            descriptions, listener.getsockname()[1], group=group.cpu_group
        )
        if group.rank_in_group == 0:
            self.socket = socket.create_connection(("127.0.0.1", descriptions[1]), 60)
        else:
            self.socket, _ = listener.accept()
        listener.close()
        self.socket.settimeout(60)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.reader = self.socket.makefile("rb")
        self.rejections = 0

    def close(self):
        # The communicator and TCP channel are reused across transactions.
        pass

    def send_control(self, item):
        self.socket.sendall(json.dumps(item, separators=(",", ":")).encode() + b"\n")

    def receive_control(self):
        line = self.reader.readline(16 << 20)
        if not line or not line.endswith(b"\n"):
            raise ConnectionError("PipeLive control channel closed or oversized frame")
        return json.loads(line)

    def send(self, tensor, metadata):
        while True:
            with self.mutex:
                if pending := getattr(self.group, "pipelive_pending", None):
                    pending.synchronize()
                self.send_control({"op": "ACK", **metadata})
                reply = self.receive_control()
                if reply["op"] == "ACCEPT":
                    with torch.cuda.stream(self.stream):
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        begin.record()
                        if tensor.numel():
                            self.comm.send(
                                tensor, 1 - self.group.rank_in_group, self.stream
                            )
                        end.record()
                        end.synchronize()
                    applied = self.receive_control()
                    if applied["op"] != "APPLIED":
                        raise RuntimeError(
                            f"invalid migration acknowledgement: {applied}"
                        )
                    return begin, end
                if reply["op"] != "REJECT":
                    raise RuntimeError(f"invalid migration handshake: {reply}")
                self.rejections += 1
            time.sleep(0.001)

    def receive(self, apply):
        metadata = self.receive_control()
        if metadata["op"] == "STOP":
            self.send_control({"op": "STOPPED"})
            return False
        if metadata["op"] != "ACK":
            raise RuntimeError(f"invalid migration frame: {metadata}")
        if not self.mutex.acquire(blocking=False):
            self.send_control({"op": "REJECT"})
            return True
        try:
            pending = getattr(self.group, "pipelive_pending", None)
            if pending is not None and not pending.query():
                self.send_control({"op": "REJECT"})
                return True
            self.send_control({"op": "ACCEPT"})
            with torch.cuda.stream(self.stream):
                tensor = torch.empty(
                    metadata["shape"],
                    dtype=getattr(torch, metadata["dtype"]),
                    device=self.device,
                )
                if tensor.numel():
                    self.comm.recv(tensor, 1 - self.group.rank_in_group, self.stream)
                apply(tensor, metadata)
                self.stream.synchronize()
            self.send_control({"op": "APPLIED"})
        finally:
            self.mutex.release()
        return True
