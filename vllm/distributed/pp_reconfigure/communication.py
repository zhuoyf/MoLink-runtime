# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online migration: asymmetric TCP entry protocol for overlapping NCCL groups."""

import functools
import json
import socket
import threading
import time
from collections import defaultdict
from typing import Any

import torch


def serialize_inference_nccl(operation):
    @functools.wraps(operation)
    def call(group, *args, **kwargs):
        mutex = getattr(group, "pp_reconfigure_mutex", None)
        if mutex is None:
            return operation(group, *args, **kwargs)
        with mutex:
            if pending := getattr(group, "pp_reconfigure_pending", None):
                pending.synchronize()
            result = operation(group, *args, **kwargs)
            handles = []
            if operation.__name__ == "isend_tensor_dict":
                handles = result
            elif operation.__name__ == "_irecv_tensor_dict_payload":
                handles = result[1]
            for handle in handles:
                handle.wait()
            # Reserve GPU entry until completion while allowing the worker to
            # enqueue computation. Every inference/migration entry checks it.
            group.pp_reconfigure_pending = torch.cuda.current_stream(
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
        group.pp_reconfigure_mutex = self.mutex
        self.stream = torch.cuda.Stream(device=device, priority=0)
        self.comm = PyNcclCommunicator(group.cpu_group, device)
        if not self.comm.available or self.comm.disabled:
            raise ValueError(
                "Online migration requires an available PyNccl communicator"
            )
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
        self.timings_ms = defaultdict(float)

    def close(self):
        # The communicator and TCP channel are reused across transactions.
        pass

    def send_control(self, item):
        self.socket.sendall(json.dumps(item, separators=(",", ":")).encode() + b"\n")

    def receive_control(self):
        line = self.reader.readline(16 << 20)
        if not line or not line.endswith(b"\n"):
            raise ConnectionError(
                "Online migration control channel closed or oversized frame"
            )
        return json.loads(line)

    def send(self, tensor, metadata):
        while True:
            started = time.perf_counter()
            with self.mutex:
                acquired = time.perf_counter()
                self.timings_ms["communication_lock_wait"] += (
                    acquired - started
                ) * 1000
                if pending := getattr(self.group, "pp_reconfigure_pending", None):
                    pending.synchronize()
                pending_done = time.perf_counter()
                self.timings_ms["inference_nccl_wait"] += (
                    pending_done - acquired
                ) * 1000
                self.send_control({"op": "ACK", **metadata})
                reply = self.receive_control()
                handshake_done = time.perf_counter()
                self.timings_ms["handshake"] += (handshake_done - pending_done) * 1000
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
                    nccl_done = time.perf_counter()
                    self.timings_ms["nccl_enqueue_and_wait"] += (
                        nccl_done - handshake_done
                    ) * 1000
                    applied = self.receive_control()
                    self.timings_ms["applied_ack_wait"] += (
                        time.perf_counter() - nccl_done
                    ) * 1000
                    if applied["op"] != "APPLIED":
                        raise RuntimeError(
                            f"invalid migration acknowledgement: {applied}"
                        )
                    return begin, end
                if reply["op"] != "REJECT":
                    raise RuntimeError(f"invalid migration handshake: {reply}")
                self.rejections += 1
            retry_started = time.perf_counter()
            time.sleep(0.001)
            self.timings_ms["retry_backoff"] += (
                time.perf_counter() - retry_started
            ) * 1000

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
            pending = getattr(self.group, "pp_reconfigure_pending", None)
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
