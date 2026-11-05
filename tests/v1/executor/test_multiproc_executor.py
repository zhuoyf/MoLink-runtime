# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.executor.multiproc_executor import WorkerProc


class _ExitWorkerLoop(RuntimeError):
    pass


class _RpcPayload:
    pass


class _PayloadLifetimeCheckingQueue:
    def __init__(self) -> None:
        self.payload_ref: weakref.ReferenceType[_RpcPayload] | None = None
        self.dequeue_count = 0

    def dequeue(self, *, indefinite: bool):
        assert indefinite
        self.dequeue_count += 1
        if self.dequeue_count == 1:
            payload = _RpcPayload()
            self.payload_ref = weakref.ref(payload)
            return "consume", (payload,), {}, None

        assert self.payload_ref is not None
        assert self.payload_ref() is None
        raise _ExitWorkerLoop


def test_worker_rpc_payload_released_before_next_dequeue():
    queue = _PayloadLifetimeCheckingQueue()
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rpc_broadcast_mq = queue
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(consume=lambda payload: payload)
    worker_proc.handle_output = lambda output: None

    with pytest.raises(_ExitWorkerLoop):
        worker_proc.worker_busy_loop()

    assert queue.dequeue_count == 2


def test_execute_worker_rpc_returns_worker_exception():
    def fail():
        raise RuntimeError("test error")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(fail=fail)
    outputs: list[Any] = []
    worker_proc.handle_output = outputs.append

    worker_proc._execute_worker_rpc(("fail", (), {}, None))

    assert len(outputs) == 1
    assert isinstance(outputs[0], RuntimeError)
    assert str(outputs[0]) == "test error"


@pytest.mark.parametrize("configured,expected_mb", [(None, 24), ("1", 1)])
def test_worker_response_queue_honors_explicit_memory_limit(
    monkeypatch, configured, expected_mb
):
    """Allow local PP smoke tests in containers with 64 MiB shared memory."""
    from unittest.mock import Mock

    from vllm.v1.executor import multiproc_executor

    if configured is None:
        monkeypatch.delenv("VLLM_MQ_MAX_CHUNK_BYTES_MB", raising=False)
    else:
        monkeypatch.setenv("VLLM_MQ_MAX_CHUNK_BYTES_MB", configured)
    queue = Mock()
    monkeypatch.setattr(multiproc_executor, "MessageQueue", queue)
    proc = WorkerProc.__new__(WorkerProc)
    proc.worker = SimpleNamespace(rank=0)
    config = SimpleNamespace(parallel_config=SimpleNamespace(nnodes_within_dp=1))
    proc._init_message_queues(None, config)
    queue.assert_called_once_with(1, 1, max_chunk_bytes=expected_mb * 1024 * 1024)
