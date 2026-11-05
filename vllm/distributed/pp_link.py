"""hetvllm-fork (pp2): link-scheduled PP transport for the cross-machine hop.

Replaces the NCCL isend/irecv tensor-dict path between pipeline stages with
an application-level transport that treats the shared NIC as a schedulable
resource (plan.md technical points 1+2, wire part):

  * messages sliced (default 32 KiB frames) so higher-priority traffic can
    overtake prefill bytes already queued at the socket,
  * two priority lanes per direction (hi = decode-bearing batches, lo =
    prefill-heavy batches); decode wins the wire, prefill fills gaps,
  * per-direction unacked byte window (receiver acks once a message is fully
    reassembled) = upstream backpressure: downstream congestion stops new
    committed bytes instead of piling them into kernel buffers,
  * per-lane token-bucket pacing so the two TP lanes never oversubscribe
    the shared NIC,
  * metadata rides in the message header (removes the per-step gloo
    send_object/recv_object round trip from the critical path).

BF16 payloads by default; HETVLLM_PP_LINK_CODEC=int8 quantizes boundary
tensors before slicing (technical point 3).

HETVLLM_PP_MERGE=1 ships the pre-norm sum s = h + r as a single vector
instead of (hidden_states, residual) - halves the boundary payload again
(plan.md round 3, priority 1). The receiver fabricates a zero residual so
the next stage's fused add+RMSNorm computes exactly RMSNorm(s), residual=s.

HETVLLM_LINK_STATS=<path> appends a per-(stage,lane) byte/token counter
line every ~2s for the WAN-bytes-per-effective-token audit.

Wire format (zmq multipart, one message = one engine step's tensor dict):
  frame 0: header json {"id", "prio", "nbytes", "tensors": [
      {"key", "codec", "n", "h", "nbytes"} ...]}
  frames 1..: payload chunks, sliced from the concatenated payloads at
      SLICE-byte granularity (chunks may span tensor boundaries).
Acks are tiny json frames {"ack": "<id>"} sent on the opposite direction's
hi socket; a {"fb": ...} frame type is reserved for sampled-token feedback.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import zmq
except ImportError:  # pragma: no cover
    zmq = None

_ENABLED = os.environ.get("HETVLLM_PP_LINK", "0") == "1"
_SLICE = int(os.environ.get("HETVLLM_PP_LINK_SLICE_KIB", "32")) * 1024
_WIN = int(os.environ.get("HETVLLM_PP_LINK_WIN_MB", "8")) * (1 << 20)
# per-lane rate cap in bytes/s (0 = unlimited); two lanes share the link
_RATE = float(os.environ.get("HETVLLM_PP_LINK_LANE_MBPS", "0")) * (1 << 20)
_CODEC = os.environ.get("HETVLLM_PP_LINK_CODEC", "bf16")
_BASE_PORT = int(os.environ.get("HETVLLM_PP_LINK_PORT", "27100"))
# stage-0 machine (A) and stage-1 machine (B) addresses on the A<->B link;
# required when the link is enabled
_BIND_IP = os.environ.get("HETVLLM_PP_LINK_BIND", "")
_PEER_IP = os.environ.get("HETVLLM_PP_LINK_PEER", "")
# merged boundary: ship s = h + r once instead of both vectors
_MERGE = os.environ.get("HETVLLM_PP_MERGE", "0") == "1"
# the link carries the cross-machine hop stage _FROM_STAGE (last stage on A)
# -> _FROM_STAGE + 1 (first on B): 0 for PP2, 1 for PP4 = 2 + 2 stages
_FROM_STAGE = int(os.environ.get("HETVLLM_PP_LINK_FROM", "0"))
_STATS_PATH = os.environ.get("HETVLLM_LINK_STATS", "")
# async tx/rx (default on): the worker thread never blocks on the GPU or on
# the peer - D2H completion is awaited by the send thread (CUDA event), the
# step's send handle completes once the payload is handed to zmq (not on
# the peer's ack: no WAN round trip per step), and the receiver frees its
# pinned staging buffer by event instead of a device-wide synchronize. 0 = old behaviour
# (stream sync before send, handle waits for the peer's ack, recv
# torch.cuda.synchronize()).
_ASYNC_TX = os.environ.get("HETVLLM_PP_LINK_ASYNC", "1") == "1"

_HWM = 256
# preallocated pinned staging buffers per direction and lane
_POOL_N = int(os.environ.get("HETVLLM_PP_LINK_POOL", "8"))
# dequantize this many groups per slice (bounds receiver GPU temporaries)
_DEC_CHUNK = 1 << 15

# step hint set by the worker before each send: (total_tok, n_reqs) of the
# batch whose boundary tensors are about to go on the wire. Used for the
# priority-lane decision without touching call signatures.
_STEP_HINT = (0, 0, None)


def set_step_hint(total_tok: int, n_reqs: int, hi: bool | None = None) -> None:
    """Classify the step about to be sent. `hi` is the scheduler's explicit
    TP-C class (True = decode micro-batch, False = prefill micro-batch);
    None falls back to the shape heuristic (pure-decode ~= hi)."""
    global _STEP_HINT
    _STEP_HINT = (total_tok, n_reqs, hi)


def enabled() -> bool:
    return _ENABLED and zmq is not None


def is_link_sender(stage: int) -> bool:
    """Should PP rank `stage` SEND its boundary dict through the link?
    Only the hop's sender (_FROM_STAGE) does."""
    return enabled() and stage == _FROM_STAGE


def is_link_receiver(stage: int) -> bool:
    """Should PP rank `stage` RECEIVE its boundary dict through the link?
    Only the hop's receiver (_FROM_STAGE + 1) does."""
    return enabled() and stage == _FROM_STAGE + 1


# ---------------------------------------------------------------------------
# codec
# ---------------------------------------------------------------------------
class _Codec:
    """Symmetric quantization of boundary tensors (technical point 3).

    int8: per-token per-64-channel symmetric, fp16 scales sent inline.
    int4: per-token per-32-channel symmetric ([-7,7]), fp16 scales, two
    nibbles packed per byte - halves the wire bytes again vs int8.
    """

    def __init__(self, mode: str):
        self.mode = mode
        self.group = (
            int(os.environ.get("HETVLLM_PP_LINK_INT4_GROUP", "32") or 32)
            if mode == "int4"
            else 64
        )

    def cpu_need(self, numel: int) -> int:
        if self.mode == "bf16":
            return numel * 2
        if self.mode == "int8":
            # scales (2B per 64ch) + int8 payload
            return numel // 8 * 9 + 64
        # int4: scales (2B per group) + packed payload (4b per element)
        return numel // self.group * 2 + numel // 2 + 64

    def encode(self, t: torch.Tensor, scratch: torch.Tensor):
        """t [N,H] bf16 cuda -> flat uint8 pinned view of scratch.

        The whole wire image (scales + payload) is built on the GPU and
        leaves in ONE non-blocking D2H copy on the current stream; the
        caller records an event after it and the send thread waits on that
        event, so the worker's host thread never blocks on the GPU here
        (_ASYNC_TX=0 restores the old blocking scale copy + stream sync).
        """
        N, H = t.shape
        if self.mode == "bf16":
            dst = scratch[: N * H * 2]
            dst.copy_(t.view(torch.uint8).reshape(-1), non_blocking=True)
            return dst, {"codec": "bf16", "n": N, "h": H, "nbytes": N * H * 2}
        g = self.group
        assert H % g == 0, H
        x = t.float().view(N, H // g, g)
        amax = x.abs().amax(dim=-1) + 1e-8
        if self.mode == "int4":
            scale = amax / 7.0
            q = (x / scale.unsqueeze(-1)).clamp(-7, 7).to(torch.int32) & 0xF
            pairs = q.reshape(-1, 2)
            body = (pairs[:, 0] | (pairs[:, 1] << 4)).to(torch.uint8)
        else:
            scale = amax / 127.0
            body = (
                (x / scale.unsqueeze(-1))
                .clamp(-127, 127)
                .to(torch.int8)
                .view(torch.uint8)
                .reshape(-1)
            )
        sc = scale.to(torch.float16).view(torch.uint8).reshape(-1)
        ns = sc.numel()
        dst = scratch[: ns + body.numel()]
        if _ASYNC_TX:
            dst.copy_(torch.cat((sc, body)), non_blocking=True)
        else:
            dst[:ns].copy_(sc.cpu(), non_blocking=False)
            dst[ns:].copy_(body, non_blocking=True)
        return dst, {"codec": self.mode, "n": N, "h": H, "nbytes": ns + body.numel()}

    def decode_gpu(self, seg: torch.Tensor, tm: dict, device):
        """Wire image -> bf16 [N, H] on `device`.

        Dequantizes in slices of _DEC_CHUNK groups into one preallocated
        bf16 output: the int32/fp32 temporaries of a whole 4096-row 32B
        step (~250 MB) are not covered by vLLM's startup memory profile and
        OOM'd a 3090 at 0.90 utilization; per-slice they stay at a few MB.
        """
        N, H = tm["n"], tm["h"]
        if tm["codec"] == "bf16":
            return seg.view(torch.bfloat16).view(N, H).to(device, non_blocking=True)
        g = self.group
        G = N * (H // g)  # groups in the message
        ns = G * 2
        dev = seg.to(device, non_blocking=True)
        scales = dev[:ns].view(torch.float16)  # [G]
        body = dev[ns:]
        out = torch.empty(G * g, dtype=torch.bfloat16, device=device)
        int4 = tm["codec"] == "int4"
        for g0 in range(0, G, _DEC_CHUNK):
            g1 = min(G, g0 + _DEC_CHUNK)
            sc = scales[g0:g1].float().unsqueeze(-1)
            if int4:
                b = body[g0 * g // 2 : g1 * g // 2].to(torch.int32)
                q = torch.stack((b & 0xF, b >> 4), dim=1).view(g1 - g0, g)
                q = torch.where(q >= 8, q - 16, q)
            else:
                q = body[g0 * g : g1 * g].view(torch.int8).view(g1 - g0, g)
            out[g0 * g : g1 * g] = (q.float() * sc).view(-1).to(torch.bfloat16)
        return out.view(N, H)


_CODEC_OBJ = _Codec(_CODEC)


# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------
@dataclass
class _OutMsg:
    mid: int
    header: bytes
    payload: torch.Tensor  # flat uint8 (pinned scratch or cpu)
    nbytes: int
    prio: int
    event: threading.Event = field(default_factory=threading.Event)
    # CUDA event after the D2H copies (async tx): the send thread waits on
    # it before the first payload byte goes out
    ready: torch.cuda.Event | None = None
    # set once the last frame is handed to zmq
    queued: threading.Event = field(default_factory=threading.Event)
    bases: list = field(default_factory=list)  # pool buffers to return


class MsgHandle:
    """wait() blocks until the receiver acked the message (window slot)."""

    def __init__(self, msg: _OutMsg):
        self._msg = msg

    def wait(self) -> None:
        self._msg.event.wait()


class _QueuedHandle:
    """Async tx: wait() returns once the whole message has been handed to
    zmq (its IO thread owns the bytes from then on) - not for the peer's
    ack. The worker waits on this at the start of its NEXT step.

    Never returning earlier is a liveness rule, not a tuning choice: under
    Model Runner V2 the next steps' kernels queue behind a stream wait on
    stage 1's sampled tokens; once the launch queue is full the host
    thread blocks inside a CUDA launch WITH the GIL held, the send thread
    can't run, stage 1 never gets this message, and the tokens never come
    (observed deadlock, 2026-10-30).
    """

    def __init__(self, msg: _OutMsg):
        self._msg = msg

    def wait(self) -> None:
        self._msg.queued.wait()


@dataclass
class _InMsg:
    header: dict
    buf: torch.Tensor | None = None
    event: threading.Event = field(default_factory=threading.Event)
    t_ready: float = 0.0  # wall time the message was fully reassembled
    base: torch.Tensor | None = None  # whole staging buffer (pool unit)

    @property
    def nbytes(self) -> int:
        return int(self.header["nbytes"])


class _RecvFuture:
    """Lazy dict-of-tensors: wait() materializes GPU tensors (in arrival
    order). Returned by recv_tensor_dict_async; consumed by the model
    forward via AsyncIntermediateTensors's handle wait."""

    def __init__(self, link: PPLink, sid: int):
        self.link = link
        self.sid = sid
        self.result: dict | None = None

    def wait(self) -> dict:
        if self.result is None:
            self.result = self.link.next_message(self.sid)
        return self.result


class _FillHandle:
    """Fills the step's tensor dict in place when waited.

    A plain module-level class on purpose: the previous per-call
    `class _Fill` closure was a new class object per step, and class
    objects always sit in reference cycles - so each step's received GPU
    tensors (wire image, all-gathered full tensor, zero residual) lived
    until a cyclic GC pass, which vLLM's gc.freeze() + a quiet hot loop
    make rare. On Qwen3-32B with full 4096-row steps that leaked ~80 MB
    per step and OOM'd the 3090s within ~20 s of prefill.
    """

    __slots__ = ("_fut", "_out")

    def __init__(self, fut: _RecvFuture, out: dict):
        self._fut = fut
        self._out = out

    def wait(self) -> None:
        fut = self._fut
        if fut is None:
            return
        self._out.update(fut.wait())
        fut.result = None  # the dict above is now the only owner
        self._fut = None


class PPLink:
    def __init__(self):
        self.stage = None
        self.lane = None
        self.ctx = None
        self.sock: dict = {}
        self.send_q: queue.Queue[_OutMsg | None] = queue.Queue()
        self.ack_q: queue.Queue[int] = queue.Queue()
        self.pending: dict[int, _OutMsg] = {}
        self.plock = threading.Lock()
        self.unacked = 0
        self.msg_id = 0
        self.send_sid = 0  # step id of the next message to send
        self.recv_sid = 0  # step id the next recv call expects
        self.in_ready: dict[int, _InMsg] = {}
        self._requeue = deque()
        self.stats = {
            "sent_msg": 0,
            "sent_bytes": 0,
            "recv_msg": 0,
            "recv_bytes": 0,
            "ack_wait_ms": 0.0,
            "net_wait_ms": 0.0,
            "hdr_wait_ms": 0.0,
            "pace_sleep_s": 0.0,
            "win_wait_s": 0.0,
        }
        self._pool_out: list[torch.Tensor] = []
        self._pool_in: list[torch.Tensor] = []
        self._in_pend: deque = deque()  # (event, staging buf) in flight
        self._cap = 4 << 20  # staging buffer size (setup)
        self._stash: dict[str, deque] = {}
        self._tok_acc = 0  # scheduled tokens seen (bytes/token audit)
        self._last_dump = 0.0

    # ------------------------------------------------------------- setup
    def setup(self, stage: int, lane: int):
        self.stage, self.lane = stage, lane
        self.device = torch.cuda.current_device()
        assert zmq is not None
        self.ctx = zmq.Context.instance()
        # attention-DP (tp=1, dp=2): each engine core's stage-0 rank runs its
        # own link instance, both with lane=0 - offset the ports by the DP
        # rank so the two senders on machine A don't fight over one port.
        from vllm.distributed.parallel_state import get_dp_group

        self.dp = get_dp_group().rank_in_group
        p = _BASE_PORT + lane * 8 + self.dp * 16
        for name, port in (
            ("fwd_hi", p),
            ("fwd_lo", p + 1),
            ("bwd_hi", p + 2),
            ("bwd_lo", p + 3),
        ):
            s = self.ctx.socket(zmq.PAIR)
            s.setsockopt(zmq.SNDHWM, _HWM)
            s.setsockopt(zmq.RCVHWM, _HWM)
            s.setsockopt(zmq.LINGER, 0)
            is_fwd = name.startswith("fwd")
            # fwd sockets are owned (bound) by the hop's sender
            # (_FROM_STAGE, machine A); bwd sockets by the receiver
            # (_FROM_STAGE + 1, machine B). The owner binds its own IP;
            # the peer connects to the owner's IP.
            owner_ip = _BIND_IP if is_fwd else _PEER_IP
            if (stage == _FROM_STAGE) == is_fwd:
                s.bind(f"tcp://{owner_ip}:{port}")
            else:
                s.connect(f"tcp://{owner_ip}:{port}")
            self.sock[name] = s
        self._prefill_pools()
        self._handshake()
        threading.Thread(target=self._recv_loop, daemon=True).start()
        threading.Thread(target=self._send_loop, daemon=True).start()

    def _handshake(self):
        """PAIR drops frames sent before the peer connected, so exchange
        HS/HA per socket. Any application frames that arrive mid-handshake
        are stashed (not dropped) and replayed into the recv path."""
        socks = self.sock
        for s in socks.values():
            s.setsockopt(zmq.RCVTIMEO, 50)
        done = set()
        deadline = time.time() + 120
        while len(done) < len(socks) and time.time() < deadline:
            for name, s in socks.items():
                if name in done:
                    continue
                with suppress(zmq.Again):
                    s.send(b"HS")
                try:
                    m = s.recv()
                except zmq.Again:
                    continue
                if m == b"HS":
                    with suppress(zmq.Again):
                        s.send(b"HA")
                    done.add(name)
                elif m == b"HA":
                    done.add(name)
                else:
                    self._stash.setdefault(name, deque()).append(m)
        for s in socks.values():
            s.setsockopt(zmq.RCVTIMEO, -1)
        if len(done) < len(socks):
            raise TimeoutError("pp_link handshake failed")

    # ------------------------------------------------------- engine API
    def _prio(self, nbytes: int) -> int:
        tok, reqs, hi = _STEP_HINT
        if hi is not None:
            return 0 if hi else 1
        return 0 if tok <= 2 * max(1, reqs) + 8 else 1

    def _scratch(self, pool, need):
        """Take (not borrow) a buffer from the pool: while a message holds
        it, no other tensor may alias it. Returned to the pool by the send
        loop once the message is fully on the wire (sender) / once its H2D
        copy has passed (receiver).

        Pools are filled at setup with max-step-size buffers: pinned
        allocation on the hot path is a driver call that can stall behind
        in-flight NCCL work (V2 runs the sampled-token broadcast on a side
        stream), and from the recv thread it hung stage 1 for good."""
        if pool is self._pool_in:
            self._recycle_in()
        for i, t in enumerate(pool):
            if t.numel() >= need:
                return pool.pop(i)
        if pool is self._pool_in and self._in_pend:
            # pool exhausted: wait (polling, GIL released) for the oldest
            # in-flight H2D to pass instead of allocating
            while not self._in_pend[0][0].query():
                time.sleep(0.0002)
            self._recycle_in()
            for i, t in enumerate(pool):
                if t.numel() >= need:
                    return pool.pop(i)
        logger.warning(
            "pp_link: hot-path pinned alloc %d B (pool %s)",
            need,
            "in" if pool is self._pool_in else "out",
        )
        return torch.empty(max(need, self._cap), dtype=torch.uint8, pin_memory=True)

    def _recycle_in(self):
        while self._in_pend and self._in_pend[0][0].query():
            self._pool_in.append(self._in_pend.popleft()[1])

    def _prefill_pools(self):
        """Size = the largest boundary message one lane can carry (full
        step budget, both tensors in bf16)."""
        try:
            from vllm.config import get_current_vllm_config

            cfg = get_current_vllm_config()
            rows = cfg.scheduler_config.max_num_batched_tokens
            hid = cfg.model_config.get_hidden_size()
            tp = cfg.parallel_config.tensor_parallel_size
        except Exception:  # noqa: BLE001
            rows, hid, tp = 8192, 8192, 1
        self._cap = max(4 << 20, 2 * rows * hid * 2 // max(1, tp) + 4096)
        # only the side that uses a pool fills it; the sender may hold one
        # buffer per tensor (2 unmerged) for every in-flight step
        sender = self.stage == _FROM_STAGE
        pool = self._pool_out if sender else self._pool_in
        for _ in range(_POOL_N * (2 if sender else 1)):
            pool.append(torch.empty(self._cap, dtype=torch.uint8, pin_memory=True))

    def send_tensor_dict(self, tensor_dict: dict) -> list[MsgHandle]:
        from vllm.distributed.parallel_state import get_tp_group

        tp = get_tp_group().world_size
        merged = False
        if _MERGE and "hidden_states" in tensor_dict and "residual" in tensor_dict:
            # plan.md round 3: the next layer consumes s = h + r only via
            # fused add+RMSNorm (norm out, s as the new residual), so the
            # pair carries no independent information - ship s once.
            # fp32 sum, rounded to bf16 = the wire representation; the
            # receiver's zeros residual makes the first layer reproduce
            # RMSNorm(s), residual = s exactly.
            h = tensor_dict["hidden_states"]
            r = tensor_dict["residual"]
            tensor_dict = dict(tensor_dict)
            tensor_dict["hidden_states"] = (h.float() + r.float()).to(torch.bfloat16)
            del tensor_dict["residual"]
            merged = True
        self._tok_acc += _STEP_HINT[0]
        parts = []
        bases = []  # whole pool buffers behind the segment views
        metas = []
        total = 0
        for k, v in tensor_dict.items():
            if not isinstance(v, torch.Tensor):
                metas.append({"key": k, "codec": "obj", "nbytes": 0})
                continue
            assert v.is_cuda and v.dim() == 2 and v.dtype == torch.bfloat16, (
                k,
                v.shape,
                v.dtype,
            )
            # match the native all-gather optimization: each TP lane sends
            # only its slice of the flattened tensor (half the link bytes)
            flat = v.reshape(-1)
            if tp > 1 and flat.numel() % tp == 0:
                flat = flat.reshape(tp, -1)[self.lane].contiguous()
            need = _CODEC_OBJ.cpu_need(flat.numel())
            base = self._scratch(self._pool_out, need)
            bases.append(base)
            seg, meta = _CODEC_OBJ.encode(flat.view(1, -1), base)
            parts.append(seg)
            metas.append(
                meta
                | {
                    "key": k,
                    "fn": v.shape[0],
                    "fh": v.shape[1],
                    "sl": flat.numel() != v.numel(),
                }
            )
            total += meta["nbytes"]
        ready = None
        if _ASYNC_TX:
            ready = torch.cuda.Event()
            ready.record()
        else:
            torch.cuda.current_stream().synchronize()
        mid = self.msg_id
        hdr = {
            "id": mid,
            "sid": self.send_sid,
            "prio": self._prio(total),
            "nbytes": total,
            "tensors": metas,
            "mg": 1 if merged else 0,
            "tok": _STEP_HINT[0],
        }
        msg = _OutMsg(
            mid,
            json.dumps(hdr).encode(),
            parts if parts else torch.empty(0, dtype=torch.uint8),
            total,
            hdr["prio"],
            ready=ready,
            bases=bases,
        )
        with self.plock:
            self.pending[mid] = msg
            self.msg_id += 1
            self.send_sid += 1
        self.send_q.put(msg)
        return [_QueuedHandle(msg) if _ASYNC_TX else MsgHandle(msg)]

    def recv_tensor_dict_async(self):
        """Mirrors irecv_tensor_dict's (tensor_dict, handles, postprocess)
        contract: the dict fills in place when the handle is waited.

        Each call consumes the NEXT step id: messages are matched to steps
        explicitly (the wire is allowed to reorder between the hi/lo
        priority sockets, so arrival order cannot be used)."""
        sid = self.recv_sid
        self.recv_sid += 1
        out: dict = {}
        return out, [_FillHandle(_RecvFuture(self, sid), out)], []

    def peek_ready(self, sid: int) -> tuple[bool, float]:
        """Is the message for step `sid` already fully reassembled?

        Used by the stage-1 worker to classify its own idle time: a message
        that was complete BEFORE the forward call started means the GPU sat
        idle with legal ready work (scheduler/driver-bound bubble), not
        waiting on the wire. Returns (ready, seconds_since_ready).
        """
        m = self.in_ready.get(sid)
        if m is None:
            return False, 0.0
        return True, time.time() - m.t_ready

    def next_message(self, sid: int) -> dict:
        self._recycle_in()
        t0 = time.perf_counter()
        while sid not in self.in_ready:
            time.sleep(0.0002)
        m = self.in_ready.pop(sid)
        self.stats["net_wait_ms"] += (time.perf_counter() - t0) * 1000
        out = {}
        off = 0
        buf = m.buf
        for tm in m.header["tensors"]:
            nb = tm["nbytes"]
            if nb == 0:
                continue
            seg = buf[off : off + nb]
            t = _CODEC_OBJ.decode_gpu(
                seg, tm, torch.device("cuda", torch.cuda.current_device())
            )
            if tm.get("sl"):
                from vllm.distributed.parallel_state import get_tp_group

                tpg = get_tp_group()
                # reassemble the full tensor across the intra-machine TP
                # group (same trick the native path uses, NCCL over PCIe)
                full = torch.empty(tm["fn"] * tm["fh"], dtype=t.dtype, device=t.device)
                torch.distributed.all_gather_into_tensor(
                    full, t.reshape(-1), group=tpg.device_group
                )
                t = full.view(tm["fn"], tm["fh"])
            else:
                t = t.view(tm["fn"], tm["fh"])
            out[tm["key"]] = t
            off += nb
        if m.header.get("mg"):
            # merged wire: fabricate the residual the model forward reads.
            # First layer: fused add+RMSNorm(h=s, r=0) -> (norm(s), s) -
            # exactly the pre-merge semantics, zero extra wire bytes.
            out["residual"] = torch.zeros_like(out["hidden_states"])
        self._tok_acc += int(m.header.get("tok", 0))
        base = m.base if m.base is not None and m.base.numel() >= (4 << 20) else None
        if not _ASYNC_TX:
            torch.cuda.synchronize()
            if base is not None:
                self._pool_in.append(base)
        elif base is not None:
            # the H2D copy may still be reading the pinned staging buffer:
            # hand it back to the pool once this event has passed
            ev = torch.cuda.Event()
            ev.record()
            self._in_pend.append((ev, base))
        return out

    # ------------------------------------------------------------ loops
    def _stats_dump(self):
        """Throttled per-(stage,lane) counters for the WAN-bytes-per-token
        audit: one json line every >=2s, appended from whichever loop is
        active on this side (send loop on the sender, recv thread on the
        receiver - its send loop never wakes)."""
        if not _STATS_PATH:
            return
        now = time.time()
        if now - self._last_dump < 2.0:
            return
        self._last_dump = now
        try:
            # dp suffix keeps the two engine cores' counters apart
            suffix = f".s{self.stage}l{self.lane}"
            if getattr(self, "dp", 0):
                suffix += f"d{self.dp}"
            with open(f"{_STATS_PATH}{suffix}", "a") as f:
                f.write(
                    json.dumps(
                        {
                            "t": round(now, 3),
                            "stage": self.stage,
                            "lane": self.lane,
                            "tok": self._tok_acc,
                            **self.stats,
                        }
                    )
                    + "\n"
                )
        except OSError:
            pass

    def _send_loop(self):
        """Slice-level scheduler: at most one in-flight message per priority
        socket; a newly queued hi (decode-bearing) message preempts the lo
        (prefill) stream at slice boundaries. Each socket stays FIFO, so
        the receiver reassembles per socket without interleaving."""
        # event queries/syncs below run on this thread
        torch.cuda.set_device(self.device)
        bucket = _RATE
        last = time.perf_counter()
        # active[prio] = [msg, cursor(si), cursor(so), off]
        active: dict[int, list | None] = {0: None, 1: None}
        while True:
            # pick up new messages without blocking when work is in flight
            if active[0] is None and active[1] is None:
                msg = self.send_q.get()
                if msg is None:
                    return
                active[msg.prio] = [msg, 0, 0, 0]
            else:
                while True:
                    try:
                        msg = self.send_q.get_nowait()
                    except queue.Empty:
                        break
                    if msg is None:
                        return
                    # later arrival on the same prio queue replaces waiting
                    if active[msg.prio] is None:
                        active[msg.prio] = [msg, 0, 0, 0]
                    else:  # only possible for same-prio backlog: queue it
                        self._requeue.append(msg)
            # push any requeued msgs back
            while self._requeue:
                m = self._requeue.popleft()
                if active[m.prio] is None:
                    active[m.prio] = [m, 0, 0, 0]
                else:
                    self.send_q.put(m)
                    break
            # window check before starting a NEW stream (mid-stream slices
            # always proceed; pausing them would not free bytes). unacked
            # counts only bytes of messages already on the wire, so a big
            # enqueued message never blocks its own start.
            for prio in (0, 1):
                st = active[prio]
                if st is not None and st[3] == 0 and _WIN > 0 and self.unacked > _WIN:
                    t0 = time.time()
                    while self.unacked > _WIN:
                        self._drain_acks(0.0)
                        time.sleep(0.002)
                    self.stats["win_wait_s"] += time.time() - t0
            prio = self._pick(active)
            if prio is None:
                # every in-flight stream still waits for its D2H: block on
                # the oldest (nothing else could go on the wire meanwhile)
                for p in (0, 1):
                    if active[p] is not None:
                        active[p][0].ready.synchronize()
                        break
                continue
            st = active[prio]
            msg, si, so, off = st
            n = msg.nbytes
            if off == 0:
                self._dir_sock(msg.prio, forward=True).send(msg.header, zmq.SNDMORE)
            now = time.perf_counter()
            if _RATE > 0:
                bucket = min(bucket + (now - last) * _RATE, _RATE)
                last = now
                if bucket < _SLICE:
                    time.sleep((_SLICE - bucket) / _RATE)
                    continue
                bucket -= _SLICE
            else:
                last = now
            take = min(_SLICE, n - off)
            frame = _slice_bytes(msg.payload, si, so, take)
            st[1], st[2] = _advance(msg.payload, si, so, take)
            st[3] = off + take
            if off == 0:
                with self.plock:
                    self.unacked += n
            self._dir_sock(msg.prio, forward=True).send(
                frame, 0 if st[3] >= n else zmq.SNDMORE
            )
            if st[3] >= n:
                msg.queued.set()
                self.stats["sent_msg"] += 1
                self.stats["sent_bytes"] += n
                active[prio] = None
                # payload buffers are copied into zmq frames; return the
                # whole pool buffers (the payload holds only views - small
                # decode messages used to leak theirs and pinned-alloc a
                # fresh 4 MiB buffer every step)
                self._pool_out.extend(msg.bases)
            self._drain_acks(0.0)
            self._stats_dump()

    @staticmethod
    def _pick(active) -> int | None:
        """hi first, but a stream may only START once its payload is staged
        (async tx); an unstaged hi does not stall the lo stream."""
        for p in (0, 1):
            st = active[p]
            if st is None:
                continue
            ev = st[0].ready
            if st[3] > 0 or ev is None or ev.query():
                return p
        return None

    def _dir_sock(self, prio: int, forward: bool):
        hi = "fwd_hi" if forward else "bwd_hi"
        lo = "fwd_lo" if forward else "bwd_lo"
        return self.sock[hi if prio == 0 else lo]

    def _recv_loop(self):
        poller = zmq.Poller()
        # the hop's receiver (stage _FROM_STAGE + 1) polls the forward
        # sockets; the sender polls the backward ones (acks)
        mine = (
            ("fwd_hi", "fwd_lo")
            if self.stage == _FROM_STAGE + 1
            else ("bwd_hi", "bwd_lo")
        )
        for name in mine:
            poller.register(self.sock[name], zmq.POLLIN)
        while True:
            # stashed frames (handshake races) replay in per-socket order
            # before polling live sockets; a replayed message may continue
            # into frames still queued on the live socket
            replayed = False
            for name in ("fwd_hi", "bwd_hi", "fwd_lo", "bwd_lo"):
                q = self._stash.get(name)
                while q:
                    replayed = True
                    self._recv_frame(name, self.sock[name], replay=q)
            if replayed:
                continue
            try:
                evts = dict(poller.poll(timeout=100))
            except zmq.ZMQError:
                return
            if not evts:
                continue
            # drain hi before lo (decode priority on the receiver);
            # poller events are keyed by socket object
            for name in ("fwd_hi", "bwd_hi", "fwd_lo", "bwd_lo"):
                if self.sock[name] in evts:
                    self._recv_frame(name, self.sock[name])

    def _recv_frame(self, name: str, s, replay=None):
        def next_frame(copy=True):
            if replay:
                return replay.popleft()
            if copy:
                return s.recv()
            # payload: zero-copy frame, read once into the staging buffer
            return s.recv(copy=False).buffer

        hdr = next_frame()
        if hdr in (b"HS", b"HA"):
            if hdr == b"HS":
                with suppress(zmq.Again):
                    s.send(b"HA")
            return
        try:
            h = json.loads(hdr)
        except (ValueError, UnicodeDecodeError):
            return
        if "ack" in h:
            self._complete_ack(int(h["ack"]))
            return
        m = _InMsg(h)
        remaining = m.nbytes
        buf = self._scratch(self._pool_in, remaining) if remaining else None
        dst = buf.numpy() if buf is not None else None
        off = 0
        while remaining > 0:
            got = next_frame(copy=False)
            take = min(len(got), remaining)
            # one memcpy straight into the pinned staging buffer
            dst[off : off + take] = np.frombuffer(got, dtype=np.uint8, count=take)
            off += take
            remaining -= take
        m.base = buf
        m.buf = (
            buf[: m.nbytes] if buf is not None else torch.empty(0, dtype=torch.uint8)
        )
        m.t_ready = time.time()
        m.event.set()
        self.in_ready[h["sid"]] = m
        self.stats["recv_msg"] += 1
        self.stats["recv_bytes"] += m.nbytes
        self._stats_dump()
        self._dir_sock(0, forward=self.stage != _FROM_STAGE + 1).send(
            json.dumps({"ack": str(h["id"])}).encode()
        )

    def _complete_ack(self, mid: int):
        """Ack handling happens on the recv thread: complete the pending
        message (free its window bytes, wake its handle) immediately."""
        with self.plock:
            msg = self.pending.pop(mid, None)
            if msg is not None:
                self.unacked -= msg.nbytes
                msg.event.set()

    def _drain_acks(self, timeout: float):
        """Leftover ack-queue drain (acks are normally completed inline by
        the recv thread); kept so the send loop can still make progress
        if frames ever land here."""
        while True:
            try:
                mid = self.ack_q.get(timeout=timeout if timeout > 0 else 0.0)
            except queue.Empty:
                return
            self._complete_ack(mid)


def _slice_bytes(segs, si: int, so: int, take: int):
    """Zero-copy view of `take` bytes from the segment list at (si, so).
    zmq accepts any buffer protocol object; only slices spanning two
    segments fall back to a copy."""
    seg = segs[si]
    end = min(so + take, seg.numel())
    if end - so == take:
        return seg[so:end].numpy().data  # memoryview, no copy
    b = seg[so:end].numpy().tobytes()
    got = end - so
    if got < take and si + 1 < len(segs):
        b += _slice_bytes(segs, si + 1, 0, take - got)
    return b


def _advance(segs, si: int, so: int, take: int):
    so += take
    while si < len(segs) and so >= segs[si].numel():
        so -= segs[si].numel()
        si += 1
    return si, so


_LINK: PPLink | None = None
_LINK_LOCK = threading.Lock()


def get_link() -> PPLink:
    global _LINK
    with _LINK_LOCK:
        if _LINK is None:
            if not (_BIND_IP and _PEER_IP):
                raise RuntimeError(
                    "HETVLLM_PP_LINK=1 needs HETVLLM_PP_LINK_BIND (stage-0 "
                    "host IP) and HETVLLM_PP_LINK_PEER (stage-1 host IP)"
                )
            from vllm.distributed.parallel_state import get_pp_group, get_tp_group

            lnk = PPLink()
            lnk.setup(
                stage=get_pp_group().rank_in_group, lane=get_tp_group().rank_in_group
            )
            _LINK = lnk
        return _LINK
