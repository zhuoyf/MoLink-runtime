# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


def bucket(value):
    return 1 << max(0, math.ceil(math.log2(max(1, value))))


@dataclass
class Estimate:
    value: float
    updated: float
    count: int = 1
    relative_error: float = 0.0


class PerformanceModel:
    """Conditioned EWMA; prediction error is measured before each update."""

    def __init__(self, alpha=0.2):
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.estimates: dict[str, Estimate] = {}
        self.conditions: dict = {}
        self.compute_baseline: dict[str, float] = {}
        self.rejected = 0

    def update(self, key, observed, timestamp):
        if not all(math.isfinite(v) and v >= 0 for v in (observed, timestamp)):
            self.rejected += 1
            return False
        old = self.estimates.get(key)
        if old and timestamp <= old.updated:
            self.rejected += 1
            return False
        baseline = None
        if key.startswith("compute/") and self.compute_baseline:
            baseline = self._compute_curve(key, self.compute_baseline)
        if baseline:
            self.update(self._scale_key(key), observed / baseline, timestamp)
        if old:
            error = abs(old.value - observed) / max(observed, 1e-9)
            old.relative_error = (
                1 - self.alpha
            ) * old.relative_error + self.alpha * error
            old.value = (1 - self.alpha) * old.value + self.alpha * observed
            old.updated = timestamp
            old.count += 1
        else:
            self.estimates[key] = Estimate(observed, timestamp)
        return True

    def predict(self, key):
        estimate = self.estimates.get(key)
        return estimate.value if estimate else None

    @staticmethod
    def compute_key(stage, layers, phase, batch, context, tokens):
        return (
            f"compute/{stage}/{layers}/{phase}/"
            f"{bucket(batch)}/{bucket(context)}/{bucket(tokens)}"
        )

    @staticmethod
    def _scale_key(key):
        parts = key.split("/")
        return "/".join(["compute_scale", parts[1], *parts[3:]])

    def freeze_compute_baseline(self):
        self.compute_baseline = {
            k: e.value for k, e in self.estimates.items() if k.startswith("compute/")
        }
        for key in self.compute_baseline:
            self.estimates[self._scale_key(key)] = Estimate(1.0, 0)

    @staticmethod
    def _compute_curve(key, values):
        parts = key.split("/")
        points = [
            (int(k.split("/")[2]), value)
            for k, value in values.items()
            if k.startswith(f"compute/{parts[1]}/") and k.split("/")[3:] == parts[3:]
        ]
        if not points:
            return None
        layers = int(parts[2])
        if len(points) == 1:
            return points[0][1] * layers / points[0][0]
        xmean = sum(x for x, _ in points) / len(points)
        ymean = sum(y for _, y in points) / len(points)
        variance = sum((x - xmean) ** 2 for x, _ in points)
        slope = max(0, sum((x - xmean) * (y - ymean) for x, y in points) / variance)
        intercept = ymean - slope * xmean
        if intercept < 0:
            intercept = 0
            slope = sum(x * y for x, y in points) / sum(x * x for x, _ in points)
        return intercept + slope * layers

    def compute(self, stage, layers, phase, batch, context, tokens):
        key = self.compute_key(stage, layers, phase, batch, context, tokens)
        baseline = self._compute_curve(key, self.compute_baseline)
        if baseline is not None:
            return baseline * (self.predict(self._scale_key(key)) or 1.0)
        return self._compute_curve(
            key,
            {k: e.value for k, e in self.estimates.items() if k.startswith("compute/")},
        )

    def migration(self, key, metric):
        exact = self.predict(key + "/" + metric)
        if exact is not None:
            return exact
        _, direction, layers, size = key.split("/")
        candidates = []
        for name, estimate in self.estimates.items():
            parts = name.split("/")
            if (
                len(parts) == 5
                and parts[:2] == ["migration", direction]
                and parts[4] == metric
            ):
                distance = abs(math.log2(int(parts[3]) / int(size))) + abs(
                    int(parts[2]) - int(layers)
                )
                value = estimate.value
                if metric != "interference":
                    value *= int(layers) / int(parts[2])
                    if metric == "total":
                        bandwidth = self.predict("network/bytes_per_second")
                        if bandwidth:
                            value += max(0, int(size) - int(parts[3])) / bandwidth
                candidates.append((distance, value))
        return min(candidates)[1] if candidates else None

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "alpha": self.alpha,
                    "conditions": self.conditions,
                    "compute_baseline": self.compute_baseline,
                    "estimates": {k: asdict(v) for k, v in self.estimates.items()},
                },
                indent=2,
                allow_nan=False,
            )
        )
        temporary.replace(target)

    @classmethod
    def load(cls, path, conditions=None):
        data = json.loads(Path(path).read_text())
        if data["schema"] != 1 or (
            conditions is not None and data["conditions"] != conditions
        ):
            raise ValueError("profile schema or test conditions do not match")
        model = cls(data["alpha"])
        model.conditions = data["conditions"]
        for key, values in data["estimates"].items():
            estimate = Estimate(**values)
            if not all(
                math.isfinite(x) and x >= 0
                for x in (estimate.value, estimate.updated, estimate.relative_error)
            ):
                raise ValueError("invalid saved estimate")
            model.estimates[key] = estimate
        model.compute_baseline = data.get("compute_baseline", {})
        return model


@dataclass
class NodeState:
    stage: int
    layers: int
    memory_budget: int
    non_kv_bytes: int
    weight_bytes_per_layer: int
    kv_bytes_per_layer_block: float
    num_blocks: int
    used_blocks: int
    gpu_utilization: float = 0
    block_cap: int | None = None
    kv_payload_bytes_per_layer_block: float | None = None

    @property
    def transfer_bytes_per_block(self):
        return (
            self.kv_payload_bytes_per_layer_block
            if self.kv_payload_bytes_per_layer_block is not None
            else self.kv_bytes_per_layer_block
        )

    def __post_init__(self):
        values = (
            self.memory_budget,
            self.non_kv_bytes,
            self.weight_bytes_per_layer,
            self.kv_bytes_per_layer_block,
            self.num_blocks,
            self.used_blocks,
            self.gpu_utilization,
        )
        if not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError("invalid node measurement")
        if (
            self.stage not in (0, 1)
            or self.layers <= 0
            or self.num_blocks < 2
            or self.kv_bytes_per_layer_block <= 0
        ):
            raise ValueError("invalid node layout")


@dataclass
class RuntimeState:
    timestamp: float
    version: int
    boundary: int
    total_layers: int
    nodes: list[NodeState]
    batch: int
    context: int
    tokens: int
    phase: str
    growth_blocks_per_second: float = 0
    preemptions: int = 0
    waiting: int = 0
    paused: bool = False
    activation_bytes: int = 0
    capabilities: list = field(default_factory=list)
    running: int | None = None
