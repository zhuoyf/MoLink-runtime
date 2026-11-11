# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass


@dataclass(frozen=True)
class SecurityCapabilities:
    node: str
    tee: bool = False
    trusted_gpu: bool = False
    attestation_verified: bool = False
    attestation_expires: float = 0


@dataclass(frozen=True)
class PlacementPlan:
    boundary: int
    environments: tuple[str, str] = ("ree", "ree")
    security_mode: str = "off"


@dataclass(frozen=True)
class SecurityPolicy:
    mode: str = "off"

    def validate(self, plan, capabilities, now):
        if self.mode not in ("off", "trusted") or plan.security_mode != self.mode:
            raise ValueError("unsupported or mismatched security mode")
        if len(plan.environments) != 2:
            raise ValueError("two execution environments required")
        if self.mode == "off":
            if plan.environments != ("ree", "ree"):
                raise ValueError("TEE placement requires a trusted policy")
            return
        if len(capabilities) != 2 or len({c.node for c in capabilities}) != 2:
            raise ValueError("two independently attested nodes required")
        for environment, capability in zip(plan.environments, capabilities):
            if not (
                environment == "tee"
                and capability.tee
                and capability.trusted_gpu
                and capability.attestation_verified
                and capability.attestation_expires > now
            ):
                raise ValueError("trusted GPU execution requires valid attestation")


class SecurityCostModel:
    def cost_seconds(self, plan, state):
        if plan.security_mode != "off":
            raise ValueError("trusted execution cost provider is not installed")
        return 0.0
