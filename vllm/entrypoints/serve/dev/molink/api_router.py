# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, StrictInt

import vllm.envs as envs
from vllm.distributed.pp_reconfigure.control import ReconfigurationController
from vllm.distributed.pp_reconfigure.errors import ReconfigurationValidationError

router = APIRouter(prefix="/molink")


class ReconfigureRequest(BaseModel):
    boundary: StrictInt
    mode: Literal["sync", "async", "async_no_patching"] = "sync"
    verify_kv: bool = False
    expected_version: StrictInt | None = None
    expected_boundary: StrictInt | None = None


class ResizeRequest(BaseModel):
    num_blocks: StrictInt


def controller(request):
    if not hasattr(request.app.state, "molink_controller"):
        request.app.state.molink_controller = ReconfigurationController(
            request.app.state.engine_client,
            envs.VLLM_PP_RECONFIGURE_LOG_DIR,
        )
    return request.app.state.molink_controller


@router.get("/monitor")
async def monitor(request: Request):
    control = controller(request)
    result = await control.state(True)
    result["configuration_version"] = control.version
    return result


@router.post("/reconfigure")
async def reconfigure(body: ReconfigureRequest, request: Request):
    try:
        return await asyncio.shield(
            controller(request).reconfigure(
                body.boundary,
                body.mode,
                body.verify_kv,
                expected_version=body.expected_version,
                expected_boundary=body.expected_boundary,
            )
        )
    except ReconfigurationValidationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/resource_status")
async def resource_status(request: Request):
    try:
        result = await request.app.state.engine_client.engine_core.call_utility_async(
            "molink_resource_state"
        )
        result["configuration_version"] = controller(request).version
        return result
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/resize_kv")
async def resize_kv(body: ResizeRequest, request: Request):
    control = controller(request)
    if control.lock.locked():
        raise HTTPException(
            status_code=409, detail="a reconfiguration is already running"
        )
    async with control.lock:
        control.version += 1
        try:
            return await control.engine.engine_core.call_utility_async(
                "molink_resize_kv", body.num_blocks
            )
        except ReconfigurationValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


class SimulationRequest(BaseModel):
    stages: list[dict]
    target_boundary: StrictInt | None = None


@router.post("/heterogeneity")
async def heterogeneity(body: SimulationRequest, request: Request):
    if os.environ.get("MOLINK_HETEROGENEITY_SIMULATION") != "1":
        raise HTTPException(
            status_code=403, detail="experimental simulation is disabled"
        )
    control = controller(request)
    async with control.lock:
        try:
            result = await control.engine.engine_core.call_utility_async(
                "molink_simulate", body.stages, body.target_boundary
            )
            control.version += 1
            result["configuration_version"] = control.version
            return result
        except (ValueError, ReconfigurationValidationError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


def attach_router(app: FastAPI):
    if envs.VLLM_PP_RECONFIGURE_ENABLED:
        app.include_router(router)
        if profile := os.environ.get("VLLM_PP_ADAPTIVE_PROFILE"):
            from vllm.distributed.pp_reconfigure.adaptive.optimizer import (
                OptimizerConfig,
            )
            from vllm.distributed.pp_reconfigure.adaptive.runtime import (
                AdaptiveScheduler,
            )

            async def start_adaptive():
                control = ReconfigurationController(
                    app.state.engine_client, envs.VLLM_PP_RECONFIGURE_LOG_DIR
                )
                app.state.molink_controller = control
                scheduler = AdaptiveScheduler(
                    control,
                    profile,
                    os.environ.get(
                        "VLLM_PP_ADAPTIVE_LOG_DIR", "benchmark_results/adaptive"
                    ),
                    OptimizerConfig(
                        **{
                            "stack_size": envs.VLLM_PP_RECONFIGURE_STACK_SIZE,
                            **json.loads(
                                os.environ.get("VLLM_PP_ADAPTIVE_CONFIG", "{}")
                            ),
                        }
                    ),
                    interval=float(os.environ.get("VLLM_PP_ADAPTIVE_INTERVAL", "5")),
                )
                conditions = scheduler.model.conditions
                if (
                    conditions.get("stack_size") != envs.VLLM_PP_RECONFIGURE_STACK_SIZE
                    or scheduler.optimizer.config.stack_size
                    != envs.VLLM_PP_RECONFIGURE_STACK_SIZE
                ):
                    raise ValueError("adaptive profile layer stacking does not match")
                config_file = (
                    Path(app.state.vllm_config.model_config.model) / "config.json"
                )
                if (
                    conditions.get("config_sha256")
                    != hashlib.sha256(config_file.read_bytes()).hexdigest()
                ):
                    raise ValueError(
                        "adaptive profile model configuration does not match"
                    )
                if conditions.get("dtype") != str(
                    app.state.vllm_config.model_config.dtype
                ).removeprefix("torch."):
                    raise ValueError("adaptive profile dtype does not match")
                if "VLLM_PP_ADAPTIVE_ALPHA" in os.environ:
                    from vllm.distributed.pp_reconfigure.adaptive.model import (
                        PerformanceModel,
                    )

                    scheduler.model.alpha = PerformanceModel(
                        float(os.environ["VLLM_PP_ADAPTIVE_ALPHA"])
                    ).alpha
                app.state.molink_adaptive = scheduler
                scheduler.start()

            async def stop_adaptive():
                if scheduler := getattr(app.state, "molink_adaptive", None):
                    await scheduler.stop()

            previous_lifespan = app.router.lifespan_context

            @asynccontextmanager
            async def adaptive_lifespan(application):
                async with previous_lifespan(application):
                    await start_adaptive()
                    try:
                        yield
                    finally:
                        await stop_adaptive()

            app.router.lifespan_context = adaptive_lifespan


@router.get("/adaptive/status")
async def adaptive_status(request: Request):
    scheduler = getattr(request.app.state, "molink_adaptive", None)
    return scheduler.status() if scheduler else {"running": False}
