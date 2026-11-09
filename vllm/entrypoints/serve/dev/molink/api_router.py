# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
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
    return await controller(request).state(True)


@router.post("/reconfigure")
async def reconfigure(body: ReconfigureRequest, request: Request):
    try:
        return await asyncio.shield(
            controller(request).reconfigure(body.boundary, body.mode, body.verify_kv)
        )
    except ReconfigurationValidationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/resource_status")
async def resource_status(request: Request):
    try:
        return await request.app.state.engine_client.engine_core.call_utility_async(
            "molink_resource_state"
        )
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
        try:
            return await control.engine.engine_core.call_utility_async(
                "molink_resize_kv", body.num_blocks
            )
        except ReconfigurationValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


def attach_router(app: FastAPI):
    if envs.VLLM_PP_RECONFIGURE_ENABLED:
        app.include_router(router)
