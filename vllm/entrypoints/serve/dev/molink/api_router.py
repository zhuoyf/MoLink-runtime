# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, StrictInt

from vllm.distributed.pp_reconfigure.control import ReconfigurationController

router = APIRouter(prefix="/molink")


class ReconfigureRequest(BaseModel):
    boundary: StrictInt


def controller(request):
    if not hasattr(request.app.state, "molink_controller"):
        request.app.state.molink_controller = ReconfigurationController(
            request.app.state.engine_client,
            os.environ.get(
                "HETVLLM_PP_RECONFIGURE_LOG",
                "/workspace/molink-reconfiguration-results/migrations",
            ),
        )
    return request.app.state.molink_controller


@router.get("/monitor")
async def monitor(request: Request):
    return await controller(request).state(True)


@router.post("/reconfigure")
async def reconfigure(body: ReconfigureRequest, request: Request):
    try:
        return await asyncio.shield(controller(request).reconfigure(body.boundary))
    except ValueError as exc:
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


def attach_router(app: FastAPI):
    if os.environ.get("HETVLLM_PP_RECONFIGURE") == "1":
        app.include_router(router)
