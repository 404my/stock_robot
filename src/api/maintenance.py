"""维护任务的状态与手动触发接口。"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from api.auth import is_trusted_client
from maintenance.runner import MaintenanceRunner
from utils.config import DEFAULT_CONFIG


def create_maintenance_router(runner: MaintenanceRunner | None = None) -> APIRouter:
    router = APIRouter()

    @router.get("/api/v1/maintenance/status")
    async def maintenance_status():
        if runner is not None:
            return JSONResponse(runner.status())
        maintenance = DEFAULT_CONFIG["maintenance"]
        return JSONResponse({
            "config": {
                "report_cleanup_enabled": maintenance["report_cleanup_enabled"],
                "sweep_interval_minutes": maintenance["sweep_interval_minutes"],
            },
            "running": False,
            "last_run": None,
        })

    @router.post("/api/v1/maintenance/sweep")
    async def maintenance_sweep(request: Request):
        # 删除文件只允许本机页面或运维脚本触发，避免跨站请求借浏览器删报告。
        if not is_trusted_client(request):
            raise HTTPException(status_code=403, detail="仅允许本机或已登录的同源页面触发报告清理")
        if runner is None:
            raise HTTPException(status_code=503, detail="维护任务未启动")
        return JSONResponse(await asyncio.to_thread(runner.run_once))

    return router
