"""
v1 预取 API 路由
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import PreloadStrategy, get_preload_strategy
from app.models.window import TranscodeProfile
from app.schemas.jit_request import PreloadRequest, PreloadResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/preload", tags=["预取策略"])


@router.post("/ranges", response_model=PreloadResponse)
async def preload_time_ranges(
    request: PreloadRequest,
    preload_strategy: PreloadStrategy = Depends(get_preload_strategy),
):
    """
    预取指定时间范围的窗口

    - 支持多个时间范围
    - 自定义优先级
    - 返回排队和缓存统计
    """
    try:
        # 检查文件是否存在
        file_path = Path(request.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=404, detail=f"文件不存在: {request.file_path}"
            )

        # 构建转码配置
        profile = TranscodeProfile()
        if request.video_codec:
            profile.video_codec = request.video_codec
        if request.video_preset:
            profile.video_preset = request.video_preset
        if request.video_bitrate:
            profile.video_bitrate = request.video_bitrate

        # 执行预取
        queued_windows, cached_windows = await preload_strategy.preload_time_ranges(
            request.file_path, request.time_ranges, profile, request.priority
        )

        # 估算完成时间（简单估算：每个窗口10秒）
        estimated_time = None
        if queued_windows > 0:
            estimated_time = queued_windows * 10.0  # 10秒每窗口

        return PreloadResponse(
            success=True,
            queued_windows=queued_windows,
            cached_windows=cached_windows,
            estimated_time=estimated_time,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"预取时间范围失败: {e}")
        raise HTTPException(status_code=500, detail=f"预取失败: {str(e)}")


@router.post("/cancel")
async def cancel_preload_tasks(
    file_path: str | None = None,
    preload_strategy: PreloadStrategy = Depends(get_preload_strategy),
):
    """
    取消预取任务

    - 可选择性取消指定文件的预取任务
    - 返回取消的任务数量
    """
    try:
        cancelled_count = preload_strategy.cancel_preload_tasks(file_path)

        return {
            "success": True,
            "cancelled_tasks": cancelled_count,
            "file_path": file_path,
        }

    except Exception as e:
        logger.error(f"取消预取任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")


@router.get("/status")
async def get_preload_status(
    preload_strategy: PreloadStrategy = Depends(get_preload_strategy),
):
    """
    获取预取状态

    - 显示活跃任务数
    - 播放模式分析统计
    """
    try:
        status = preload_strategy.get_preload_status()
        return {"success": True, **status}

    except Exception as e:
        logger.error(f"获取预取状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"状态查询失败: {str(e)}")
