from enum import Enum


class TranscodeStatus(str, Enum):
    """转码状态枚举"""

    STARTING = "starting"  # 正在启动
    RUNNING = "running"  # 转码中
    READY = "ready"  # 准备就绪，可播放
    FAILED = "failed"  # 转码失败
    STOPPED = "stopped"  # 已停止
    EXPIRED = "expired"  # 会话过期
