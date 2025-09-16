"""
转码服务配置管理
"""

import logging
import os
import platform
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class Settings:
    """转码服务配置"""

    # 服务配置
    HOST: str = "127.0.0.1"
    PORT: int = 8799
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # 会话存储配置
    SESSIONS_ROOT: str = os.getenv(
        "SESSIONS_ROOT",
        str(Path(tempfile.gettempdir()) / "echoplayer-transcoder" / "sessions"),
    )

    # v1 缓存存储配置
    v1_CACHE_ROOT: str = os.getenv(
        "v1_CACHE_ROOT",
        str(Path(__file__).parent.parent.parent / "cache" / "v1_hls"),
    )

    # HLS 配置
    HLS_TIME: float = 1.5  # 分片时长（秒）
    HLS_LIST_SIZE: int = 3  # 播放器缓冲片段数
    GOP_SECONDS: float = 1.5  # GOP 时长，与 HLS_TIME 对齐

    # 并发控制
    MAX_CONCURRENT: int = 3  # 最大并发转码数
    SESSION_TTL: int = 120  # 会话空闲回收时间（秒）

    # FFmpeg 配置
    FFMPEG_EXECUTABLE: str = os.getenv("FFMPEG_PATH", "ffmpeg")
    FFPROBE_EXECUTABLE: str = os.getenv("FFPROBE_PATH", "ffprobe")
    PREFER_HW: bool = os.getenv("PREFER_HW", "true").lower() == "true"

    # 性能配置
    SEEK_BUFFER: float = 0.5  # seek 前置缓冲时间（秒）
    SESSION_REUSE_TOLERANCE: float = 30.0  # 会话复用容差时间（秒）
    CLEANUP_INTERVAL: int = 30  # 清理检查间隔（秒）

    @classmethod
    def ensure_sessions_dir(cls) -> Path:
        """确保会话目录存在"""
        sessions_path = Path(cls.SESSIONS_ROOT)
        sessions_path.mkdir(parents=True, exist_ok=True)
        return sessions_path

    @classmethod
    def get_ram_disk_path(cls) -> str | None:
        """获取 RAM 磁盘路径（如果可用）"""

        system = platform.system().lower()

        if system == "linux":
            ram_path = "/dev/shm/echoplayer-transcoder"
            if Path("/dev/shm").exists():
                return ram_path
        elif system == "darwin":  # macOS
            # 检查是否有现有的 RAM 磁盘挂载
            ram_paths = ["/Volumes/RAMDisk", "/tmp/ramdisk"]
            for path in ram_paths:
                if Path(path).exists():
                    return f"{path}/echoplayer-transcoder"
        elif system == "windows":
            # Windows RAM 磁盘通常是 R:\ 等
            ram_drives = ["R:\\", "T:\\", "Z:\\"]
            for drive in ram_drives:
                if Path(drive).exists():
                    return f"{drive}echoplayer-transcoder"

        return None

    @classmethod
    def auto_configure_storage(cls) -> str:
        """自动配置存储路径，优先使用 RAM 磁盘"""
        ram_path = cls.get_ram_disk_path()
        if ram_path:
            logger.info("✅ 检测到 RAM 磁盘，使用路径: %s", ram_path)
            cls.SESSIONS_ROOT = ram_path
            return ram_path
        logger.info("ℹ️  未检测到 RAM 磁盘，使用临时目录: %s", cls.SESSIONS_ROOT)
        return cls.SESSIONS_ROOT


# 全局配置实例
settings = Settings()
settings.auto_configure_storage()
