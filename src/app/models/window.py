"""
窗口化转码数据模型
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class WindowStatus(str, Enum):
    """窗口状态"""

    PENDING = "pending"  # 等待转码
    RUNNING = "running"  # 转码中
    COMPLETED = "completed"  # 转码完成
    FAILED = "failed"  # 转码失败
    CACHED = "cached"  # 缓存命中


# ffmpeg -hide_banner -y -ss 0 -t 30 -i "/Users/mark/Movies/老友记/老友记.H265.1080P.SE01.01.mkv" \
#   -map 0:v:0 -map 0:a:0 \
#   -c:v libx264 -preset veryfast -pix_fmt yuv420p \
#   -g 48 -keyint_min 48 -sc_threshold 0 \
#   -c:a aac -profile:a aac_low -ar 48000 -ac 2 -b:a 192k \
#   -fflags +genpts -avoid_negative_ts make_zero \
#   -af "aresample=async=1:first_pts=0" \
#   -f hls -hls_time 4 -hls_list_size 0 \
#   -hls_segment_type fmp4 \
#   -hls_fmp4_init_filename "init.mp4" \
#   -hls_segment_filename   "seg_%05d.m4s" \
#   "index.m3u8"


@dataclass
class TranscodeProfile:
    """转码配置"""

    # 视频配置
    video_codec: str = "libx264"  # 视频编码器
    video_preset: str = "veryfast"  # 编码预设
    video_bitrate: str = "2000k"  # 视频码率
    pixel_format: str = "yuv420p"  # 像素格式
    gop_size: int = 48  # GOP大小
    keyint_min: int = 48  # 最小关键帧间隔
    sc_threshold: int = 0  # 场景切换检测阈值

    # 音频配置
    audio_codec: str = "aac"  # 音频编码器
    audio_bitrate: str = "192k"  # 音频码率
    audio_sample_rate: int = 48000  # 采样率
    audio_channels: int = 2  # 音频声道数
    aac_profile: str = "aac_low"  # AAC 配置文件

    # 音频滤镜配置
    audio_filter: str = "aresample=async=1:first_pts=0"  # 音频重采样滤镜

    # FFmpeg 标志配置
    fflags: str = "+genpts"  # FFmpeg 标志
    avoid_negative_ts: str = "make_zero"  # 避免负时间戳

    # HLS 配置
    hls_time: float = 4.0  # 分片时长
    hls_list_size: int = 0  # 播放列表大小（0=无限制）
    hls_segment_type: str = "fmp4"  # 分片类型
    hls_fmp4_init_filename: str = "init.mp4"  # fMP4 初始化文件名
    hls_segment_filename: str = "seg_%05d.m4s"  # 分片文件名模板

    # 窗口配置
    window_segments: int = 3  # 每窗口分片数（基于 hls_time * 3 = 12s）
    window_duration: float = 12.0  # 窗口时长（秒）

    # 版本控制
    version: str = "0"  # 配置版本


@dataclass
class WindowCache:
    """窗口缓存信息"""

    window_id: int  # 窗口ID
    asset_hash: str  # 资产哈希
    profile_hash: str  # 配置哈希
    cache_dir: Path  # 缓存目录
    playlist_path: Path  # m3u8 文件路径
    created_at: float = field(default_factory=time.time)  # 创建时间
    last_access: float = field(default_factory=time.time)  # 最后访问时间
    hit_count: int = 0  # 命中次数
    file_size_bytes: int = 0  # 文件总大小

    def update_access(self) -> None:
        """更新访问时间"""
        self.last_access = time.time()
        self.hit_count += 1

    def is_valid(self) -> bool:
        """检查缓存是否有效"""
        return self.playlist_path.exists()

    def get_age_seconds(self) -> float:
        """获取缓存年龄（秒）"""
        return time.time() - self.created_at

    def get_idle_seconds(self) -> float:
        """获取空闲时间（秒）"""
        return time.time() - self.last_access


@dataclass
class TranscodeWindow:
    """转码窗口"""

    window_id: int  # 窗口ID
    asset_hash: str  # 资产哈希
    profile_hash: str  # 配置哈希
    input_file: Path  # 输入文件
    start_time: float  # 窗口开始时间（秒）
    duration: float  # 窗口时长（秒）
    output_dir: Path  # 输出目录
    status: WindowStatus = WindowStatus.PENDING  # 状态
    created_at: float = field(default_factory=time.time)  # 创建时间
    started_at: float | None = None  # 开始转码时间
    completed_at: float | None = None  # 完成时间
    error_message: str | None = None  # 错误信息
    process: asyncio.subprocess.Process | None = None  # FFmpeg 进程
    future: asyncio.Future | None = None  # 转码任务Future

    @property
    def playlist_path(self) -> Path:
        """获取 m3u8 文件路径"""
        return self.output_dir / "index.m3u8"

    @property
    def is_running(self) -> bool:
        """是否正在转码"""
        return self.status == WindowStatus.RUNNING

    @property
    def is_completed(self) -> bool:
        """是否已完成"""
        return self.status == WindowStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """是否失败"""
        return self.status == WindowStatus.FAILED

    @property
    def duration_seconds(self) -> float:
        """获取转码耗时（秒）"""
        if self.started_at is None:
            return 0.0
        end_time = self.completed_at or time.time()
        return end_time - self.started_at

    def start_transcoding(self) -> None:
        """开始转码"""
        self.status = WindowStatus.RUNNING
        self.started_at = time.time()
        logger.info(f"窗口 {self.window_id} 开始转码")

    def complete_transcoding(self) -> None:
        """完成转码"""
        self.status = WindowStatus.COMPLETED
        self.completed_at = time.time()
        duration = self.duration_seconds
        logger.info(f"窗口 {self.window_id} 转码完成，耗时 {duration:.1f}s")

    def fail_transcoding(self, error: str) -> None:
        """转码失败"""
        self.status = WindowStatus.FAILED
        self.completed_at = time.time()
        self.error_message = error
        duration = self.duration_seconds
        logger.error(f"窗口 {self.window_id} 转码失败，耗时 {duration:.1f}s: {error}")

    def cleanup(self) -> None:
        """清理资源"""
        if self.process and not self.process.returncode:
            try:
                self.process.kill()
                logger.info(f"窗口 {self.window_id} FFmpeg 进程已终止")
            except Exception as e:
                logger.warning(f"终止 FFmpeg 进程失败: {e}")

        if self.future and not self.future.done():
            self.future.cancel()
            logger.info(f"窗口 {self.window_id} 转码任务已取消")


@dataclass
class CacheStats:
    """缓存统计信息"""

    total_windows: int = 0  # 总窗口数
    total_size_bytes: int = 0  # 总大小（字节）
    total_hit_count: int = 0  # 总命中次数
    avg_window_size: float = 0.0  # 平均窗口大小
    cache_hit_rate: float = 0.0  # 缓存命中率
    oldest_window_age: float = 0.0  # 最老窗口年龄
    lru_candidates: int = 0  # LRU 候选数量
