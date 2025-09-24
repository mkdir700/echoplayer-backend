"""
会话级播放模型
支持跨窗口连续播放的会话管理
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .window import TranscodeProfile, WindowCache


class SessionStatus(str, Enum):
    """会话状态"""

    ACTIVE = "active"  # 活跃状态
    IDLE = "idle"  # 空闲状态
    EXPIRED = "expired"  # 已过期
    ERROR = "error"  # 错误状态


@dataclass
class WindowReference:
    """窗口引用信息"""

    window_id: int  # 窗口ID
    asset_hash: str  # 资产哈希
    profile_hash: str  # 配置哈希
    start_time: float  # 窗口开始时间（秒）
    duration: float  # 窗口时长（秒）
    end_time: float  # 窗口结束时间（秒）
    segments: list[str] = field(default_factory=list)  # 片段文件名列表
    ready: bool = False  # 是否已准备好
    needs_discontinuity: bool = False  # 是否需要不连续标记

    @property
    def time_range(self) -> tuple[float, float]:
        """获取时间范围"""
        return (self.start_time, self.end_time)

    def overlaps_with(self, other: "WindowReference") -> bool:
        """检查是否与另一个窗口重叠"""
        return not (
            self.end_time <= other.start_time or other.end_time <= self.start_time
        )

    def is_adjacent_to(self, other: "WindowReference") -> bool:
        """检查是否与另一个窗口相邻"""
        return (
            abs(self.end_time - other.start_time) < 0.1
            or abs(other.end_time - self.start_time) < 0.1
        )


@dataclass
class PlaylistSegment:
    """播放列表片段信息"""

    url: str  # 片段URL
    duration: float  # 片段时长
    sequence_number: int  # 序列号
    window_id: int  # 所属窗口ID
    discontinuity_before: bool = False  # 之前是否需要不连续标记

    def to_m3u8_lines(self) -> list[str]:
        """转换为m3u8格式的行"""
        lines = []
        if self.discontinuity_before:
            lines.append("#EXT-X-DISCONTINUITY")
        lines.append(f"#EXTINF:{self.duration:.1f},")
        lines.append(self.url)
        return lines


@dataclass
class SessionPlaylist:
    """会话播放列表"""

    session_id: str  # 会话ID
    target_duration: float  # 目标片段时长
    segments: list[PlaylistSegment] = field(default_factory=list)  # 片段列表
    version: int = 6  # HLS版本
    sequence_number: int = 0  # 媒体序列号
    is_live: bool = False  # 是否为直播

    def get_total_duration(self) -> float:
        """获取播放列表总时长"""
        return sum(segment.duration for segment in self.segments)

    def get_window_count(self) -> int:
        """获取窗口数量"""
        return len({segment.window_id for segment in self.segments})

    def to_m3u8(self) -> str:
        """生成m3u8播放列表内容

        优化版本：使用全局统一EXT-X-MAP，减少DISCONTINUITY标记
        """
        # 计算实际的最大片段时长，避免0.0时长问题
        if self.segments:
            valid_durations = [seg.duration for seg in self.segments if seg.duration > 0]
            max_duration = max(valid_durations, default=self.target_duration)
        else:
            max_duration = self.target_duration

        target_duration = max(int(max_duration) + 1, int(self.target_duration) + 1)

        lines = [
            "#EXTM3U",
            f"#EXT-X-VERSION:{self.version}",
            f"#EXT-X-TARGETDURATION:{target_duration}",
        ]

        # 根据是否完整决定播放列表类型
        if not self.is_live:
            lines.append("#EXT-X-PLAYLIST-TYPE:VOD")
        else:
            lines.append("#EXT-X-PLAYLIST-TYPE:EVENT")

        # 添加全局统一的fMP4初始化段引用
        if self.segments:
            # 使用第一个窗口的init.mp4作为全局初始化段
            # 由于编码参数已统一，所有窗口的init.mp4兼容
            first_segment = self.segments[0]
            init_path = self._get_init_segment_path(first_segment.url)
            if init_path:
                lines.append(f'#EXT-X-MAP:URI="{init_path}"')

            # 添加独立片段标记（用于更好的seek支持）
            lines.append("#EXT-X-INDEPENDENT-SEGMENTS")

            lines.append(f"#EXT-X-MEDIA-SEQUENCE:{self.sequence_number}")

        # 添加片段 - 优化后减少了不必要的DISCONTINUITY
        for segment in self.segments:
            lines.extend(segment.to_m3u8_lines())

        if not self.is_live:
            lines.append("#EXT-X-ENDLIST")

        return "\n".join(lines)

    def _get_init_segment_path(self, segment_url: str) -> str | None:
        """从片段URL推导出初始化段路径"""
        try:
            # 片段URL格式: /api/v1/jit/hls/{asset_hash}/{profile_hash}/{window_id:06d}/{segment_file}
            parts = segment_url.strip("/").split("/")
            if len(parts) >= 7 and parts[0] == "api" and parts[1] == "v1" and parts[2] == "jit":
                asset_hash = parts[4]
                profile_hash = parts[5]
                window_id = parts[6]
                return f"/api/v1/jit/hls/{asset_hash}/{profile_hash}/{window_id}/init.mp4"
        except Exception:
            pass
        return None


@dataclass
class PlaybackSession:
    """播放会话"""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: Path = field(default_factory=lambda: Path())
    asset_hash: str = ""
    profile: TranscodeProfile = field(default_factory=TranscodeProfile)
    profile_hash: str = ""

    # 会话状态
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    expires_at: float = field(
        default_factory=lambda: time.time() + 3600
    )  # 默认1小时过期

    # 播放状态
    current_time: float = 0.0  # 当前播放时间
    duration: float | None = None  # 视频总时长

    # 窗口管理
    windows: dict[int, WindowReference] = field(default_factory=dict)  # 窗口引用
    loaded_windows: set[int] = field(default_factory=set)  # 已加载的窗口
    preloading_windows: set[int] = field(default_factory=set)  # 正在预加载的窗口

    # 播放列表
    playlist: SessionPlaylist | None = None
    playlist_dirty: bool = True  # 播放列表是否需要重新生成

    # 配置
    max_windows_in_playlist: int = 10  # 播放列表中最大窗口数
    preload_window_count: int = 2  # 预加载窗口数量
    ttl_seconds: int = 3600  # 会话生存时间

    def is_expired(self) -> bool:
        """检查会话是否过期"""
        return time.time() > self.expires_at

    def is_idle(self, idle_timeout: int = 300) -> bool:
        """检查会话是否空闲（默认5分钟）"""
        return time.time() - self.last_access > idle_timeout

    def update_access(self) -> None:
        """更新访问时间"""
        self.last_access = time.time()
        if self.status == SessionStatus.IDLE:
            self.status = SessionStatus.ACTIVE

    def get_current_window_id(self) -> int:
        """获取当前时间对应的窗口ID"""
        window_duration = self.profile.window_duration
        return int(self.current_time // window_duration)

    def get_time_range_window_ids(
        self, start_time: float, end_time: float
    ) -> list[int]:
        """获取时间范围对应的窗口ID列表"""
        window_duration = self.profile.window_duration
        start_window = int(start_time // window_duration)
        end_window = int(end_time // window_duration)
        return list(range(start_window, end_window + 1))

    def add_window(self, window_id: int, window_cache: WindowCache) -> None:
        """添加窗口引用"""
        if window_id not in self.windows:
            start_time = window_id * self.profile.window_duration
            end_time = start_time + self.profile.window_duration

            window_ref = WindowReference(
                window_id=window_id,
                asset_hash=window_cache.asset_hash,
                profile_hash=window_cache.profile_hash,
                start_time=start_time,
                duration=self.profile.window_duration,
                end_time=end_time,
                ready=True,
            )

            self.windows[window_id] = window_ref
            self.loaded_windows.add(window_id)
            self.playlist_dirty = True

    def get_required_windows(self) -> list[int]:
        """获取需要的窗口ID列表（基于当前播放位置）"""
        current_window = self.get_current_window_id()
        required_windows = []

        # 当前窗口及前后窗口
        for offset in range(-1, self.preload_window_count + 1):
            window_id = current_window + offset
            if window_id >= 0:
                required_windows.append(window_id)

        return required_windows

    def needs_playlist_rebuild(self) -> bool:
        """检查是否需要重建播放列表"""
        return self.playlist_dirty or self.playlist is None

    def mark_playlist_clean(self) -> None:
        """标记播放列表为最新"""
        self.playlist_dirty = False

    def cleanup_old_windows(self) -> list[int]:
        """清理不再需要的旧窗口"""
        current_window = self.get_current_window_id()
        cleanup_threshold = current_window - self.max_windows_in_playlist

        removed_windows = []
        for window_id in list(self.windows.keys()):
            if window_id < cleanup_threshold:
                del self.windows[window_id]
                self.loaded_windows.discard(window_id)
                removed_windows.append(window_id)
                self.playlist_dirty = True

        return removed_windows


@dataclass
class SessionStats:
    """会话统计信息"""

    total_sessions: int = 0
    active_sessions: int = 0
    idle_sessions: int = 0
    expired_sessions: int = 0
    total_windows_loaded: int = 0
    cache_hit_rate: float = 0.0
    avg_session_duration: float = 0.0
