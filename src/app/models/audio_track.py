"""
音频轨道数据模型
用于支持音频整体转码和时间戳连续性
"""

import time
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, computed_field

from app.config.schemas import AudioConfig


class AudioTrackStatus(str, Enum):
    """音频轨道状态"""

    PENDING = "pending"  # 等待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"  # 处理完成
    FAILED = "failed"  # 处理失败
    CACHED = "cached"  # 缓存命中


class AudioSegment(BaseModel):
    """音频分片信息"""

    segment_index: int = Field(description="分片索引")
    start_time: float = Field(description="开始时间（秒）")
    duration: float = Field(gt=0, description="分片时长（秒）")
    file_path: Path = Field(description="分片文件路径")
    file_size: int = Field(default=0, ge=0, description="文件大小（字节）")
    created_at: float = Field(default_factory=time.time, description="创建时间")

    class Config:
        arbitrary_types_allowed = True

    @computed_field
    @property
    def end_time(self) -> float:
        """结束时间"""
        return self.start_time + self.duration

    @computed_field
    @property
    def url(self) -> str:
        """分片URL（相对路径）"""
        return f"audio_seg_{self.segment_index:05d}.aac"

    def exists(self) -> bool:
        """检查分片文件是否存在"""
        return self.file_path.exists()


class AudioTrackProfile(BaseModel):
    """音频轨道转码配置"""

    # 音频编码设置
    codec: str = Field(description="音频编码器")
    bitrate: str = Field(description="音频码率")
    sample_rate: int = Field(gt=0, description="采样率")
    channels: int = Field(ge=1, le=8, description="声道数")
    profile: str = Field(description="音频配置文件")

    # 分片设置
    segment_duration: float = Field(gt=0, description="分片时长（秒）")

    # 版本控制
    version: str = Field(description="配置版本")

    @classmethod
    def from_config(cls, audio_config: AudioConfig) -> "AudioTrackProfile":
        """从配置创建音频轨道配置"""
        return cls(
            codec=audio_config.codec,
            bitrate=audio_config.bitrate,
            sample_rate=audio_config.sample_rate,
            channels=audio_config.channels,
            profile=audio_config.profile,
            segment_duration=audio_config.segment_duration,
            version="1",
        )

    def to_dict(self) -> dict:
        """转换为字典，用于哈希计算"""
        return {
            "codec": self.codec,
            "bitrate": self.bitrate,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "profile": self.profile,
            "segment_duration": self.segment_duration,
            "version": self.version,
        }


class AudioTrack(BaseModel):
    """音频轨道"""

    # 基本信息
    asset_hash: str = Field(description="原始文件哈希")
    profile_hash: str = Field(description="配置哈希")
    input_file: Path = Field(description="输入文件路径")
    output_dir: Path = Field(description="输出目录")
    duration: float = Field(gt=0, description="总时长（秒）")

    # 状态信息
    status: AudioTrackStatus = Field(
        default=AudioTrackStatus.PENDING, description="处理状态"
    )
    created_at: float = Field(default_factory=time.time, description="创建时间")
    started_at: float | None = Field(default=None, description="开始处理时间")
    completed_at: float | None = Field(default=None, description="完成时间")
    error_message: str | None = Field(default=None, description="错误信息")

    # 分片信息
    segments: list[AudioSegment] = Field(
        default_factory=list, description="音频分片列表"
    )
    segment_count: int = Field(default=0, ge=0, description="分片数量")
    total_size: int = Field(default=0, ge=0, description="总文件大小（字节）")

    # 配置信息
    profile: AudioTrackProfile = Field(description="音频轨道配置")

    class Config:
        arbitrary_types_allowed = True

    @property
    def track_file_path(self) -> Path:
        """完整音频轨道文件路径"""
        return self.output_dir / "audio_track.aac"

    @property
    def metadata_file_path(self) -> Path:
        """元数据文件路径"""
        return self.output_dir / "audio_track.meta.json"

    @property
    def is_processing(self) -> bool:
        """是否正在处理"""
        return self.status == AudioTrackStatus.PROCESSING

    @property
    def is_completed(self) -> bool:
        """是否已完成"""
        return self.status == AudioTrackStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """是否失败"""
        return self.status == AudioTrackStatus.FAILED

    @property
    def processing_duration(self) -> float:
        """处理耗时（秒）"""
        if self.started_at is None:
            return 0.0
        end_time = self.completed_at or time.time()
        return end_time - self.started_at

    def start_processing(self) -> None:
        """开始处理"""
        self.status = AudioTrackStatus.PROCESSING
        self.started_at = time.time()

    def complete_processing(self) -> None:
        """完成处理"""
        self.status = AudioTrackStatus.COMPLETED
        self.completed_at = time.time()

    def fail_processing(self, error: str) -> None:
        """处理失败"""
        self.status = AudioTrackStatus.FAILED
        self.completed_at = time.time()
        self.error_message = error

    def add_segment(self, segment: AudioSegment) -> None:
        """添加音频分片"""
        self.segments.append(segment)
        self.segment_count = len(self.segments)
        if segment.file_size > 0:
            self.total_size += segment.file_size

    def get_segment_by_time(self, time_seconds: float) -> AudioSegment | None:
        """根据时间获取对应的音频分片"""
        for segment in self.segments:
            if segment.start_time <= time_seconds < segment.end_time:
                return segment
        return None

    def get_segments_in_range(
        self, start_time: float, end_time: float
    ) -> list[AudioSegment]:
        """获取时间范围内的所有分片"""
        segments = []
        for segment in self.segments:
            # 检查分片是否与时间范围有重叠
            if segment.start_time < end_time and segment.end_time > start_time:
                segments.append(segment)  # noqa: PERF401
        return sorted(segments, key=lambda s: s.start_time)

    def validate_segments(self) -> bool:
        """验证分片完整性"""
        if not self.segments:
            return False

        # 检查分片连续性
        self.segments.sort(key=lambda s: s.start_time)

        for i in range(len(self.segments)):
            segment = self.segments[i]

            # 检查文件是否存在
            if not segment.exists():
                return False

            # 检查时间连续性（允许小误差）
            if i > 0:
                prev_segment = self.segments[i - 1]
                time_gap = segment.start_time - prev_segment.end_time
                if abs(time_gap) > 0.1:  # 允许100ms误差
                    return False

        return True

    def cleanup(self) -> None:
        """清理音频轨道文件"""
        try:
            # 删除完整轨道文件
            if self.track_file_path.exists():
                self.track_file_path.unlink()

            # 删除分片文件
            for segment in self.segments:
                if segment.file_path.exists():
                    segment.file_path.unlink()

            # 删除元数据文件
            if self.metadata_file_path.exists():
                self.metadata_file_path.unlink()

        except Exception:
            # 静默失败，避免影响主流程
            pass


class AudioTrackCache(BaseModel):
    """音频轨道缓存信息"""

    asset_hash: str = Field(description="资产哈希")
    profile_hash: str = Field(description="配置哈希")
    track_dir: Path = Field(description="轨道目录")
    duration: float = Field(description="总时长")
    segment_count: int = Field(ge=0, description="分片数量")
    total_size: int = Field(default=0, ge=0, description="总大小（字节）")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    last_access: float = Field(default_factory=time.time, description="最后访问时间")
    hit_count: int = Field(default=0, ge=0, description="命中次数")

    class Config:
        arbitrary_types_allowed = True

    def update_access(self) -> None:
        """更新访问时间"""
        self.last_access = time.time()
        self.hit_count += 1

    def is_valid(self) -> bool:
        """检查缓存是否有效"""
        return self.track_dir.exists() and (self.track_dir / "audio_track.aac").exists()

    def get_age_seconds(self) -> float:
        """获取缓存年龄（秒）"""
        return time.time() - self.created_at

    def get_idle_seconds(self) -> float:
        """获取空闲时间（秒）"""
        return time.time() - self.last_access


class AudioTrackStats(BaseModel):
    """音频轨道统计信息"""

    total_tracks: int = Field(default=0, ge=0, description="总轨道数")
    total_size_bytes: int = Field(default=0, ge=0, description="总大小（字节）")
    total_segments: int = Field(default=0, ge=0, description="总分片数")
    total_hit_count: int = Field(default=0, ge=0, description="总命中次数")
    avg_track_size: float = Field(default=0.0, ge=0, description="平均轨道大小")
    cache_hit_rate: float = Field(default=0.0, ge=0, le=1, description="缓存命中率")
    oldest_track_age: float = Field(default=0.0, ge=0, description="最老轨道年龄")
