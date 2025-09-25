"""
会话管理器
负责管理播放会话的生命周期和状态
"""

import asyncio
import json
import logging
import time
from pathlib import Path

from app.models.session import (
    PlaybackSession,
    SessionStats,
    SessionStatus,
)
from app.models.window import TranscodeProfile
from app.services.jit_transcoder import JITTranscoder
from app.services.session_playlist_generator import SessionPlaylistGenerator
from app.settings import settings
from app.utils.hash import calculate_asset_hash, calculate_profile_hash

logger = logging.getLogger(__name__)


class SessionManager:
    """会话管理器"""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):  # noqa: ARG004
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, jit_transcoder: JITTranscoder):
        if self._initialized:
            return

        self.jit_transcoder = jit_transcoder
        self.playlist_generator = SessionPlaylistGenerator(jit_transcoder)

        # 会话存储
        self.sessions: dict[str, PlaybackSession] = {}
        self.sessions_by_file: dict[str, set[str]] = {}  # 文件路径 -> 会话ID集合

        # 配置
        self.session_timeout = 3600  # 会话超时时间（秒）
        self.idle_timeout = 300  # 空闲超时时间（秒）
        self.cleanup_interval = 60  # 清理检查间隔（秒）

        # 背景任务
        self.cleanup_task: asyncio.Task | None = None
        self.running = True

        logger.info("会话管理器初始化完成")
        self._initialized = True

    async def _get_video_duration(self, file_path: Path) -> float | None:
        """获取视频文件时长

        Args:
            file_path: 视频文件路径

        Returns:
            视频时长（秒），如果获取失败返回None
        """
        try:
            cmd = [
                settings.FFPROBE_EXECUTABLE,
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(file_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"ffprobe获取视频时长失败: {stderr.decode()}")
                return None

            result = json.loads(stdout.decode())
            duration = float(result["format"]["duration"])
            logger.debug(f"视频 {file_path} 时长: {duration:.1f}s")
            return duration

        except Exception as e:
            logger.error(f"获取视频时长失败: {e}")
            return None

    async def create_session(
        self,
        file_path: Path,
        profile: TranscodeProfile | None = None,
        initial_time: float = 0.0,
    ) -> PlaybackSession:
        """
        创建新的播放会话

        Args:
            file_path: 视频文件路径
            profile: 转码配置
            initial_time: 初始播放时间

        Returns:
            PlaybackSession: 创建的会话
        """
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if profile is None:
            profile = TranscodeProfile()

        # 计算哈希值
        asset_hash = calculate_asset_hash(file_path)
        profile_hash = calculate_profile_hash(profile.__dict__, profile.version)

        # 获取视频时长
        duration = await self._get_video_duration(file_path)

        # 创建会话
        session = PlaybackSession(
            file_path=file_path,
            asset_hash=asset_hash,
            profile=profile,
            profile_hash=profile_hash,
            current_time=initial_time,
            duration=duration,
            expires_at=time.time() + self.session_timeout,
        )

        # 存储会话
        self.sessions[session.session_id] = session

        # 按文件路径索引
        file_key = str(file_path)
        if file_key not in self.sessions_by_file:
            self.sessions_by_file[file_key] = set()
        self.sessions_by_file[file_key].add(session.session_id)

        logger.info(f"创建会话 {session.session_id}: {file_path} @ {initial_time}s")

        # 预加载初始窗口
        await self._preload_initial_windows(session)

        # 启动清理任务（如果还未启动）
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        return session

    async def get_session(self, session_id: str) -> PlaybackSession | None:
        """获取会话"""
        session = self.sessions.get(session_id)
        if session:
            session.update_access()
        return session

    async def update_session_time(
        self, session_id: str, current_time: float
    ) -> PlaybackSession | None:
        """
        更新会话播放时间

        Args:
            session_id: 会话ID
            current_time: 当前播放时间

        Returns:
            更新后的会话，如果不存在返回None
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        old_time = session.current_time
        session.current_time = current_time

        logger.debug(
            f"会话 {session_id} 时间更新: {old_time:.1f}s -> {current_time:.1f}s"
        )

        # 检查是否需要预加载新窗口
        await self._ensure_windows_available(session)

        return session

    async def seek_session(
        self, session_id: str, seek_time: float
    ) -> PlaybackSession | None:
        """
        会话内seek操作

        Args:
            session_id: 会话ID
            seek_time: 目标时间

        Returns:
            更新后的会话
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        logger.info(
            f"会话 {session_id} seek: {session.current_time:.1f}s -> {seek_time:.1f}s"
        )

        # 更新播放列表（处理跨窗口seek）
        updated_playlist = await self.playlist_generator.update_playlist_for_seek(
            session, seek_time
        )

        if updated_playlist:
            logger.debug(f"会话 {session_id} 播放列表已更新 (seek)")

        # 确保新位置的窗口可用
        await self._ensure_windows_available(session)

        # 更新分片可用性
        await self.playlist_generator.update_segment_availability(session)

        return session

    async def get_session_playlist(self, session_id: str) -> str | None:
        """获取会话的播放列表内容"""
        session = await self.get_session(session_id)
        if not session:
            return None

        # 生成或更新播放列表
        playlist = await self.playlist_generator.generate_playlist(session)

        # 验证播放列表
        if not await self.playlist_generator.validate_playlist(playlist):
            logger.error(f"会话 {session_id} 播放列表验证失败")
            return None

        return playlist.to_m3u8()

    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        session = self.sessions.get(session_id)
        if not session:
            return False

        # 从索引中移除
        file_key = str(session.file_path)
        if file_key in self.sessions_by_file:
            self.sessions_by_file[file_key].discard(session_id)
            if not self.sessions_by_file[file_key]:
                del self.sessions_by_file[file_key]

        # 删除会话
        del self.sessions[session_id]

        logger.info(f"删除会话 {session_id}")
        return True

    async def _preload_initial_windows(self, session: PlaybackSession) -> None:
        """预加载初始窗口"""
        current_window_id = session.get_current_window_id()
        required_windows = []

        # 加载当前窗口及后续几个窗口
        for offset in range(session.preload_window_count + 1):
            window_id = current_window_id + offset
            required_windows.append(window_id)

        await self._load_windows(session, required_windows)

    async def _ensure_windows_available(self, session: PlaybackSession) -> None:
        """确保所需的窗口可用"""
        required_windows = session.get_required_windows()
        missing_windows = [
            wid
            for wid in required_windows
            if wid not in session.loaded_windows
            and wid not in session.preloading_windows
        ]

        if missing_windows:
            logger.debug(f"会话 {session.session_id} 需要加载窗口: {missing_windows}")
            await self._load_windows(session, missing_windows)

            # 更新分片可用性
            await self.playlist_generator.update_segment_availability(session)

    async def _load_windows(
        self, session: PlaybackSession, window_ids: list[int]
    ) -> None:
        """加载指定的窗口"""
        for window_id in window_ids:
            if (
                window_id in session.loaded_windows
                or window_id in session.preloading_windows
            ):
                continue

            try:
                # 标记正在预加载
                session.preloading_windows.add(window_id)

                # 计算窗口时间
                start_time = window_id * session.profile.window_duration

                # 使用JIT转码器确保窗口可用
                await self.jit_transcoder.ensure_window(
                    session.file_path, start_time, session.profile
                )

                # 获取窗口缓存信息
                cache_key = (session.asset_hash, session.profile_hash, window_id)
                await self.jit_transcoder._ensure_cache_loaded()

                if cache_key in self.jit_transcoder.cache_index:
                    window_cache = self.jit_transcoder.cache_index[cache_key]

                    # 添加到会话窗口列表
                    session.add_window(window_id, window_cache)

                    logger.debug(f"会话 {session.session_id} 窗口 {window_id} 已加载")
                else:
                    logger.warning(f"窗口 {window_id} 转码完成但未在缓存索引中找到")

            except Exception as e:
                logger.error(f"加载窗口 {window_id} 失败: {e}")
            finally:
                # 清除预加载标记
                session.preloading_windows.discard(window_id)

    async def _cleanup_loop(self) -> None:
        """背景清理任务循环"""
        logger.info("会话清理任务启动")

        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"会话清理任务出错: {e}")

    async def _cleanup_expired_sessions(self) -> None:
        """清理过期的会话"""
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)
                session.status = SessionStatus.EXPIRED
            elif session.is_idle(self.idle_timeout):
                session.status = SessionStatus.IDLE

        # 删除过期的会话
        for session_id in expired_sessions:
            await self.delete_session(session_id)

        if expired_sessions:
            logger.info(f"清理了 {len(expired_sessions)} 个过期会话")

    async def get_session_stats(self) -> SessionStats:
        """获取会话统计信息"""
        total_sessions = len(self.sessions)
        active_sessions = sum(
            1 for s in self.sessions.values() if s.status == SessionStatus.ACTIVE
        )
        idle_sessions = sum(
            1 for s in self.sessions.values() if s.status == SessionStatus.IDLE
        )
        expired_sessions = sum(
            1 for s in self.sessions.values() if s.status == SessionStatus.EXPIRED
        )

        total_windows = sum(len(s.loaded_windows) for s in self.sessions.values())

        # 计算平均会话时长
        avg_duration = 0.0
        if total_sessions > 0:
            total_duration = sum(
                time.time() - s.created_at for s in self.sessions.values()
            )
            avg_duration = total_duration / total_sessions

        return SessionStats(
            total_sessions=total_sessions,
            active_sessions=active_sessions,
            idle_sessions=idle_sessions,
            expired_sessions=expired_sessions,
            total_windows_loaded=total_windows,
            avg_session_duration=avg_duration,
        )

    async def list_sessions(self) -> list[PlaybackSession]:
        """列出所有会话"""
        return list(self.sessions.values())

    async def get_sessions_for_file(self, file_path: Path) -> list[PlaybackSession]:
        """获取指定文件的所有会话"""
        file_key = str(file_path)
        session_ids = self.sessions_by_file.get(file_key, set())
        return [self.sessions[sid] for sid in session_ids if sid in self.sessions]

    def shutdown(self) -> None:
        """关闭会话管理器"""
        logger.info("会话管理器关闭中...")

        self.running = False

        if self.cleanup_task:
            self.cleanup_task.cancel()

        # 清理所有会话
        self.sessions.clear()
        self.sessions_by_file.clear()

        logger.info("会话管理器已关闭")
