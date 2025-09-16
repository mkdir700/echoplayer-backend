"""
预取策略服务
基于播放模式的智能预取
"""

import asyncio
import logging
import time

from app.models.window import TranscodeProfile
from app.utils.hash import (
    calculate_asset_hash,
    calculate_profile_hash,
    calculate_window_id,
)

from .jit_transcoder import JITTranscoder

logger = logging.getLogger(__name__)


class PreloadStrategy:
    """预取策略管理器"""

    _instance = None
    _initilized = False

    def __new__(cls, *args, **kwargs):  # noqa: ARG004
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, jit_transcoder: JITTranscoder, max_preload_tasks: int = 2):
        if self._initilized:
            return
        self.jit_transcoder = jit_transcoder
        self.max_preload_tasks = max_preload_tasks

        # 预取任务管理
        self.preload_tasks: dict[str, asyncio.Task] = {}
        self.preload_semaphore = asyncio.Semaphore(max_preload_tasks)

        # 播放历史记录
        self.playback_history: dict[
            str, list[tuple[float, float]]
        ] = {}  # file_path -> [(time, timestamp), ...]
        self.preload_patterns: dict[str, list[int]] = {}  # file_path -> [window_ids]

        logger.info(f"预取策略初始化，最大并发预取任务数: {max_preload_tasks}")
        self._initilized = True

    async def record_playback(
        self, file_path: str, time_seconds: float, profile: TranscodeProfile
    ) -> None:
        """
        记录播放历史

        Args:
            file_path: 文件路径
            time_seconds: 播放时间点
            profile: 转码配置
        """
        current_time = time.time()

        if file_path not in self.playback_history:
            self.playback_history[file_path] = []

        # 记录播放点
        self.playback_history[file_path].append((time_seconds, current_time))

        # 保留最近100个播放记录
        if len(self.playback_history[file_path]) > 100:
            self.playback_history[file_path] = self.playback_history[file_path][-100:]

        # 分析播放模式并启动预取
        await self._analyze_and_preload(file_path, time_seconds, profile)

    async def _analyze_and_preload(
        self, file_path: str, current_time: float, profile: TranscodeProfile
    ) -> None:
        """
        分析播放模式并启动预取

        Args:
            file_path: 文件路径
            current_time: 当前播放时间
            profile: 转码配置
        """
        history = self.playback_history.get(file_path, [])
        if len(history) < 2:
            # 首次播放，预取下一个窗口
            await self._preload_next_window(file_path, current_time, profile)
            return

        # 分析播放模式
        playback_pattern = self._detect_playback_pattern(history)

        if playback_pattern == "sequential":
            # 连续播放模式，预取后续窗口
            await self._preload_sequential(file_path, current_time, profile)
        elif playback_pattern == "seeking":
            # 跳跃播放模式，根据历史预测可能的跳跃点
            await self._preload_predicted_seeks(
                file_path, current_time, profile, history
            )
        else:
            # 默认预取下一个窗口
            await self._preload_next_window(file_path, current_time, profile)

    def _detect_playback_pattern(self, history: list[tuple[float, float]]) -> str:
        """
        检测播放模式

        Args:
            history: 播放历史记录

        Returns:
            str: 播放模式 ("sequential", "seeking", "random")
        """
        if len(history) < 5:
            return "unknown"

        # 获取最近的播放记录
        recent_history = history[-5:]
        time_diffs = []
        timestamp_diffs = []

        for i in range(1, len(recent_history)):
            prev_time, prev_timestamp = recent_history[i - 1]
            curr_time, curr_timestamp = recent_history[i]

            time_diff = curr_time - prev_time
            timestamp_diff = curr_timestamp - prev_timestamp

            time_diffs.append(time_diff)
            timestamp_diffs.append(timestamp_diff)

        # 分析时间间隔
        avg_time_diff = sum(time_diffs) / len(time_diffs)
        avg_timestamp_diff = sum(timestamp_diffs) / len(timestamp_diffs)

        # 连续播放判断：时间差与时间戳差近似
        if (
            all(
                abs(td - tsd) < 5.0
                for td, tsd in zip(time_diffs, timestamp_diffs, strict=False)
            )
            and avg_time_diff > 0
            and avg_time_diff < 60
        ):  # 连续播放，且不超过1分钟间隔
            return "sequential"

        # 跳跃播放判断：时间差变化较大
        if any(abs(td) > 30 for td in time_diffs):  # 有大于30秒的跳跃
            return "seeking"

        return "random"

    async def _preload_next_window(
        self, file_path: str, current_time: float, profile: TranscodeProfile
    ) -> None:
        """
        预取下一个窗口

        Args:
            file_path: 文件路径
            current_time: 当前时间点
            profile: 转码配置
        """
        current_window_id = calculate_window_id(current_time, profile.window_duration)
        next_window_id = current_window_id + 1

        await self._queue_preload_task(
            file_path, next_window_id * profile.window_duration, profile, priority=8
        )

    async def _preload_sequential(
        self, file_path: str, current_time: float, profile: TranscodeProfile
    ) -> None:
        """
        连续播放预取策略

        Args:
            file_path: 文件路径
            current_time: 当前时间点
            profile: 转码配置
        """
        current_window_id = calculate_window_id(current_time, profile.window_duration)

        # 预取后续2-3个窗口
        preload_count = min(3, self.max_preload_tasks)
        for i in range(1, preload_count + 1):
            next_window_time = (current_window_id + i) * profile.window_duration
            priority = max(1, 9 - i)  # 优先级递减
            await self._queue_preload_task(
                file_path, next_window_time, profile, priority
            )

    async def _preload_predicted_seeks(
        self,
        file_path: str,
        current_time: float,
        profile: TranscodeProfile,
        history: list[tuple[float, float]],
    ) -> None:
        """
        基于跳跃模式的预测预取

        Args:
            file_path: 文件路径
            current_time: 当前时间点
            profile: 转码配置
            history: 播放历史
        """
        # 分析跳跃模式
        seek_targets = self._predict_seek_targets(history, current_time)

        # 预取预测的跳跃点
        for target_time, confidence in seek_targets[:2]:  # 最多预取2个预测点
            priority = max(1, int(confidence * 10))
            await self._queue_preload_task(file_path, target_time, profile, priority)

    def _predict_seek_targets(
        self, history: list[tuple[float, float]], current_time: float
    ) -> list[tuple[float, float]]:
        """
        预测可能的跳跃目标

        Args:
            history: 播放历史
            current_time: 当前时间

        Returns:
            List[Tuple[float, float]]: [(预测时间, 置信度), ...]
        """
        if len(history) < 3:
            return []

        # 简单的模式识别：查找重复访问的时间点
        time_points = [h[0] for h in history]
        time_frequency = {}

        for t in time_points:
            # 按5秒间隔聚合
            bucket = int(t // 5) * 5
            time_frequency[bucket] = time_frequency.get(bucket, 0) + 1

        # 找出访问频率高的时间点
        frequent_points = []
        for time_point, freq in time_frequency.items():
            if freq > 1 and abs(time_point - current_time) > 30:  # 不在当前附近
                confidence = min(1.0, freq / len(history))
                frequent_points.append((float(time_point), confidence))

        # 按置信度排序
        frequent_points.sort(key=lambda x: x[1], reverse=True)
        return frequent_points

    async def _queue_preload_task(
        self,
        file_path: str,
        time_seconds: float,
        profile: TranscodeProfile,
        priority: int = 5,
    ) -> None:
        """
        将预取任务加入队列

        Args:
            file_path: 文件路径
            time_seconds: 目标时间点
            profile: 转码配置
            priority: 优先级（1-10，数字越大优先级越高）
        """
        window_id = calculate_window_id(time_seconds, profile.window_duration)
        task_key = f"{file_path}:{window_id}"

        # 避免重复预取
        if task_key in self.preload_tasks:
            existing_task = self.preload_tasks[task_key]
            if not existing_task.done():
                logger.debug(f"预取任务已存在: {task_key}")
                return

        # 检查是否已缓存
        try:
            asset_hash = calculate_asset_hash(file_path)
            profile_hash = calculate_profile_hash(profile.__dict__, profile.version)
            cache_key = (asset_hash, profile_hash, window_id)

            await self.jit_transcoder._ensure_cache_loaded()
            if await self.jit_transcoder._check_cache_hit(cache_key):
                logger.debug(f"窗口 {window_id} 已缓存，跳过预取")
                return

        except Exception as e:
            logger.warning(f"检查缓存失败: {e}")

        # 创建预取任务
        task = asyncio.create_task(
            self._execute_preload(file_path, time_seconds, profile, priority, task_key)
        )
        self.preload_tasks[task_key] = task

        logger.info(f"启动预取任务: {task_key}, 优先级: {priority}")

    async def _execute_preload(
        self,
        file_path: str,
        time_seconds: float,
        profile: TranscodeProfile,
        priority: int,
        task_key: str,
    ) -> None:
        """
        执行预取任务

        Args:
            file_path: 文件路径
            time_seconds: 目标时间点
            profile: 转码配置
            priority: 优先级
            task_key: 任务键
        """
        async with self.preload_semaphore:
            try:
                start_time = time.time()
                await self.jit_transcoder.ensure_window(
                    file_path, time_seconds, profile
                )
                elapsed_time = time.time() - start_time

                logger.info(f"预取完成: {task_key}, 耗时: {elapsed_time:.1f}s")

            except asyncio.CancelledError:
                logger.info(f"预取任务被取消: {task_key}")
                raise
            except Exception as e:
                logger.error(f"预取任务失败 {task_key}: {e}")
            finally:
                # 清理任务记录
                if task_key in self.preload_tasks:
                    del self.preload_tasks[task_key]

    async def preload_time_ranges(
        self,
        file_path: str,
        time_ranges: list[tuple[float, float]],
        profile: TranscodeProfile,
        priority: int = 5,
    ) -> tuple[int, int]:
        """
        预取指定时间范围

        Args:
            file_path: 文件路径
            time_ranges: 时间范围列表
            profile: 转码配置
            priority: 优先级

        Returns:
            Tuple[int, int]: (排队的窗口数, 已缓存的窗口数)
        """
        queued_windows = 0
        cached_windows = 0

        for start_time, end_time in time_ranges:
            current_time = start_time

            while current_time < end_time:
                window_id = calculate_window_id(current_time, profile.window_duration)
                task_key = f"{file_path}:{window_id}"

                # 检查是否已缓存
                try:
                    asset_hash = calculate_asset_hash(file_path)
                    profile_hash = calculate_profile_hash(
                        profile.__dict__, profile.version
                    )
                    cache_key = (asset_hash, profile_hash, window_id)

                    await self.jit_transcoder._ensure_cache_loaded()
                    if await self.jit_transcoder._check_cache_hit(cache_key):
                        cached_windows += 1
                    else:
                        await self._queue_preload_task(
                            file_path, current_time, profile, priority
                        )
                        queued_windows += 1

                except Exception as e:
                    logger.warning(f"预取时间范围检查失败 {current_time}: {e}")

                # 移动到下一个窗口
                current_time = (window_id + 1) * profile.window_duration

        return queued_windows, cached_windows

    def cancel_preload_tasks(self, file_path: str | None = None) -> int:
        """
        取消预取任务

        Args:
            file_path: 可选的文件路径过滤器

        Returns:
            int: 取消的任务数
        """
        cancelled_count = 0

        tasks_to_cancel = []
        for task_key, task in self.preload_tasks.items():
            if file_path is None or task_key.startswith(f"{file_path}:"):
                tasks_to_cancel.append((task_key, task))

        for task_key, task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                cancelled_count += 1
                logger.debug(f"取消预取任务: {task_key}")

        return cancelled_count

    def get_preload_status(self) -> dict:
        """获取预取状态"""
        active_tasks = sum(1 for task in self.preload_tasks.values() if not task.done())
        completed_tasks = sum(1 for task in self.preload_tasks.values() if task.done())

        return {
            "active_tasks": active_tasks,
            "completed_tasks": completed_tasks,
            "total_tasks": len(self.preload_tasks),
            "max_concurrent": self.max_preload_tasks,
            "playback_patterns": len(self.playback_history),
        }

    def shutdown(self) -> None:
        """关闭预取服务"""
        logger.info("预取服务正在关闭...")

        # 取消所有任务
        cancelled = self.cancel_preload_tasks()
        if cancelled > 0:
            logger.info(f"已取消 {cancelled} 个预取任务")

        # 清理数据
        self.preload_tasks.clear()
        self.playback_history.clear()
        self.preload_patterns.clear()

        logger.info("预取服务已关闭")
