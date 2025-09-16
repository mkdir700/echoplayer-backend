from fastapi import Depends

from app.services.cache_manager import CacheManager
from app.services.jit_transcoder import JITTranscoder
from app.services.preload_strategy import PreloadStrategy


async def get_cache_manager():
    return CacheManager()


async def get_jit_transcoder():
    return JITTranscoder()


async def get_preload_strategy(
    jit_transcoder: JITTranscoder = Depends(get_jit_transcoder),
):
    return PreloadStrategy(jit_transcoder)
