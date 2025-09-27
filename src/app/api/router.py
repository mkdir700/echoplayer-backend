from fastapi import APIRouter

from .v1 import audio, jit, session

router = APIRouter(prefix="/api/v1")

router.include_router(jit.router)
router.include_router(session.router)
router.include_router(audio.router)
