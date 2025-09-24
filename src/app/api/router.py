from fastapi import APIRouter

from .v1 import jit, preload, session

router = APIRouter(prefix="/api/v1")

router.include_router(jit.router)
router.include_router(preload.router)
router.include_router(session.router)
