# File: app/api/router.py
from fastapi import APIRouter, Depends

from backend.app.api.v1 import (
    auth, datasets, evaluations, agents, prompts, comparisons
)
from backend.app.core.config import settings

api_router = APIRouter()

# Include specific endpoint routers with explicit tags
# api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
# api_router.include_router(evaluations.router, prefix="/evaluations", tags=["Evaluations"])
# api_router.include_router(agents.router, prefix="/microagents", tags=["MicroAgents"],)
api_router.include_router(prompts.router, prefix="/prompts", tags=["Prompts"])
# api_router.include_router(comparisons.router, prefix="/comparisons", tags=["Comparisons"])
