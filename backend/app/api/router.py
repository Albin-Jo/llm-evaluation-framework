from fastapi import APIRouter

from backend.app.api.v1.agents import router as agents_router
from backend.app.api.v1.auth import auth_router
from backend.app.api.v1.comparisons import comparisons_router
from backend.app.api.v1.datasets import router as datasets_router
from backend.app.api.v1.evaluations import router as evaluations_router
from backend.app.api.v1.prompts import router as prompts_router
from backend.app.api.v1.reports import router as reports_router

# Main API router
api_router = APIRouter()

# Include all routers with prefixes
api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(agents_router, prefix="/agents", tags=["Agents"])
api_router.include_router(datasets_router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(evaluations_router, prefix="/evaluations", tags=["Evaluations"])
api_router.include_router(prompts_router, prefix="/prompts", tags=["Prompts"])
api_router.include_router(reports_router, prefix="/reports", tags=["Reports"])
api_router.include_router(comparisons_router, prefix="/comparisons", tags=["Comparisons"])