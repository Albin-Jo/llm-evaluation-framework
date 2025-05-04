from fastapi import APIRouter

from backend.app.api.v1 import datasets, prompts, evaluations, agents, reports

api_router = APIRouter()
api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(prompts.router, prefix="/prompts", tags=["Prompts"])
api_router.include_router(evaluations.router, prefix="/evaluations", tags=["Evaluations"])
api_router.include_router(agents.router, prefix="/agents", tags=["Agents"])
api_router.include_router(reports.router, prefix="/reports", tags=["Reports"])
