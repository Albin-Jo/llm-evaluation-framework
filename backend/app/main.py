# File: main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi

from backend.app.api.router import api_router
from backend.app.core.config import settings

# Initialize FastAPI application with more descriptive metadata
app = FastAPI(
    title=settings.APP_NAME,
    description="A framework for evaluating LLM-based micro-agents using RAGAS and DeepEval",
    version=settings.APP_VERSION,
    docs_url="/api/docs" if settings.APP_DEBUG else None,
    redoc_url="/api/redoc" if settings.APP_DEBUG else None,
    openapi_url="/api/openapi.json" if settings.APP_DEBUG else None,
)


def custom_openapi():
    # Skip the caching to always get fresh values
    openapi_schema = get_openapi(
        title=settings.APP_NAME,  # Will get current value each time
        version=settings.APP_VERSION,
        description="A framework for evaluating LLM-based micro-agents using RAGAS and DeepEval",
        routes=app.routes,
    )

    # Customize the schema if needed
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(api_router, prefix="/api")

# Mount static files
app.mount("/static", StaticFiles(directory="frontend_v1/static"), name="static")


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}