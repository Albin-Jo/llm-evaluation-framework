from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from backend.app.api.middleware.error_handler import error_handler_middleware
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


# Custom OpenAPI schema
def custom_openapi():
    # if app.openapi_schema:
    #     return app.openapi_schema

    openapi_schema = get_openapi(
        title=settings.APP_NAME,
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

app.middleware("http")(error_handler_middleware)

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


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}
