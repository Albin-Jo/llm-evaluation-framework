# File: backend/app/services/microagent_service.py
import logging
import time
from typing import Dict, Any, Optional, Union, List
import httpx
from pydantic import BaseModel, Field

from backend.app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class MicroAgentResponse(BaseModel):
    """Response model for micro-agent API."""
    answer: str
    processing_time_ms: Optional[int] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class MicroAgentService:
    """Service for interacting with micro-agents."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the micro-agent service.

        Args:
            api_key: API key for authentication (defaults to settings)
        """
        # self.api_key = api_key or settings.OPENAI_API_KEY.get_secret_value()
        self.api_key = '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'

    async def query_openai(
            self,
            prompt: str,
            query: str,
            context: Optional[str] = None,
            model: str = "gpt-4",
            temperature: float = 0.0,
            max_tokens: int = 1000,
            use_azure: bool = False
    ) -> MicroAgentResponse:
        """
        Query OpenAI API directly (for testing when no micro-agent is available).

        Args:
            prompt: Formatted prompt
            query: User query
            context: Optional context
            model: OpenAI model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_azure: Whether to use Azure OpenAI

        Returns:
            MicroAgentResponse: API response
        """
        # Build message content
        messages = [
            {"role": "system", "content": prompt}
        ]

        if context:
            messages.append({
                "role": "user",
                "content": f"Query: {query}\n\nContext: {context}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Query: {query}"
            })

        try:
            start_time = time.time()

            if use_azure:
                response = await self._call_azure_openai(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                response = await self._call_openai(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            processing_time = int((time.time() - start_time) * 1000)

            # Extract the answer
            answer = response.get("choices", [{}])[0].get("message", {}).get("content", "")

            return MicroAgentResponse(
                answer=answer,
                processing_time_ms=processing_time,
                confidence=0.9,  # Placeholder confidence
                metadata={
                    "model": model,
                    "tokens": response.get("usage", {})
                }
            )

        except Exception as e:
            logger.exception(f"Error querying OpenAI: {str(e)}")
            raise

    async def _call_openai(
            self,
            model: str,
            messages: List[Dict[str, str]],
            temperature: float,
            max_tokens: int
    ) -> Dict[str, Any]:
        """
        Call OpenAI API.

        Args:
            model: Model name
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dict[str, Any]: API response
        """
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )

            response.raise_for_status()
            return response.json()

    async def _call_azure_openai(
            self,
            model: str,
            messages: List[Dict[str, str]],
            temperature: float,
            max_tokens: int
    ) -> Dict[str, Any]:
        """
        Call Azure OpenAI API.

        Args:
            model: Model name (deployment name in Azure)
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dict[str, Any]: API response
        """
        # Extract deployment name from model
        deployment = model

        # Use environment settings for Azure
        azure_endpoint = settings.AZURE_OPENAI_ENDPOINT
        api_version = settings.AZURE_OPENAI_API_VERSION

        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint not configured")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{azure_endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}",
                headers={
                    "api-key": self.api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )

            response.raise_for_status()
            return response.json()


class TestMicroAgentEndpoint:
    """Utility class for testing evaluation with a simulated micro-agent endpoint."""

    @staticmethod
    async def create_test_endpoint():
        """
        Create a FastAPI app with a test endpoint for micro-agent simulation.

        This is for testing purposes only, not for production use.

        Returns:
            FastAPI app
        """
        from fastapi import FastAPI, Depends, HTTPException
        from pydantic import BaseModel

        class MicroAgentRequest(BaseModel):
            prompt: str
            query: str
            context: Optional[str] = None

        app = FastAPI(title="Test Micro-agent API")
        service = MicroAgentService()

        @app.post("/api/ask")
        async def ask(request: MicroAgentRequest):
            try:
                response = await service.query_openai(
                    prompt=request.prompt,
                    query=request.query,
                    context=request.context
                )
                return response.model_dump()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return app

# Example usage:
# async def test_microagent():
#     service = MicroAgentService()
#     response = await service.query_openai(
#         prompt="You are a helpful assistant. Answer the question based on the provided context.",
#         query="What is the capital of France?",
#         context="France is a country in Western Europe. Its capital is Paris."
#     )
#     print(response.answer)