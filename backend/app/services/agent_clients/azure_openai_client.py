import asyncio
import logging
import time
from typing import Dict, Any, Optional, List

import httpx
from backend.app.services.agent_clients.base import AgentClient
from backend.app.db.models.orm import Agent
from backend.app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class AzureOpenAIClient(AgentClient):
    """Client for Azure OpenAI-based agents."""

    def __init__(self, agent: Agent):
        """
        Initialize an Azure OpenAI client.

        Args:
            agent: The agent configuration
        """
        self.agent = agent
        self.api_key = settings.AZURE_OPENAI_KEY
        self.endpoint = agent.api_endpoint
        self.timeout = agent.config.get("timeout", 60.0) if agent.config else 60.0

        # Use custom API key if provided in credentials
        if agent.auth_credentials and agent.auth_credentials.get("api_key"):
            self.api_key = agent.auth_credentials.get("api_key")

    async def initialize(self) -> None:
        """Initialize client - no-op as Azure OpenAI uses a stateless REST API."""
        pass

    async def process_query(self,
                            query: str,
                            context: Optional[str] = None,
                            system_message: Optional[str] = None,
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query using Azure OpenAI.

        Args:
            query: The query to process
            context: Optional context for the query
            system_message: Optional system message
            config: Optional additional configuration

        Returns:
            Dict containing the response and metadata

        Raises:
            Exception: If query processing fails
        """
        start_time = time.time()

        # Prepare user message
        user_message = query
        if context:
            user_message = f"Question: {query}\n\nContext: {context}"

        # Prepare system message
        system_content = system_message or "You are a helpful AI assistant."

        # Prepare payload
        payload = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": config.get("max_tokens", 1000) if config else 1000,
            "temperature": config.get("temperature", 0.0) if config else 0.0
        }

        # Apply any model-specific configuration from agent config
        if self.agent.config:
            for key, value in self.agent.config.items():
                if key not in payload and key not in ["timeout"]:
                    payload[key] = value

        # Get retry config
        retry_config = self.agent.retry_config or {
            "max_retries": 3,
            "backoff_factor": 1.5,
            "status_codes": [429, 500, 502, 503, 504]
        }

        # Execute with retries
        return await self._execute_with_retry(payload, retry_config)

    async def _execute_with_retry(self, payload: Dict[str, Any], retry_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Azure OpenAI API call with retry logic.

        Args:
            payload: The request payload
            retry_config: Retry configuration

        Returns:
            Dict[str, Any]: Response dictionary
        """
        start_time = time.time()
        max_retries = retry_config.get("max_retries", 3)
        backoff_factor = retry_config.get("backoff_factor", 1.5)
        retry_status_codes = retry_config.get("status_codes", [429, 500, 502, 503, 504])

        retries = 0
        last_exception = None

        while retries <= max_retries:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.endpoint,
                        json=payload,
                        headers={
                            "api-key": self.api_key,
                            "Content-Type": "application/json"
                        }
                    )

                    # Handle non-success responses
                    if response.status_code >= 400:
                        error_text = response.text
                        logger.warning(f"API call failed with status {response.status_code}: {error_text}")

                        # Check if we should retry based on status code
                        if response.status_code in retry_status_codes and retries < max_retries:
                            retries += 1
                            wait_time = backoff_factor ** retries
                            logger.info(f"Retrying in {wait_time:.2f} seconds (attempt {retries}/{max_retries})")
                            await asyncio.sleep(wait_time)
                            continue

                        # If we shouldn't retry, return error response
                        return {
                            "answer": f"Error: API returned status code {response.status_code}",
                            "processing_time_ms": int((time.time() - start_time) * 1000),
                            "error": error_text,
                            "status_code": response.status_code,
                            "success": False
                        }

                    # Process successful response
                    response_data = response.json()
                    processing_time = int((time.time() - start_time) * 1000)

                    # Extract answer from response
                    answer = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

                    return {
                        "answer": answer,
                        "processing_time_ms": processing_time,
                        "raw_response": response_data,
                        "success": True
                    }

            except httpx.RequestError as e:
                last_exception = e
                logger.error(f"Request error: {e}")
                retries += 1
                if retries <= max_retries:
                    wait_time = backoff_factor ** retries
                    logger.info(f"Retrying in {wait_time:.2f} seconds (attempt {retries}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    # Max retries exceeded
                    return {
                        "answer": f"Error: Could not connect to API after {max_retries} attempts",
                        "processing_time_ms": int((time.time() - start_time) * 1000),
                        "error": str(e),
                        "success": False
                    }

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return {
                    "answer": f"Error: {str(e)}",
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                    "error": str(e),
                    "success": False
                }

        # Should not reach here, but just in case
        return {
            "answer": f"Error: Failed after {max_retries} attempts",
            "processing_time_ms": int((time.time() - start_time) * 1000),
            "error": str(last_exception) if last_exception else "Unknown error",
            "success": False
        }

    async def health_check(self) -> bool:
        """
        Check if the Azure OpenAI API is available.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Simple health check payload
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"}
                ],
                "max_tokens": 5,
                "temperature": 0.0
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.endpoint,
                    json=payload,
                    headers={
                        "api-key": self.api_key,
                        "Content-Type": "application/json"
                    }
                )
                return response.status_code < 400
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False