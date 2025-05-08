import json
import logging
import time
from typing import Dict, Any, Optional, List

import httpx
from fastapi import HTTPException, status

from backend.app.core.config import settings
from backend.app.db.models.orm import Agent

logger = logging.getLogger(__name__)

# Timeout for agent API calls (in seconds)
DEFAULT_TIMEOUT = 30.0


async def test_agent_service(agent: Agent, test_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test an agent by making a request to its API endpoint.

    Args:
        agent: The agent to test
        test_input: The input data for testing

    Returns:
        The response from the agent

    Raises:
        HTTPException: If the agent API returns an error
    """
    start_time = time.time()

    try:
        logger.info(f"Testing agent {agent.id} ({agent.name}) with input: {json.dumps(test_input)[:100]}...")

        # Call the agent API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                agent.api_endpoint,
                json=test_input,
                timeout=DEFAULT_TIMEOUT,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": settings.AZURE_OPENAI_KEY,  # If your agents need authentication
                }
            )

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Handle response
            response.raise_for_status()
            result = response.json()

            # Add processing time to the result
            return {
                "result": result,
                "processing_time_ms": processing_time_ms,
                "status": "success"
            }

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from agent API: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error from Agent API: {e.response.status_code} - {e.response.text}"
        )

    except httpx.RequestError as e:
        logger.error(f"Connection error to agent API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Error connecting to Agent API: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Unexpected error testing agent: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


async def get_agent_capabilities(agent: Agent) -> Dict[str, Any]:
    """
    Get the capabilities of an agent by querying its metadata endpoint.

    Args:
        agent: The agent to query

    Returns:
        The agent's capabilities

    Raises:
        HTTPException: If the agent API returns an error
    """
    try:
        # Assuming the agent has a metadata endpoint
        metadata_endpoint = f"{agent.api_endpoint}/metadata"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                metadata_endpoint,
                timeout=DEFAULT_TIMEOUT,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": settings.AGENT_API_KEY,
                }
            )

            response.raise_for_status()
            return response.json()

    except (httpx.HTTPStatusError, httpx.RequestError, Exception) as e:
        logger.warning(f"Could not retrieve agent capabilities: {str(e)}")
        # Return basic capabilities if we can't get detailed ones
        return {
            "name": agent.name,
            "description": agent.description,
            "domain": agent.domain,
            "capabilities": [],
            "error": str(e)
        }


async def batch_test_agent(
        agent: Agent,
        test_inputs: List[Dict[str, Any]],
        max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    """
    Test an agent with multiple inputs in parallel.

    Args:
        agent: The agent to test
        test_inputs: List of input data for testing
        max_concurrent: Maximum number of concurrent requests

    Returns:
        List of responses from the agent
    """
    results = []

    # Create a rate limiter for concurrent requests
    limits = httpx.Limits(max_connections=max_concurrent)

    async with httpx.AsyncClient(limits=limits) as client:
        pending_tasks = []

        for test_input in test_inputs:
            # Create task for each input
            task = client.post(
                agent.api_endpoint,
                json=test_input,
                timeout=DEFAULT_TIMEOUT,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": settings.AGENT_API_KEY,
                }
            )
            pending_tasks.append((test_input, task))

        # Process results as they complete
        for test_input, task in pending_tasks:
            try:
                start_time = time.time()
                response = await task
                processing_time_ms = int((time.time() - start_time) * 1000)

                response.raise_for_status()
                result = response.json()

                results.append({
                    "input": test_input,
                    "result": result,
                    "processing_time_ms": processing_time_ms,
                    "status": "success"
                })

            except Exception as e:
                results.append({
                    "input": test_input,
                    "error": str(e),
                    "status": "error"
                })

    return results


async def azure_openai_agent_call(
        agent: Agent,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
) -> Dict[str, Any]:
    """
    Call an Azure OpenAI-based agent.

    Args:
        agent: The agent to call
        prompt: The prompt to send to the agent
        system_message: Optional system message to customize the agent's behavior
        temperature: Temperature parameter for response generation
        max_tokens: Maximum number of tokens to generate

    Returns:
        The response from the agent
    """
    # Prepare the payload according to Azure OpenAI API
    payload = {
        "messages": [
            {"role": "system", "content": system_message or "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model": agent.model_type or "gpt-4",  # Default to GPT-4 if not specified
    }

    # Add agent's custom configuration if available
    if agent.config:
        payload.update(agent.config)

    try:
        async with httpx.AsyncClient() as client:
            # Call the agent API
            response = await client.post(
                agent.api_endpoint,
                json=payload,
                timeout=DEFAULT_TIMEOUT,
                headers={
                    "Content-Type": "application/json",
                    "api-key": settings.AZURE_OPENAI_KEY,  # Azure OpenAI API key
                }
            )

            response.raise_for_status()
            return response.json()

    except Exception as e:
        logger.error(f"Error calling Azure OpenAI agent: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calling Azure OpenAI agent: {str(e)}"
        )
