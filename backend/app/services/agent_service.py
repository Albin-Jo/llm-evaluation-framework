import json
import logging
import time
from typing import Dict, Any, Optional, List

import httpx
from fastapi import HTTPException, status

from backend.app.core.config import settings
from backend.app.db.models.orm import Agent
from backend.app.services.agent_clients.factory import AgentClientFactory

logger = logging.getLogger(__name__)

# Timeout for agent API calls (in seconds)
DEFAULT_TIMEOUT = 30.0


async def test_agent_service(agent: Agent, test_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test an agent by making a request using the appropriate client.

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

        # Create appropriate client based on agent type
        client = await AgentClientFactory.create_client(agent)

        # Extract query, context, and system_message if provided
        query = test_input.get("query", test_input.get("message", ""))
        context = test_input.get("context", "")
        system_message = test_input.get("system_message", None)

        # Process query through client
        response = await client.process_query(
            query=query,
            context=context,
            system_message=system_message,
            config=test_input.get("config", {})
        )

        # Calculate processing time if not already included
        if "processing_time_ms" not in response:
            processing_time_ms = int((time.time() - start_time) * 1000)
            response["processing_time_ms"] = processing_time_ms

        # Set status if not already included
        if "status" not in response:
            response["status"] = "success" if response.get("success", True) else "error"

        return response

    except Exception as e:
        logger.error(f"Unexpected error testing agent: {str(e)}", exc_info=True)

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Return error response
        return {
            "result": None,
            "processing_time_ms": processing_time_ms,
            "status": "error",
            "error": str(e)
        }


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
    import asyncio

    try:
        # Create appropriate client
        client = await AgentClientFactory.create_client(agent)

        # Create tasks for each input
        tasks = []
        for i, test_input in enumerate(test_inputs):
            # Extract query and context
            query = test_input.get("query", test_input.get("message", ""))
            context = test_input.get("context", "")
            system_message = test_input.get("system_message", None)

            # Add progress token/identifier
            config = test_input.get("config", {}).copy()
            config["progress_token"] = i

            # Create processing task
            task = asyncio.create_task(
                client.process_query(
                    query=query,
                    context=context,
                    system_message=system_message,
                    config=config
                )
            )
            tasks.append((test_input, task))

        # Process with concurrency limit
        # This will process tasks in batches of max_concurrent
        results = []
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(
                *[task for _, task in batch],
                return_exceptions=True
            )

            # Process results
            for j, result in enumerate(batch_results):
                input_data = batch[j][0]
                if isinstance(result, Exception):
                    results.append({
                        "input": input_data,
                        "error": str(result),
                        "status": "error"
                    })
                else:
                    result["input"] = input_data
                    if "status" not in result:
                        result["status"] = "success" if result.get("success", True) else "error"
                    results.append(result)

        return results
    except Exception as e:
        logger.error(f"Error in batch testing: {str(e)}", exc_info=True)
        return [{
            "error": str(e),
            "status": "error"
        }]


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
