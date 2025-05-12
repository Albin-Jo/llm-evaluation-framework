import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, AsyncIterator

from mcp import ClientSession
from mcp.client.sse import sse_client

from backend.app.db.models.orm import Agent
from backend.app.services.agent_clients.base import AgentClient

# Configure logging
logger = logging.getLogger(__name__)


def _extract_text_from_response(response: Any) -> str:
    """
    Extract text from response based on the observed format.

    Args:
        response: The raw response from the policy bot

    Returns:
        Extracted text or error message
    """
    try:
        # Check if response has expected structure
        if hasattr(response, 'content') and isinstance(response.content, list):
            # Extract text from all content items
            text_parts = []
            for item in response.content:
                if hasattr(item, 'text'):
                    text_parts.append(item.text)
            return "\n".join(text_parts)

        # Check for error
        if hasattr(response, 'isError') and response.isError:
            return f"Error from server: {getattr(response, 'error_message', 'Unknown error')}"

        # Fall back to string representation
        return str(response)
    except Exception as e:
        logger.error(f"Error extracting text from response: {e}")
        return f"Failed to parse response: {str(e)}"


class MCPAgentClient(AgentClient):
    """Client for MCP-based agents."""

    def __init__(self, agent: Agent):
        """
        Initialize an MCP client.

        Args:
            agent: The agent configuration
        """
        self.agent = agent
        self.session = None
        self.sse_url = agent.api_endpoint

        # Get bearer token from credentials
        if agent.auth_type != "bearer_token" or not agent.auth_credentials:
            raise ValueError("MCP client requires bearer token authentication")

        self.bearer_token = agent.auth_credentials.get("token")
        if not self.bearer_token:
            raise ValueError("Bearer token not found in credentials")

    async def initialize(self) -> None:
        """Initialize client - no-op as connection is handled per-request."""
        pass

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[ClientSession]:
        """
        Context manager for MCP connection.

        Usage:
            async with client.connection() as session:
                # Use session here

        Yields:
            ClientSession: An initialized MCP client session

        Raises:
            Exception: If connection fails
        """
        # Set up authentication headers
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json",
        }

        logger.info(f"Connecting to SSE endpoint with Bearer authentication: {self.sse_url}")

        try:
            # Establish SSE connection with authentication headers
            async with sse_client(url=self.sse_url, headers=headers) as sse_transport:
                read_stream, write_stream = sse_transport

                # Create MCP client session
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize the session
                    await session.initialize()
                    logger.info("Successfully connected and initialized MCP session")
                    self.session = session
                    yield session

        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise
        finally:
            self.session = None
            logger.info("Disconnected from MCP server")

    async def list_tools(self) -> List[Any]:
        """
        List available tools on the server.

        Returns:
            List[Any]: List of available tools

        Raises:
            Exception: If connection fails or tools cannot be retrieved
        """
        async with self.connection() as session:
            response = await session.list_tools()
            return response.tools

    async def process_query(self,
                            query: str,
                            context: Optional[str] = None,
                            system_message: Optional[str] = None,
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query using MCP.

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

        # Build user message
        user_message = query
        if context:
            user_message = f"{user_message}\n\nContext: {context}"

        # Get tool name from config or use default
        tool_name = self.agent.config.get("tool_name", "McpAskPolicyBot") if self.agent.config else "McpAskPolicyBot"

        # Format arguments exactly as shown in the working script
        arguments = {
            "model": [
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "prompt": system_message or "You are an AI Assistant",
            "_meta": {
                "progressToken": config.get("progress_token", 1) if config else 1
            }
        }

        # Add system message if provided
        if system_message:
            arguments["model"].insert(0, {
                "role": "system",
                "content": system_message
            })

        try:
            # Process using MCP connection
            async with self.connection() as session:
                # Call the tool
                response = await session.call_tool(
                    name=tool_name,
                    arguments=arguments
                )

                # Extract text
                text_response = _extract_text_from_response(response)

                # Check if this is an error response
                is_error = hasattr(response, 'isError') and response.isError

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000

                return {
                    "answer": text_response,
                    "processing_time_ms": int(processing_time),
                    "raw_response": str(response),
                    "success": not is_error,
                    "error": text_response if is_error else None
                }

        except Exception as e:
            logger.error(f"Error processing query with MCP: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "error": str(e),
                "success": False
            }

    async def health_check(self) -> bool:
        """
        Check if the MCP server is available.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            async with self.connection() as session:
                # Just list tools to check connection
                await session.list_tools()
                return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
