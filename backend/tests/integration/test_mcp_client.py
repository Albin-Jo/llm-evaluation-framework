import asyncio
import os
import logging
from uuid import uuid4

from backend.app.db.models.orm import Agent, AuthType, IntegrationType
from backend.app.services.agent_clients.mcp_client import MCPAgentClient

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp-test")


async def test_mcp_client():
    """Test the MCP client implementation."""
    # Get configuration from environment variables
    sse_url = os.environ.get("MCP_SSE_URL")
    bearer_token = os.environ.get("MCP_BEARER_TOKEN")

    if not sse_url or not bearer_token:
        logger.error("Missing required environment variables: MCP_SSE_URL and MCP_BEARER_TOKEN")
        return

    # Create a mock agent
    mock_agent = Agent(
        id=uuid4(),
        name="Test MCP Agent",
        description="Agent for testing MCP client",
        api_endpoint=sse_url,
        domain="test",
        integration_type=IntegrationType.MCP,
        auth_type=AuthType.BEARER_TOKEN,
        auth_credentials={"token": bearer_token},
        config={"tool_name": "McpAskPolicyBot"},
        is_active=True
    )

    try:
        # Create MCP client
        client = MCPAgentClient(mock_agent)

        # Test connection and tool listing
        logger.info("Testing connection and listing tools...")
        tools = await client.list_tools()
        logger.info(f"Found {len(tools)} tools:")
        for i, tool in enumerate(tools):
            logger.info(f"{i + 1}. {tool.name}: {getattr(tool, 'description', 'No description')}")

        # Test query processing
        logger.info("\nTesting query processing...")
        response = await client.process_query(
            query="What can you do?",
            system_message="You are a helpful assistant for testing purposes."
        )

        logger.info(f"Query response: {response.get('answer')}")
        logger.info(f"Processing time: {response.get('processing_time_ms')}ms")

        # Test health check
        logger.info("\nTesting health check...")
        health = await client.health_check()
        logger.info(f"Health check result: {'Healthy' if health else 'Unhealthy'}")

        # Test with context
        logger.info("\nTesting query with context...")
        response_with_context = await client.process_query(
            query="What is mentioned in the context?",
            context="The MCP client is working properly and can process this context.",
            system_message="You are a helpful assistant for testing purposes."
        )

        logger.info(f"Query with context response: {response_with_context.get('answer')}")

        logger.info("\nAll tests completed successfully!")

    except Exception as e:
        logger.error(f"Error testing MCP client: {e}", exc_info=True)


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_mcp_client())