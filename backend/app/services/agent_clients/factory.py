import logging

from backend.app.db.models.orm import Agent, IntegrationType
from backend.app.services.agent_clients.base import AgentClient
from backend.app.services.agent_clients.azure_openai_client import AzureOpenAIClient
from backend.app.services.agent_clients.mcp_client import MCPAgentClient

# Configure logging
logger = logging.getLogger(__name__)


class AgentClientFactory:
    """Factory for creating agent clients."""

    @staticmethod
    async def create_client(agent: Agent) -> AgentClient:
        """
        Create an appropriate client for the given agent.

        Args:
            agent: The agent to create a client for

        Returns:
            An initialized agent client

        Raises:
            ValueError: If the agent type is not supported
        """
        client = None

        if agent.integration_type == IntegrationType.AZURE_OPENAI:
            logger.info(f"Creating Azure OpenAI client for agent {agent.id}")
            client = AzureOpenAIClient(agent)
        elif agent.integration_type == IntegrationType.MCP:
            logger.info(f"Creating MCP client for agent {agent.id}")
            client = MCPAgentClient(agent)
        else:
            raise ValueError(f"Unsupported integration type: {agent.integration_type}")

        # Initialize the client
        await client.initialize()
        return client