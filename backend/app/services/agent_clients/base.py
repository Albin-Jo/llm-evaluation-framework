from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class AgentClient(ABC):
    """Abstract base class for agent clients."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize client connection if needed."""
        pass

    @abstractmethod
    async def process_query(self,
                            query: str,
                            context: Optional[str] = None,
                            system_message: Optional[str] = None,
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query and return the response.

        Args:
            query: The query to process
            context: Optional context for the query
            system_message: Optional system message
            config: Optional additional configuration

        Returns:
            Dict containing the response and metadata
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the agent is healthy and available.

        Returns:
            bool: True if healthy, False otherwise
        """
        pass