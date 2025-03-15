# File: app/evaluation/engines/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseEvaluationEngine(ABC):
    """Base class for evaluation engines."""

    @abstractmethod
    async def evaluate(
            self,
            queries: List[str],
            contexts: List[str],
            responses: List[str],
            ground_truths: Optional[List[str]] = None,
            config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[float]]:
        """
        Evaluate responses against queries, contexts, and optional ground truths.

        Args:
            queries: List of queries
            contexts: List of contexts
            responses: List of responses
            ground_truths: Optional list of ground truths
            config: Optional configuration

        Returns:
            Dict[str, List[float]]: Dictionary mapping metric names to lists of scores
        """
        pass