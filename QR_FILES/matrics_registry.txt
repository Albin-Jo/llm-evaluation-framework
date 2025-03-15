# File: app/evaluation/metrics/registry.py
from typing import Any, Callable, Dict, Optional


class MetricsRegistry:
    """Registry for evaluation metrics."""

    _metrics: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, func: Callable, description: str = "", default_weight: float = 1.0):
        """Register a metric."""
        cls._metrics[name] = {
            "func": func,
            "description": description,
            "default_weight": default_weight
        }

    @classmethod
    def get(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get a metric by name."""
        return cls._metrics.get(name)

    @classmethod
    def list_metrics(cls) -> Dict[str, Dict[str, Any]]:
        """List all registered metrics."""
        return {
            name: {
                "description": info["description"],
                "default_weight": info["default_weight"]
            }
            for name, info in cls._metrics.items()
        }


# Register built-in metrics
def register_builtin_metrics():
    """Register built-in metrics."""
    from app.evaluation.metrics.ragas_metrics import (
        calculate_faithfulness, calculate_answer_relevancy,
        calculate_context_relevancy, calculate_correctness
    )

    MetricsRegistry.register(
        "faithfulness",
        calculate_faithfulness,
        "Measures how well the answer sticks to the information in the context without hallucinating.",
        1.0
    )

    MetricsRegistry.register(
        "answer_relevancy",
        calculate_answer_relevancy,
        "Measures how relevant the answer is to the query asked.",
        1.0
    )

    MetricsRegistry.register(
        "context_relevancy",
        calculate_context_relevancy,
        "Measures how relevant the context is to the query asked.",
        1.0
    )

    MetricsRegistry.register(
        "correctness",
        calculate_correctness,
        "Measures how well the answer matches the ground truth.",
        1.0
    )


register_builtin_metrics()