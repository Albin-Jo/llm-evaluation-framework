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
    """Register built-in metrics with updated descriptions."""
    from backend.app.evaluation.metrics.ragas_metrics import (
        calculate_faithfulness, calculate_response_relevancy,
        calculate_context_precision, calculate_context_recall,
        calculate_context_entity_recall, calculate_noise_sensitivity
    )

    MetricsRegistry.register(
        "faithfulness",
        calculate_faithfulness,
        "Measures how well the answer sticks to the information in the context without hallucinating.",
        1.0
    )

    MetricsRegistry.register(
        "response_relevancy",
        calculate_response_relevancy,
        "Measures how relevant the answer is to the query asked.",
        1.0
    )

    MetricsRegistry.register(
        "context_precision",
        calculate_context_precision,
        "Measures how precisely the retrieved context matches what's needed to answer the query.",
        1.0
    )

    MetricsRegistry.register(
        "context_recall",
        calculate_context_recall,
        "Measures how well the retrieved context covers all the information needed to answer the query.",
        1.0
    )

    MetricsRegistry.register(
        "context_entity_recall",
        calculate_context_entity_recall,
        "Measures how well the retrieved context captures the entities mentioned in the reference answer.",
        1.0
    )

    MetricsRegistry.register(
        "noise_sensitivity",
        calculate_noise_sensitivity,
        "Measures the model's tendency to be misled by irrelevant information in the context (lower is better).",
        1.0
    )


register_builtin_metrics()