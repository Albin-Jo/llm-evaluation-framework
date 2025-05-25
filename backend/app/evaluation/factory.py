import logging
from typing import Dict, Type, List

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import EvaluationMethod
from backend.app.evaluation.methods.base import BaseEvaluationMethod

# Configure logging
logger = logging.getLogger(__name__)

# Method registry
_method_registry: Dict[str, Type[BaseEvaluationMethod]] = {}


def register_method(method_name: str, method_class: Type[BaseEvaluationMethod]) -> None:
    """
    Register an evaluation method.

    Args:
        method_name: The name of the method
        method_class: The method class to register
    """
    _method_registry[method_name.lower()] = method_class
    logger.debug(f"Registered evaluation method: {method_name}")


class EvaluationMethodFactory:
    """Factory for creating evaluation method instances."""

    @staticmethod
    def create(method: EvaluationMethod, db_session: AsyncSession) -> BaseEvaluationMethod:
        """
        Create an evaluation method instance.

        Args:
            method: Evaluation method enum
            db_session: Database session

        Returns:
            BaseEvaluationMethod: Evaluation method instance

        Raises:
            ValueError: If method is not supported
        """
        method_name = method.value.lower()

        if method_name not in _method_registry:
            available_methods = list(_method_registry.keys())
            raise ValueError(
                f"Unsupported evaluation method: {method_name}. "
                f"Available methods: {available_methods}"
            )

        method_class = _method_registry[method_name]
        return method_class(db_session)

    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available evaluation methods."""
        return list(_method_registry.keys())

    @staticmethod
    def is_method_available(method_name: str) -> bool:
        """Check if an evaluation method is available."""
        return method_name.lower() in _method_registry


# Import and register methods
# This avoids circular imports
def register_evaluation_methods():
    """Register all evaluation methods."""

    # Register RAGAS method
    try:
        from backend.app.evaluation.methods.ragas import RagasEvaluationMethod
        register_method("ragas", RagasEvaluationMethod)
        logger.info("Successfully registered RAGAS evaluation method")
    except ImportError as e:
        logger.warning(f"Could not register RAGAS evaluation method: {e}")

    # Register DeepEval method
    try:
        from backend.app.evaluation.methods.deepeval_method import DeepEvalMethod
        register_method("deepeval", DeepEvalMethod)
        logger.info("Successfully registered DeepEval evaluation method")
    except ImportError as e:
        logger.warning(f"Could not register DeepEval evaluation method: {e}")
        logger.warning("Install DeepEval with: pip install deepeval")

    # Register Custom method
    try:
        from backend.app.evaluation.methods.custom import CustomEvaluationMethod
        register_method("custom", CustomEvaluationMethod)
        logger.info("Successfully registered Custom evaluation method")
    except ImportError as e:
        logger.warning(f"Could not register Custom evaluation method: {e}")

    # Register Manual method (if available)
    try:
        from backend.app.evaluation.methods.manual import ManualEvaluationMethod
        register_method("manual", ManualEvaluationMethod)
        logger.info("Successfully registered Manual evaluation method")
    except ImportError as e:
        logger.debug(f"Manual evaluation method not available: {e}")

    logger.info(f"Registered {len(_method_registry)} evaluation methods: {list(_method_registry.keys())}")


# Register all methods at module load time
register_evaluation_methods()
