# File: backend/app/evaluation/factory.py
import logging
from typing import Dict, Type

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm.models import EvaluationMethod
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
            raise ValueError(f"Unsupported evaluation method: {method_name}")

        method_class = _method_registry[method_name]
        return method_class(db_session)


# Import and register methods
# This avoids circular imports
def register_evaluation_methods():
    """Register all evaluation methods."""
    try:
        from backend.app.evaluation.methods.ragas import RagasEvaluationMethod
        register_method("ragas", RagasEvaluationMethod)
    except ImportError as e:
        logger.warning(f"Could not register RAGAS evaluation method: {e}")

    # Register other methods as needed
    # try:
    #     from backend.app.evaluation.methods.deepeval import DeepEvalMethod
    #     register_method("deepeval", DeepEvalMethod)
    # except ImportError as e:
    #     logger.warning(f"Could not register DeepEval evaluation method: {e}")

    # try:
    #     from backend.app.evaluation.methods.manual import ManualEvaluationMethod
    #     register_method("manual", ManualEvaluationMethod)
    # except ImportError as e:
    #     logger.warning(f"Could not register Manual evaluation method: {e}")

    try:
        from backend.app.evaluation.methods.custom import CustomEvaluationMethod
        register_method("custom", CustomEvaluationMethod)
    except ImportError as e:
        logger.warning(f"Could not register Custom evaluation method: {e}")

    logger.info(f"Registered {len(_method_registry)} evaluation methods")


# Register all methods at module load time
register_evaluation_methods()