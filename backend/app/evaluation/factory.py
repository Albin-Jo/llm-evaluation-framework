# File: app/evaluation/factory.py
from typing import Dict, Type
from sqlalchemy.ext.asyncio import AsyncSession

from app.evaluation.methods.base import BaseEvaluationMethod
from app.evaluation.methods.ragas import RagasEvaluationMethod
from app.evaluation.methods.deepeval import DeepEvalEvaluationMethod
from app.evaluation.methods.custom import CustomEvaluationMethod
from app.evaluation.methods.manual import ManualEvaluationMethod
from app.models.orm.models import EvaluationMethod


class EvaluationMethodFactory:
    """Factory for creating evaluation method instances."""

    _registry: Dict[str, Type[BaseEvaluationMethod]] = {
        EvaluationMethod.RAGAS: RagasEvaluationMethod,
        EvaluationMethod.DEEPEVAL: DeepEvalEvaluationMethod,
        EvaluationMethod.CUSTOM: CustomEvaluationMethod,
        EvaluationMethod.MANUAL: ManualEvaluationMethod,
    }

    @classmethod
    def register(cls, method_type: str, method_class: Type[BaseEvaluationMethod]) -> None:
        """
        Register a new evaluation method class.

        Args:
            method_type: Evaluation method type
            method_class: Evaluation method class
        """
        cls._registry[method_type] = method_class

    @classmethod
    def create(cls, method_type: str, db_session: AsyncSession) -> BaseEvaluationMethod:
        """
        Create an evaluation method instance.

        Args:
            method_type: Evaluation method type
            db_session: Database session

        Returns:
            BaseEvaluationMethod: Evaluation method instance

        Raises:
            ValueError: If method_type is not registered
        """
        if method_type not in cls._registry:
            raise ValueError(f"Unsupported evaluation method: {method_type}")

        return cls._registry[method_type](db_session)