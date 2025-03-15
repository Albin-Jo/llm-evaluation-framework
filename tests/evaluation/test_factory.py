# File: tests/evaluation/test_factory.py
import pytest
from unittest.mock import MagicMock

from app.evaluation.factory import EvaluationMethodFactory
from app.models.orm.models import EvaluationMethod
from app.evaluation.methods.base import BaseEvaluationMethod
from app.evaluation.methods.ragas import RagasEvaluationMethod
from app.evaluation.methods.deepeval import DeepEvalEvaluationMethod
from app.evaluation.methods.custom import CustomEvaluationMethod
from app.evaluation.methods.manual import ManualEvaluationMethod

# pytestmark = pytest.mark.skipif(
#     True,  # Change to False when ready to enable these tests
#     reason="Database tests are currently disabled"
# )
def test_evaluation_method_factory(db_session_sync):
    """Test the evaluation method factory."""
    # Create instances with factory
    ragas_method = EvaluationMethodFactory.create(EvaluationMethod.RAGAS, db_session_sync)
    deepeval_method = EvaluationMethodFactory.create(EvaluationMethod.DEEPEVAL, db_session_sync)
    custom_method = EvaluationMethodFactory.create(EvaluationMethod.CUSTOM, db_session_sync)
    manual_method = EvaluationMethodFactory.create(EvaluationMethod.MANUAL, db_session_sync)

    # Check instances
    assert isinstance(ragas_method, RagasEvaluationMethod)
    assert isinstance(deepeval_method, DeepEvalEvaluationMethod)
    assert isinstance(custom_method, CustomEvaluationMethod)
    assert isinstance(manual_method, ManualEvaluationMethod)

    # Test registering a custom method
    class TestEvaluationMethod(BaseEvaluationMethod):
        async def run_evaluation(self, evaluation):
            return []

        async def calculate_metrics(self, input_data, output_data, config):
            return {}

    # Register the custom method
    EvaluationMethodFactory.register("test_method", TestEvaluationMethod)

    # Create an instance
    test_method = EvaluationMethodFactory.create("test_method", db_session_sync)

    # Check instance
    assert isinstance(test_method, TestEvaluationMethod)

    # Test with invalid method
    with pytest.raises(ValueError):
        EvaluationMethodFactory.create("invalid_method", db_session_sync)