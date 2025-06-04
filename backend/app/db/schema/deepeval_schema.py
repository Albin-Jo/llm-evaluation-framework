from typing import Dict, List, Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field

from backend.app.db.schema.evaluation_schema import EvaluationResultCreate, MetricScoreCreate


class DeepEvalMetricResult(BaseModel):
    """Schema for DeepEval metric results with reasoning."""
    name: str
    score: float
    success: bool
    threshold: float
    reason: Optional[str] = None
    error: Optional[str] = None


class DeepEvalTestCaseResult(BaseModel):
    """Schema for individual DeepEval test case results."""
    input: str
    actual_output: str
    expected_output: Optional[str] = None
    context: List[str] = []
    metrics: List[DeepEvalMetricResult] = []
    overall_success: bool = False


class DeepEvalEvaluationResult(BaseModel):
    """Schema for complete DeepEval evaluation results."""
    evaluation_id: UUID
    test_cases: List[DeepEvalTestCaseResult]
    overall_metrics: Dict[str, Dict[str, float]]  # metric_name -> {avg, min, max, success_rate}
    summary: Dict[str, Any]


def convert_deepeval_to_platform_results(
        deepeval_result: DeepEvalEvaluationResult,
        evaluation_id: UUID
) -> List[EvaluationResultCreate]:
    """
    Convert DeepEval results to platform's EvaluationResultCreate format.

    Args:
        deepeval_result: DeepEval evaluation results
        evaluation_id: Evaluation ID

    Returns:
        List of EvaluationResultCreate objects
    """
    platform_results = []

    for i, test_case in enumerate(deepeval_result.test_cases):
        # Calculate overall score for this test case
        if test_case.metrics:
            overall_score = sum(m.score for m in test_case.metrics) / len(test_case.metrics)
        else:
            overall_score = 0.0

        # Create metric scores
        metric_scores = []
        for metric in test_case.metrics:
            metric_score = MetricScoreCreate(
                name=metric.name,
                value=metric.score,
                weight=1.0,
                meta_info={
                    "success": metric.success,
                    "threshold": metric.threshold,
                    "reason": metric.reason,
                    "error": metric.error
                }
            )
            metric_scores.append(metric_score)

        # Create platform result
        platform_result = EvaluationResultCreate(
            evaluation_id=evaluation_id,
            overall_score=overall_score,
            raw_results={
                "deepeval_metrics": [
                    {
                        "name": m.name,
                        "score": m.score,
                        "success": m.success,
                        "reason": m.reason,
                        "threshold": m.threshold
                    } for m in test_case.metrics
                ],
                "test_case_success": test_case.overall_success
            },
            dataset_sample_id=str(i),
            input_data={
                "input": test_case.input,
                "context": test_case.context,
                "expected_output": test_case.expected_output
            },
            output_data={
                "actual_output": test_case.actual_output
            },
            metric_scores=metric_scores,
            passed=test_case.overall_success,
            pass_threshold=0.7  # Default threshold
        )

        platform_results.append(platform_result)

    return platform_results


class DeepEvalConfig(BaseModel):
    """Configuration for DeepEval evaluations."""
    metrics: List[str] = Field(default=["answer_relevancy", "faithfulness"])
    thresholds: Dict[str, float] = Field(default={})
    include_reasoning: bool = Field(default=True)
    batch_size: int = Field(default=5, ge=1, le=20)
    model_config: Dict[str, Any] = Field(default={})

    def get_threshold(self, metric_name: str) -> float:
        """Get threshold for a specific metric."""
        return self.thresholds.get(metric_name, 0.7)


class DeepEvalValidationResult(BaseModel):
    """Schema for DeepEval dataset validation results."""
    compatible: bool
    warnings: List[str] = []
    requirements: List[str] = []
    supported_metrics: List[str] = []
    recommended_metrics: List[str] = []
    statistics: Dict[str, int] = {}
    conversion_preview: Optional[List[Dict[str, Any]]] = None