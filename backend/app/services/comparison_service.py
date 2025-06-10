import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

import numpy as np
from fastapi import HTTPException, status
from scipy import stats
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.exceptions import AuthorizationException
from backend.app.db.models.orm import EvaluationComparison, Evaluation, EvaluationResult, MetricScore
from backend.app.db.repositories.base import BaseRepository
from backend.app.db.repositories.comparison_repository import ComparisonRepository
from backend.app.db.repositories.evaluation_repository import EvaluationRepository
from backend.app.db.schema.comparison_schema import (
    ComparisonCreate, ComparisonUpdate, MetricDifferenceResponse,
    MetricConfig
)
from backend.app.services.utils.comparison_utils import PDFReportGenerator, StatisticsUtils

logger = logging.getLogger(__name__)

# Enhanced metric configurations with scale information
DEFAULT_METRIC_CONFIGS = {
    # RAGAS metrics (0-1 scale)
    "faithfulness": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "ragas"},
    "response_relevancy": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "ragas"},
    "context_precision": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "ragas"},
    "context_recall": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "ragas"},
    "context_entity_recall": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "ragas"},
    "noise_sensitivity": {"higher_is_better": False, "weight": 1.0, "scale": (0, 1), "method": "ragas"},
    "answer_correctness": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "ragas"},
    "answer_similarity": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "ragas"},
    "answer_relevancy": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "ragas"},
    "factual_correctness": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "ragas"},

    # DeepEval metrics (0-1 scale)
    "answer_relevancy_deepeval": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "deepeval"},
    "faithfulness_deepeval": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "deepeval"},
    "contextual_precision": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "deepeval"},
    "contextual_recall": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "deepeval"},
    "contextual_relevancy": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "deepeval"},
    "hallucination": {"higher_is_better": False, "weight": 1.0, "scale": (0, 1), "method": "deepeval"},
    "toxicity": {"higher_is_better": False, "weight": 1.0, "scale": (0, 1), "method": "deepeval"},
    "bias": {"higher_is_better": False, "weight": 1.0, "scale": (0, 1), "method": "deepeval"},
    "g_eval_coherence": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "deepeval"},
    "g_eval_correctness": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "deepeval"},
    "g_eval_completeness": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "deepeval"},
    "g_eval_helpfulness": {"higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "deepeval"},
}


def _get_config_value(config: Union[Dict, MetricConfig], key: str, default: Any = None) -> Any:
    """
    Safely extract a value from either a MetricConfig object or a dictionary.

    Args:
        config: Either a MetricConfig object or a dictionary
        key: Key to extract
        default: Default value if key not found

    Returns:
        The extracted value or default
    """
    if hasattr(config, key):
        # It's an object with attributes
        return getattr(config, key, default)
    elif isinstance(config, dict):
        # It's a dictionary
        return config.get(key, default)
    else:
        logger.warning(f"Unexpected config type: {type(config)}")
        return default


def _assess_statistical_power(
        metric_comparison: Dict[str, Any],
        sample_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Assess the statistical power of the comparison."""
    sample_size = sample_stats.get("matched_count", 0)

    # Simple power assessment based on sample size and effect sizes
    power_assessment = {
        "sample_size": sample_size,
        "power_category": "low" if sample_size < 30 else "medium" if sample_size < 100 else "high",
        "recommendations": []
    }

    if sample_size < 30:
        power_assessment["recommendations"].append("Consider collecting more data for better statistical power")

    # Count significant results
    significant_count = sum(1 for data in metric_comparison.values()
                            if data.get("comparison", {}).get("is_significant") is True)

    power_assessment["significant_results"] = significant_count
    power_assessment["significance_rate"] = (
            significant_count / len(metric_comparison) * 100) if metric_comparison else 0

    return power_assessment


def _generate_enhanced_insights(
        summary: Dict,
        compatibility_warnings: List[str] = None
) -> str:
    """
    Generate enhanced natural language insights from comparison data.
    """
    insights = []

    # Start with compatibility warnings if any
    if compatibility_warnings:
        insights.append("⚠️ **Compatibility Warnings:**")
        for warning in compatibility_warnings:
            insights.append(f"- {warning}")
        insights.append("")

    # Enhanced overall assessment
    overall_result = summary.get("overall_result", "inconclusive")
    percentage_change = summary.get("percentage_change", 0)
    eval_a_name = summary.get("evaluation_a_name", "Evaluation A")
    eval_b_name = summary.get("evaluation_b_name", "Evaluation B")

    # Check if cross-method comparison
    if summary.get("cross_method_comparison", False):
        method_a = summary.get("evaluation_a_method", "unknown")
        method_b = summary.get("evaluation_b_method", "unknown")
        insights.append(f"🔄 **Cross-Method Comparison:** Comparing {method_a} vs {method_b} evaluations. "
                        f"Metric normalization has been applied for fair comparison.")
        insights.append("")

    if overall_result == "improved":
        insights.append(f"📈 **Overall Assessment:** {eval_b_name} shows a "
                        f"{percentage_change:.1f}% improvement over {eval_a_name}.")
    elif overall_result == "regressed":
        insights.append(f"📉 **Overall Assessment:** {eval_b_name} shows a "
                        f"{abs(percentage_change):.1f}% regression compared to {eval_a_name}.")
    else:
        insights.append(f"⚖️ **Overall Assessment:** The comparison between {eval_a_name} and "
                        f"{eval_b_name} is inconclusive due to insufficient data.")

    # Enhanced sample analysis
    total_compared = summary.get("matched_samples", 0)
    if total_compared > 0:
        imp_rate = (summary.get("improved_samples", 0) / total_compared) * 100
        insights.append(f"\n**Sample Analysis:** Of {total_compared} samples analyzed, "
                        f"{summary.get('improved_samples', 0)} ({imp_rate:.1f}%) improved and "
                        f"{summary.get('regressed_samples', 0)} ({100 - imp_rate:.1f}%) regressed.")

    # Statistical power assessment
    if summary.get("statistical_power"):
        power_info = summary["statistical_power"]
        insights.append(f"\n**Statistical Power:** {power_info['power_category'].title()} power "
                        f"(sample size: {power_info['sample_size']})")

    # Enhanced metric analysis with effect sizes
    if summary.get("top_improvements"):
        insights.append("\n**Top Improvements:**")
        for m in summary["top_improvements"][:3]:
            sig_marker = "* " if m.get("is_significant", False) else ""
            effect_info = f" (Effect: {m['effect_magnitude']})" if m.get("effect_magnitude") else ""
            insights.append(f"- {sig_marker}{m['metric_name']}: +{m['percentage_change']:.1f}%{effect_info}")

    if summary.get("top_regressions"):
        insights.append("\n**Areas for Attention:**")
        for m in summary["top_regressions"][:3]:
            sig_marker = "* " if m.get("is_significant", False) else ""
            effect_info = f" (Effect: {m['effect_magnitude']})" if m.get("effect_magnitude") else ""
            insights.append(f"- {sig_marker}{m['metric_name']}: {m['percentage_change']:.1f}%{effect_info}")

    # Enhanced significance analysis
    total_significant = summary.get("significant_improvements", 0) + summary.get("significant_regressions", 0)
    if total_significant > 0:
        insights.append(f"\n**Statistical Significance:** {total_significant} of {summary.get('total_metrics', 0)} "
                        f"metrics show statistically significant changes after multiple comparison correction "
                        f"(marked with *).")

    # Consistency analysis
    if summary.get("consistency_score") is not None:
        consistency = summary["consistency_score"]
        consistency_desc = "high" if consistency > 0.8 else "moderate" if consistency > 0.5 else "low"
        insights.append(f"\n**Consistency Analysis:** The changes show {consistency_desc} consistency "
                        f"across metrics (consistency score: {consistency:.2f}).")

    # Enhanced weighted analysis
    if summary.get("weighted_improvement_score") is not None:
        weighted_score = summary["weighted_improvement_score"]
        if abs(weighted_score) > 0.1:  # Meaningful change threshold
            direction = "positive" if weighted_score > 0 else "negative"
            insights.append(f"\n**Weighted Analysis:** Considering metric weights and importance, "
                            f"the overall impact is {direction} ({weighted_score:.3f}).")
        else:
            insights.append(f"\n**Weighted Analysis:** The weighted impact is minimal ({weighted_score:.3f}), "
                            f"suggesting balanced trade-offs between metrics.")

    # Enhanced conclusion with actionability
    if overall_result == "improved":
        if summary.get("consistency_score", 0) > 0.7:
            insights.append("\n**Conclusion:** This change represents a consistent and meaningful improvement "
                            "across multiple metrics. Strongly recommended for adoption.")
        else:
            insights.append("\n**Conclusion:** This change shows overall improvement but with some inconsistency. "
                            "Review specific metric trade-offs before adoption.")
    elif overall_result == "regressed":
        insights.append("\n**Conclusion:** This change shows overall regression in performance. "
                        "Requires investigation and optimization before consideration for adoption.")
    else:
        insights.append("\n**Conclusion:** More data or different evaluation approaches may be needed "
                        "to make a definitive assessment of this change.")

    return "\n".join(insights)


def _check_evaluation_compatibility_detailed(
        evaluation_a: Evaluation, evaluation_b: Evaluation
) -> Dict[str, List[str]]:
    """
    Enhanced compatibility check between two evaluations.

    Args:
        evaluation_a: First evaluation
        evaluation_b: Second evaluation

    Returns:
        Dict with 'errors' (blocking issues) and 'warnings' (non-blocking issues)
    """
    errors = []
    warnings = []

    # Check dataset compatibility
    if evaluation_a.dataset_id != evaluation_b.dataset_id:
        warnings.append(
            f"Evaluations use different datasets: {evaluation_a.dataset_id} vs {evaluation_b.dataset_id}")

    # Check method compatibility with enhanced logic
    method_a = evaluation_a.method.value if evaluation_a.method else "unknown"
    method_b = evaluation_b.method.value if evaluation_b.method else "unknown"

    if method_a != method_b:
        warnings.append(f"Different evaluation methods detected: {method_a} vs {method_b}. "
                        f"Metric normalization will be applied for fair comparison.")

    # Check metric compatibility with more detailed analysis
    if evaluation_a.metrics and evaluation_b.metrics:
        metrics_a = set(evaluation_a.metrics)
        metrics_b = set(evaluation_b.metrics)

        if metrics_a != metrics_b:
            only_in_a = metrics_a - metrics_b
            only_in_b = metrics_b - metrics_a
            common_metrics = metrics_a & metrics_b

            if len(common_metrics) == 0:
                errors.append("No common metrics found between evaluations")
            else:
                if only_in_a:
                    warnings.append(f"Metrics only in evaluation A: {', '.join(only_in_a)}")
                if only_in_b:
                    warnings.append(f"Metrics only in evaluation B: {', '.join(only_in_b)}")
                warnings.append(f"Comparison will focus on {len(common_metrics)} common metrics")

    # Check sample size compatibility
    if len(evaluation_a.results) == 0 and len(evaluation_b.results) == 0:
        errors.append("Both evaluations have no results")
    elif len(evaluation_a.results) == 0 or len(evaluation_b.results) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"One evaluation has no results: {len(evaluation_a.results)} vs {len(evaluation_b.results)}"
        )
    else:
        size_a, size_b = len(evaluation_a.results), len(evaluation_b.results)
        size_diff_ratio = abs(size_a - size_b) / max(size_a, size_b)

        if size_diff_ratio > 0.5:  # More than 50% difference
            warnings.append(
                f"Large difference in sample sizes: {size_a} vs {size_b} (may affect statistical significance)")

    # Check evaluation status
    if evaluation_a.status.value != "completed":
        errors.append(f"Evaluation A is not completed (status: {evaluation_a.status.value})")
    if evaluation_b.status.value != "completed":
        errors.append(f"Evaluation B is not completed (status: {evaluation_b.status.value})")

    return {"errors": errors, "warnings": warnings}


async def _calculate_metric_comparison_enhanced(
        evaluation_a: Evaluation,
        evaluation_b: Evaluation,
        metric_configs: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Metric comparison with percentage calculations and statistical rigor.
    """
    # Get all metric scores for each evaluation
    metric_scores_a = {}
    metric_scores_b = {}

    # Process evaluation A results
    for result in evaluation_a.results:
        for metric_score in result.metric_scores:
            metric_name = metric_score.name
            if metric_name not in metric_scores_a:
                metric_scores_a[metric_name] = []
            metric_scores_a[metric_name].append(metric_score.value)

    # Process evaluation B results
    for result in evaluation_b.results:
        for metric_score in result.metric_scores:
            metric_name = metric_score.name
            if metric_name not in metric_scores_b:
                metric_scores_b[metric_name] = []
            metric_scores_b[metric_name].append(metric_score.value)

    # Calculate comparisons
    metric_comparison = {}
    all_metrics = set(list(metric_scores_a.keys()) + list(metric_scores_b.keys()))

    # Collect p-values for multiple comparison correction
    p_values_for_correction = []
    metric_names_with_p_values = []

    for metric_name in all_metrics:
        # Get metric configuration
        metric_config = metric_configs.get(metric_name, {
            "higher_is_better": True,
            "weight": 1.0,
            "scale": (0, 1),
            "method": "unknown"
        })

        higher_is_better = metric_config.get("higher_is_better", True)
        metric_weight = metric_config.get("weight", 1.0)
        metric_scale = metric_config.get("scale", (0, 1))

        # Calculate averages
        values_a = metric_scores_a.get(metric_name, [])
        values_b = metric_scores_b.get(metric_name, [])

        # Skip if no values
        if not values_a and not values_b:
            continue

        # Normalize values if scales differ or if cross-method comparison
        method_a = evaluation_a.method.value if evaluation_a.method else "unknown"
        method_b = evaluation_b.method.value if evaluation_b.method else "unknown"

        if method_a != method_b and values_a and values_b:
            # Apply normalization for cross-method comparison
            all_values = values_a + values_b
            if all_values:
                actual_min, actual_max = min(all_values), max(all_values)
                if actual_max > actual_min:  # Avoid division by zero
                    values_a = StatisticsUtils.normalize_metric_values(
                        values_a, (actual_min, actual_max), metric_scale)
                    values_b = StatisticsUtils.normalize_metric_values(
                        values_b, (actual_min, actual_max), metric_scale)

        # Calculate statistics for A
        stats_a = {}
        if values_a:
            values_a_np = np.array(values_a)
            stats_a = {
                "average": float(np.mean(values_a_np)),
                "median": float(np.median(values_a_np)),
                "std_dev": float(np.std(values_a_np)) if len(values_a) > 1 else None,
                "min": float(np.min(values_a_np)),
                "max": float(np.max(values_a_np)),
                "q1": float(np.percentile(values_a_np, 25)) if len(values_a) > 1 else None,
                "q3": float(np.percentile(values_a_np, 75)) if len(values_a) > 1 else None,
                "sample_count": len(values_a),
                "values": values_a
            }

        # Calculate statistics for B
        stats_b = {}
        if values_b:
            values_b_np = np.array(values_b)
            stats_b = {
                "average": float(np.mean(values_b_np)),
                "median": float(np.median(values_b_np)),
                "std_dev": float(np.std(values_b_np)) if len(values_b) > 1 else None,
                "min": float(np.min(values_b_np)),
                "max": float(np.max(values_b_np)),
                "q1": float(np.percentile(values_b_np, 25)) if len(values_b) > 1 else None,
                "q3": float(np.percentile(values_b_np, 75)) if len(values_b) > 1 else None,
                "sample_count": len(values_b),
                "values": values_b
            }

        # If both have values, calculate enhanced comparison statistics
        comparison_stats = {}
        if values_a and values_b:
            avg_a = stats_a["average"]
            avg_b = stats_b["average"]

            # Enhanced percentage change calculation
            absolute_diff = avg_b - avg_a
            percentage_diff = StatisticsUtils.safe_percentage_change(avg_b, avg_a)

            # Determine if this is an improvement based on metric direction
            is_improvement = (absolute_diff > 0) if higher_is_better else (absolute_diff < 0)

            # Calculate statistical significance and effect size
            p_value = None
            is_significant = None
            effect_size = None

            if len(values_a) > 1 and len(values_b) > 1:
                try:
                    # Use Welch's t-test (unequal variances)
                    t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)

                    # Store for multiple comparison correction
                    p_values_for_correction.append(p_value)
                    metric_names_with_p_values.append(metric_name)

                    # Calculate effect size
                    effect_size = StatisticsUtils.calculate_effect_size(values_a, values_b)

                except Exception as e:
                    logger.warning(f"Failed to calculate statistical significance for {metric_name}: {e}")

            comparison_stats = {
                "absolute_difference": absolute_diff,
                "percentage_change": percentage_diff,
                "is_improvement": is_improvement,
                "p_value": p_value,
                "is_significant": is_significant,  # Will be updated after correction
                "effect_size": effect_size,
                "weight": metric_weight
            }

        # Add metric data to comparison results
        metric_comparison[metric_name] = {
            "evaluation_a": stats_a,
            "evaluation_b": stats_b,
            "comparison": comparison_stats,
            "config": {
                "higher_is_better": higher_is_better,
                "weight": metric_weight,
                "scale": metric_scale,
                "method": metric_config.get("method", "unknown")
            }
        }

    # Apply multiple comparison correction
    if p_values_for_correction:
        corrected_p_values = StatisticsUtils.apply_multiple_comparison_correction(
            p_values_for_correction, method='bonferroni')

        # Update significance based on corrected p-values
        for i, metric_name in enumerate(metric_names_with_p_values):
            if metric_name in metric_comparison and "comparison" in metric_comparison[metric_name]:
                corrected_p = corrected_p_values[i]
                metric_comparison[metric_name]["comparison"]["corrected_p_value"] = corrected_p
                metric_comparison[metric_name]["comparison"]["is_significant"] = corrected_p < 0.05

    return metric_comparison


async def _calculate_sample_comparison_enhanced(
        evaluation_a: Evaluation,
        evaluation_b: Evaluation,
        metric_configs: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Enhanced sample comparison with improved percentage calculations.
    """
    # Match results by dataset_sample_id
    matched_results = {}

    # Organize results by sample ID
    results_a_by_sample = {
        result.dataset_sample_id: result for result in evaluation_a.results
        if result.dataset_sample_id is not None
    }

    results_b_by_sample = {
        result.dataset_sample_id: result for result in evaluation_b.results
        if result.dataset_sample_id is not None
    }

    # Find all unique sample IDs
    all_sample_ids = set(list(results_a_by_sample.keys()) + list(results_b_by_sample.keys()))
    if not all_sample_ids:
        logger.warning("No sample IDs found in evaluation results")
        return {
            "matched_results": {},
            "stats": {
                "matched_count": 0,
                "only_in_evaluation_a": 0,
                "only_in_evaluation_b": 0,
                "total_samples": 0,
                "improved_samples": 0,
                "regressed_samples": 0,
                "improvement_rate": 0,
                "significant_improvements": 0,
                "significant_regressions": 0
            }
        }

    # Compare matched samples with enhanced logic
    sample_improvements = 0
    sample_regressions = 0
    significant_improvements = 0
    significant_regressions = 0

    for sample_id in all_sample_ids:
        result_a = results_a_by_sample.get(sample_id)
        result_b = results_b_by_sample.get(sample_id)

        if result_a and result_b:
            # Both evaluations have this sample
            score_a = result_a.overall_score
            score_b = result_b.overall_score

            if score_a is not None and score_b is not None:
                # Enhanced percentage change calculation
                absolute_diff = score_b - score_a
                percentage_diff = StatisticsUtils.safe_percentage_change(score_b, score_a)

                # For overall scores, higher is always better
                is_improvement = absolute_diff > 0

                # Track improvements/regressions
                if is_improvement:
                    sample_improvements += 1
                else:
                    sample_regressions += 1

                # Record comparison
                matched_results[sample_id] = {
                    "evaluation_a": {
                        "overall_score": score_a,
                        "result_id": str(result_a.id)
                    },
                    "evaluation_b": {
                        "overall_score": score_b,
                        "result_id": str(result_b.id)
                    },
                    "comparison": {
                        "absolute_difference": absolute_diff,
                        "percentage_change": percentage_diff,
                        "is_improvement": is_improvement
                    }
                }

                # Add enhanced metric-level comparison for this sample
                metrics_a = {score.name: score.value for score in result_a.metric_scores}
                metrics_b = {score.name: score.value for score in result_b.metric_scores}

                metric_diffs = {}
                all_metric_names = set(list(metrics_a.keys()) + list(metrics_b.keys()))

                for metric_name in all_metric_names:
                    value_a = metrics_a.get(metric_name)
                    value_b = metrics_b.get(metric_name)

                    if value_a is not None and value_b is not None:
                        # Get metric configuration
                        metric_config = metric_configs.get(metric_name, {
                            "higher_is_better": True,
                            "weight": 1.0
                        })
                        higher_is_better = metric_config.get("higher_is_better", True)

                        metric_diff = value_b - value_a

                        # Enhanced percentage change calculation
                        metric_percent = StatisticsUtils.safe_percentage_change(value_b, value_a)

                        # Determine if this is an improvement based on metric direction
                        metric_improvement = (metric_diff > 0) if higher_is_better else (metric_diff < 0)

                        metric_diffs[metric_name] = {
                            "evaluation_a_value": value_a,
                            "evaluation_b_value": value_b,
                            "absolute_difference": metric_diff,
                            "percentage_change": metric_percent,
                            "is_improvement": metric_improvement,
                            "higher_is_better": higher_is_better
                        }
                    else:
                        metric_diffs[metric_name] = {
                            "evaluation_a_value": value_a,
                            "evaluation_b_value": value_b,
                            "missing_in": "evaluation_a" if value_a is None else (
                                "evaluation_b" if value_b is None else None)
                        }

                matched_results[sample_id]["metric_differences"] = metric_diffs
        else:
            # Record unmatched samples
            matched_results[sample_id] = {
                "evaluation_a": {
                    "overall_score": result_a.overall_score if result_a else None,
                    "result_id": str(result_a.id) if result_a else None
                },
                "evaluation_b": {
                    "overall_score": result_b.overall_score if result_b else None,
                    "result_id": str(result_b.id) if result_b else None
                },
                "comparison": {
                    "missing_in": "evaluation_a" if not result_a else ("evaluation_b" if not result_b else None)
                }
            }

    # Calculate enhanced sample statistics
    matched_count = len([s for s in matched_results.values() if
                         s.get("evaluation_a", {}).get("result_id") and
                         s.get("evaluation_b", {}).get("result_id")])

    only_in_a = len([s for s in matched_results.values() if
                     s.get("evaluation_a", {}).get("result_id") and
                     not s.get("evaluation_b", {}).get("result_id")])

    only_in_b = len([s for s in matched_results.values() if
                     not s.get("evaluation_a", {}).get("result_id") and
                     s.get("evaluation_b", {}).get("result_id")])

    # Calculate improvement rates safely
    improvement_rate = (sample_improvements / matched_count) * 100 if matched_count > 0 else 0
    significant_improvement_rate = (significant_improvements / matched_count) * 100 if matched_count > 0 else 0

    return {
        "matched_results": matched_results,
        "stats": {
            "matched_count": matched_count,
            "only_in_evaluation_a": only_in_a,
            "only_in_evaluation_b": only_in_b,
            "total_samples": len(all_sample_ids),
            "improved_samples": sample_improvements,
            "regressed_samples": sample_regressions,
            "improvement_rate": improvement_rate,
            "significant_improvements": significant_improvements,
            "significant_regressions": significant_regressions,
            "significant_improvement_rate": significant_improvement_rate
        }
    }


class ComparisonService:
    """Enhanced service for handling comparison operations."""

    def __init__(self, db_session: AsyncSession):
        """
        Initialize the comparison service.

        Args:
            db_session: Database session
        """
        self.db_session = db_session
        self.comparison_repo = ComparisonRepository(db_session)
        self.evaluation_repo = EvaluationRepository(db_session)
        self.result_repo = BaseRepository(EvaluationResult, db_session)
        self.metric_repo = BaseRepository(MetricScore, db_session)
        self.pdf_generator = PDFReportGenerator()

    async def create_comparison(
            self, comparison_data: ComparisonCreate
    ) -> EvaluationComparison:
        """
        Create a new comparison with user attribution.

        Args:
            comparison_data: Comparison data

        Returns:
            EvaluationComparison: Created comparison

        Raises:
            HTTPException: If referenced entities don't exist or validation fails
        """
        # Verify that referenced evaluations exist
        evaluation_a = await self.evaluation_repo.get_evaluation_with_details(comparison_data.evaluation_a_id)
        if not evaluation_a:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation A with ID {comparison_data.evaluation_a_id} not found"
            )

        evaluation_b = await self.evaluation_repo.get_evaluation_with_details(comparison_data.evaluation_b_id)
        if not evaluation_b:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation B with ID {comparison_data.evaluation_b_id} not found"
            )

        # Check if a comparison for these evaluations already exists
        existing_comparison = await self.comparison_repo.get_by_evaluations(
            comparison_data.evaluation_a_id,
            comparison_data.evaluation_b_id,
            comparison_data.created_by_id
        )

        if existing_comparison:
            logger.info(f"Comparison for evaluations {comparison_data.evaluation_a_id} and "
                        f"{comparison_data.evaluation_b_id} already exists: {existing_comparison.id}")
            return existing_comparison

        # Enhanced compatibility check before creation
        compatibility_issues = _check_evaluation_compatibility_detailed(evaluation_a, evaluation_b)
        if compatibility_issues.get("errors"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Evaluations are not compatible for comparison: {'; '.join(compatibility_issues['errors'])}"
            )

        # If no metric configs provided, use defaults
        if not comparison_data.metric_configs:
            comparison_data.metric_configs = DEFAULT_METRIC_CONFIGS

        # Create comparison
        comparison_dict = comparison_data.model_dump()
        comparison_dict["status"] = "pending"  # Initial status

        try:
            comparison = await self.comparison_repo.create(comparison_dict)
            logger.info(f"Created comparison {comparison.id} between "
                        f"evaluations {comparison_data.evaluation_a_id} and {comparison_data.evaluation_b_id}")
            return comparison
        except Exception as e:
            logger.error(f"Failed to create comparison: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create comparison: {str(e)}"
            )

    async def run_comparison_calculation(
            self, comparison_id: UUID, user_id: Optional[UUID] = None
    ) -> EvaluationComparison:
        """
        Perform the actual comparison calculation and update the comparison record.

        Args:
            comparison_id: Comparison ID
            user_id: Optional user ID for ownership verification

        Returns:
            EvaluationComparison: Updated comparison with results

        Raises:
            HTTPException: If comparison cannot be processed or user doesn't have permission
        """
        # Get comparison with user verification if user_id provided
        if user_id:
            comparison = await self.comparison_repo.get_user_owned(comparison_id, user_id)
            if not comparison:
                logger.warning(f"Comparison with ID {comparison_id} not found or user {user_id} doesn't have access")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Comparison with ID {comparison_id} not found or you don't have permission to run it"
                )
        else:
            comparison = await self.comparison_repo.get(comparison_id)
            if not comparison:
                logger.warning(f"Comparison with ID {comparison_id} not found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Comparison with ID {comparison_id} not found"
                )

        try:
            # Get detailed evaluations with repository
            evaluation_a = await self.evaluation_repo.get_evaluation_with_details(comparison.evaluation_a_id)
            evaluation_b = await self.evaluation_repo.get_evaluation_with_details(comparison.evaluation_b_id)

            if not evaluation_a or not evaluation_b:
                logger.error(f"One or both evaluations not found for comparison {comparison_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="One or both referenced evaluations not found"
                )

            # Enhanced compatibility check
            compatibility_check = _check_evaluation_compatibility_detailed(evaluation_a, evaluation_b)

            if compatibility_check.get("errors"):
                # Fatal compatibility issues
                error_msg = f"Evaluations are incompatible: {'; '.join(compatibility_check['errors'])}"
                await self.comparison_repo.update(comparison_id, {"status": "failed", "error": error_msg})
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg
                )

            # Update status to "running"
            await self.comparison_repo.update(comparison_id, {"status": "running"})

            # Calculate comparisons with enhanced logic
            comparison_results = await self._calculate_comparison_enhanced(
                evaluation_a, evaluation_b, comparison.metric_configs)

            # Add compatibility warnings
            comparison_results["compatibility_warnings"] = compatibility_check.get("warnings", [])

            # Generate enhanced natural language insights
            narrative_insights = _generate_enhanced_insights(
                comparison_results["summary"],
                compatibility_check.get("warnings", [])
            )

            # Update comparison with results
            update_data = {
                "status": "completed",
                "comparison_results": comparison_results["detailed_results"],
                "summary": comparison_results["summary"],
                "narrative_insights": narrative_insights
            }

            updated_comparison = await self.comparison_repo.update(comparison_id, update_data)
            logger.info(f"Completed comparison calculation for comparison {comparison_id}")

            return updated_comparison

        except Exception as e:
            logger.error(f"Error calculating comparison {comparison_id}: {str(e)}")
            # Update status to "failed" with error details
            await self.comparison_repo.update(
                comparison_id,
                {
                    "status": "failed",
                    "error": str(e)
                }
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error calculating comparison: {str(e)}"
            )

    async def _calculate_comparison_enhanced(
            self,
            evaluation_a: Evaluation,
            evaluation_b: Evaluation,
            metric_configs: Optional[Union[Dict[str, MetricConfig], Dict[str, Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced comparison calculation with improved statistical methods.

        Args:
            evaluation_a: First evaluation
            evaluation_b: Second evaluation
            metric_configs: Optional configuration for metrics (can be MetricConfig objects or dictionaries)

        Returns:
            Dict with comparison results and summary
        """
        try:
            # Initialize results
            detailed_results = {
                "metric_comparison": {},
                "sample_comparison": {},
                "overall_comparison": {},
                "statistical_summary": {}
            }

            # Prepare enhanced metric configurations with robust type handling
            final_metric_configs = self._prepare_enhanced_metric_configs(
                evaluation_a, evaluation_b, metric_configs)

            # Calculate enhanced metric-level comparison
            metric_comparison = await _calculate_metric_comparison_enhanced(
                evaluation_a, evaluation_b, final_metric_configs)
            detailed_results["metric_comparison"] = metric_comparison

            # Calculate sample-level comparison (matching results by dataset item)
            sample_comparison = await _calculate_sample_comparison_enhanced(
                evaluation_a, evaluation_b, final_metric_configs)
            detailed_results["sample_comparison"] = sample_comparison

            # Calculate overall evaluation comparison with enhanced scoring
            overall_comparison = self._calculate_overall_comparison_enhanced(
                evaluation_a, evaluation_b, metric_comparison)
            detailed_results["overall_comparison"] = overall_comparison

            # Generate enhanced statistical summary
            statistical_summary = self._generate_statistical_summary(
                metric_comparison, sample_comparison, overall_comparison)
            detailed_results["statistical_summary"] = statistical_summary

            # Generate enhanced summary
            summary = self._generate_comparison_summary_enhanced(
                evaluation_a,
                evaluation_b,
                metric_comparison,
                sample_comparison,
                overall_comparison,
                statistical_summary
            )

            return {
                "detailed_results": detailed_results,
                "summary": summary
            }

        except Exception as e:
            logger.error(f"Error in _calculate_comparison_enhanced: {str(e)}")
            logger.error(f"metric_configs type: {type(metric_configs)}")
            if metric_configs:
                logger.error(f"metric_configs sample: {dict(list(metric_configs.items())[:2])}")
            raise

    @staticmethod
    def _prepare_enhanced_metric_configs(
            evaluation_a: Evaluation,
            evaluation_b: Evaluation,
            metric_configs: Optional[Union[Dict[str, MetricConfig], Dict[str, Dict[str, Any]]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Prepare enhanced metric configurations with normalization support and robust type handling.

        Args:
            evaluation_a: First evaluation
            evaluation_b: Second evaluation
            metric_configs: Optional user-provided configs (can be MetricConfig objects or dictionaries)

        Returns:
            Dict with enhanced metric configurations
        """
        final_configs = {}

        # Get all metrics from both evaluations
        all_metrics = set()
        if evaluation_a.metrics:
            all_metrics.update(evaluation_a.metrics)
        if evaluation_b.metrics:
            all_metrics.update(evaluation_b.metrics)

        for metric_name in all_metrics:
            # Start with default config
            config = DEFAULT_METRIC_CONFIGS.get(metric_name, {
                "higher_is_better": True,
                "weight": 1.0,
                "scale": (0, 1),
                "method": "unknown"
            }).copy()

            # Override with user-provided config if available - with robust type handling
            if metric_configs and metric_name in metric_configs:
                user_config = metric_configs[metric_name]

                try:
                    # Handle both MetricConfig objects and dictionaries
                    higher_is_better = _get_config_value(user_config, "higher_is_better")
                    weight = _get_config_value(user_config, "weight")

                    if higher_is_better is not None:
                        config["higher_is_better"] = higher_is_better
                    if weight is not None:
                        config["weight"] = weight

                except Exception as e:
                    logger.warning(f"Error processing config for metric {metric_name}: {e}")
                    logger.warning(f"Config type: {type(user_config)}, value: {user_config}")
                    # Continue with default config

            final_configs[metric_name] = config

        return final_configs

    # Rest of the methods remain the same as they don't have the type handling issue
    # ... (continuing with the existing methods)

    @staticmethod
    def _calculate_overall_comparison_enhanced(
            evaluation_a: Evaluation,
            evaluation_b: Evaluation,
            metric_comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhanced overall comparison with improved weighted scoring.
        """
        # Calculate overall score comparison
        if not evaluation_a.results:
            logger.warning(f"Evaluation A ({evaluation_a.id}) has no results")
            overall_score_a = None
        else:
            valid_scores_a = [r.overall_score for r in evaluation_a.results if r.overall_score is not None]
            overall_score_a = sum(valid_scores_a) / len(valid_scores_a) if valid_scores_a else None

        if not evaluation_b.results:
            logger.warning(f"Evaluation B ({evaluation_b.id}) has no results")
            overall_score_b = None
        else:
            valid_scores_b = [r.overall_score for r in evaluation_b.results if r.overall_score is not None]
            overall_score_b = sum(valid_scores_b) / len(valid_scores_b) if valid_scores_b else None

        # Enhanced overall score comparison
        overall_diff = None
        overall_pct = None
        is_improvement = None

        if overall_score_a is not None and overall_score_b is not None:
            overall_diff = overall_score_b - overall_score_a
            overall_pct = StatisticsUtils.safe_percentage_change(overall_score_b, overall_score_a)
            is_improvement = overall_diff > 0  # For overall scores, higher is always better

        # Enhanced metric improvements analysis
        metric_improvements = 0
        metric_regressions = 0
        significant_improvements = 0
        significant_regressions = 0

        # Enhanced weighted improvement score calculation
        weighted_sum = 0
        total_weight = 0
        normalized_improvement_scores = []

        for metric_name, metric_data in metric_comparison.items():
            if "comparison" in metric_data and metric_data["comparison"].get("is_improvement") is not None:
                # Get weight and improvement data
                weight = metric_data["comparison"].get("weight", 1.0)
                is_metric_improvement = metric_data["comparison"]["is_improvement"]
                percentage_change = metric_data["comparison"].get("percentage_change", 0)

                # Count improvements and regressions
                if is_metric_improvement:
                    metric_improvements += 1
                else:
                    metric_regressions += 1

                # Check if significant
                if metric_data["comparison"].get("is_significant") is True:
                    if is_metric_improvement:
                        significant_improvements += 1
                    else:
                        significant_regressions += 1

                # Enhanced weighted score calculation
                # Normalize percentage change to a comparable scale (-1 to 1)
                normalized_change = max(-1.0, min(1.0, percentage_change / 100.0))

                # Apply direction (negative for regressions)
                if not is_metric_improvement:
                    normalized_change = -abs(normalized_change)
                else:
                    normalized_change = abs(normalized_change)

                weighted_sum += normalized_change * weight
                total_weight += weight
                normalized_improvement_scores.append(normalized_change)

        # Calculate final weighted improvement score
        normalized_weighted_score = None
        if total_weight > 0:
            normalized_weighted_score = weighted_sum / total_weight

        # Calculate additional statistics
        significance_rate = 0
        if metric_comparison:
            significant_metrics = significant_improvements + significant_regressions
            significance_rate = (significant_metrics / len(metric_comparison)) * 100

        metric_improvement_rate = 0
        if metric_comparison:
            metric_improvement_rate = (metric_improvements / len(metric_comparison)) * 100

        # Calculate consistency score (how consistent are the improvements)
        consistency_score = None
        if normalized_improvement_scores:
            # Calculate variance of improvement scores (lower variance = more consistent)
            score_variance = np.var(normalized_improvement_scores)
            consistency_score = max(0, 1 - score_variance)  # Convert to 0-1 scale

        return {
            "overall_scores": {
                "evaluation_a": overall_score_a,
                "evaluation_b": overall_score_b,
                "absolute_difference": overall_diff,
                "percentage_change": overall_pct,
                "is_improvement": is_improvement
            },
            "metric_stats": {
                "total_metrics": len(metric_comparison),
                "improved_metrics": metric_improvements,
                "regressed_metrics": metric_regressions,
                "significant_improvements": significant_improvements,
                "significant_regressions": significant_regressions,
                "significance_rate": significance_rate,
                "metric_improvement_rate": metric_improvement_rate,
                "weighted_improvement_score": normalized_weighted_score,
                "consistency_score": consistency_score
            }
        }

    def _generate_statistical_summary(
            self,
            metric_comparison: Dict[str, Any],
            sample_comparison: Dict[str, Any],
            overall_comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate enhanced statistical summary.
        """
        summary = {
            "sample_statistics": sample_comparison.get("stats", {}),
            "metric_statistics": overall_comparison.get("metric_stats", {}),
            "overall_statistics": overall_comparison.get("overall_scores", {}),
            "effect_sizes": {},
            "confidence_intervals": {},
            "power_analysis": {}
        }

        # Collect effect sizes
        effect_sizes = []
        for metric_name, data in metric_comparison.items():
            if "comparison" in data and data["comparison"].get("effect_size") is not None:
                effect_size = data["comparison"]["effect_size"]
                effect_sizes.append({
                    "metric": metric_name,
                    "effect_size": effect_size,
                    "magnitude": self._interpret_effect_size(effect_size)
                })

        summary["effect_sizes"] = {
            "metric_effect_sizes": effect_sizes,
            "average_effect_size": np.mean([es["effect_size"] for es in effect_sizes]) if effect_sizes else None
        }

        return summary

    @staticmethod
    def _interpret_effect_size(effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

    def _generate_comparison_summary_enhanced(
            self,
            evaluation_a: Evaluation,
            evaluation_b: Evaluation,
            metric_comparison: Dict[str, Any],
            sample_comparison: Dict[str, Any],
            overall_comparison: Dict[str, Any],
            statistical_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate enhanced comparison summary with additional insights.
        """
        # Extract key information
        overall_stats = overall_comparison["overall_scores"]
        metric_stats = overall_comparison["metric_stats"]
        sample_stats = sample_comparison["stats"]

        # Determine if there's an overall improvement
        is_improvement = overall_stats.get("is_improvement", False)
        percentage_change = overall_stats.get("percentage_change", 0)

        # Identify top improvements and regressions with effect sizes
        top_improvements = []
        top_regressions = []

        for metric_name, data in metric_comparison.items():
            if "comparison" in data and data["comparison"].get("absolute_difference") is not None:
                diff_data = {
                    "metric_name": metric_name,
                    "absolute_difference": data["comparison"]["absolute_difference"],
                    "percentage_change": data["comparison"]["percentage_change"],
                    "is_significant": data["comparison"].get("is_significant", False),
                    "p_value": data["comparison"].get("p_value"),
                    "corrected_p_value": data["comparison"].get("corrected_p_value"),
                    "effect_size": data["comparison"].get("effect_size"),
                    "effect_magnitude": self._interpret_effect_size(data["comparison"]["effect_size"])
                    if data["comparison"].get("effect_size") else None
                }

                if data["comparison"]["is_improvement"]:
                    top_improvements.append(diff_data)
                else:
                    top_regressions.append(diff_data)

        # Sort by effect size if available, otherwise by absolute difference
        top_improvements = sorted(
            top_improvements,
            key=lambda x: abs(x["effect_size"]) if x["effect_size"] is not None else abs(x["absolute_difference"]),
            reverse=True
        )[:3]

        top_regressions = sorted(
            top_regressions,
            key=lambda x: abs(x["effect_size"]) if x["effect_size"] is not None else abs(x["absolute_difference"]),
            reverse=True
        )[:3]

        # Enhanced summary
        summary = {
            "evaluation_a_name": evaluation_a.name,
            "evaluation_b_name": evaluation_b.name,
            "evaluation_a_method": evaluation_a.method.value if evaluation_a.method else "unknown",
            "evaluation_b_method": evaluation_b.method.value if evaluation_b.method else "unknown",
            "overall_result": "improved" if is_improvement else "regressed" if is_improvement is not None else "inconclusive",
            "percentage_change": percentage_change,
            "total_metrics": metric_stats["total_metrics"],
            "improved_metrics": metric_stats["improved_metrics"],
            "regressed_metrics": metric_stats["regressed_metrics"],
            "significant_improvements": metric_stats.get("significant_improvements", 0),
            "significant_regressions": metric_stats.get("significant_regressions", 0),
            "improved_samples": sample_stats["improved_samples"],
            "regressed_samples": sample_stats["regressed_samples"],
            "matched_samples": sample_stats["matched_count"],
            "top_improvements": top_improvements,
            "top_regressions": top_regressions,
            "weighted_improvement_score": metric_stats.get("weighted_improvement_score"),
            "consistency_score": metric_stats.get("consistency_score"),
            "cross_method_comparison": evaluation_a.method != evaluation_b.method,
            "effect_size_summary": statistical_summary.get("effect_sizes", {}),
            "statistical_power": _assess_statistical_power(metric_comparison, sample_stats)
        }

        return summary

    # Keep existing methods for backward compatibility
    async def get_comparison(self, comparison_id: UUID) -> Optional[EvaluationComparison]:
        """Get comparison by ID."""
        try:
            return await self.comparison_repo.get_with_evaluations(comparison_id)
        except Exception as e:
            logger.error(f"Error retrieving comparison {comparison_id}: {str(e)}")
            return None

    async def update_comparison(
            self, comparison_id: UUID, comparison_data: ComparisonUpdate, user_id: Optional[UUID] = None
    ) -> Optional[EvaluationComparison]:
        """Update comparison by ID with optional user ownership check."""
        # Filter out None values
        update_data = {
            k: v for k, v in comparison_data.model_dump().items() if v is not None
        }

        if not update_data:
            return await self.comparison_repo.get_with_evaluations(comparison_id)

        # Check ownership if user_id is provided
        if user_id:
            comparison = await self.comparison_repo.get_user_owned(comparison_id, user_id)
            if not comparison:
                logger.warning(f"Unauthorized update attempt on comparison {comparison_id} by user {user_id}")
                raise AuthorizationException(
                    detail="You don't have permission to update this comparison"
                )

        try:
            comparison = await self.comparison_repo.update(comparison_id, update_data)
            if comparison:
                logger.info(f"Updated comparison {comparison_id}: {update_data}")
                return await self.comparison_repo.get_with_evaluations(comparison_id)
            else:
                logger.warning(f"Failed to update comparison {comparison_id}: not found")
                return None
        except Exception as e:
            logger.error(f"Error updating comparison {comparison_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update comparison: {str(e)}"
            )

    async def delete_comparison(self, comparison_id: UUID, user_id: Optional[UUID] = None) -> bool:
        """Delete comparison by ID with optional user ownership check."""
        try:
            comparison = await self.comparison_repo.get(comparison_id)
            if not comparison:
                logger.warning(f"Attempted to delete non-existent comparison {comparison_id}")
                return False

            # Check ownership if user_id is provided
            if user_id and comparison.created_by_id and comparison.created_by_id != user_id:
                logger.warning(f"Unauthorized deletion attempt on comparison {comparison_id} by user {user_id}")
                raise AuthorizationException(
                    detail="You don't have permission to delete this comparison"
                )

            success = await self.comparison_repo.delete(comparison_id)
            if success:
                logger.info(f"Deleted comparison {comparison_id}")
            return success

        except AuthorizationException:
            raise
        except Exception as e:
            logger.error(f"Error deleting comparison {comparison_id}: {str(e)}")
            return False

    async def get_comparison_metrics(
            self, comparison_id: UUID, user_id: Optional[UUID] = None
    ) -> List[MetricDifferenceResponse]:
        """Get detailed metric differences for a comparison."""
        # Get comparison with user verification if user_id provided
        if user_id:
            comparison = await self.comparison_repo.get_user_owned(comparison_id, user_id)
            if not comparison:
                logger.warning(f"Comparison with ID {comparison_id} not found or user {user_id} doesn't have access")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Comparison with ID {comparison_id} not found or you don't have access to it"
                )
        else:
            comparison = await self.comparison_repo.get(comparison_id)
            if not comparison:
                logger.warning(f"Comparison with ID {comparison_id} not found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Comparison with ID {comparison_id} not found"
                )

        # Check if comparison has results
        if not comparison.comparison_results or "metric_comparison" not in comparison.comparison_results:
            logger.warning(f"No metric comparison results found for comparison {comparison_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Comparison has not been calculated yet. Please run the comparison first."
            )

        # Extract and format metric differences
        metric_differences = []
        metric_comparison = comparison.comparison_results["metric_comparison"]

        for metric_name, data in metric_comparison.items():
            if "comparison" in data and data["comparison"].get("absolute_difference") is not None:
                metric_diff = MetricDifferenceResponse(
                    metric_name=metric_name,
                    evaluation_a_value=data["evaluation_a"]["average"],
                    evaluation_b_value=data["evaluation_b"]["average"],
                    absolute_difference=data["comparison"]["absolute_difference"],
                    percentage_change=data["comparison"]["percentage_change"],
                    is_improvement=data["comparison"]["is_improvement"],
                    p_value=data["comparison"].get("corrected_p_value") or data["comparison"].get("p_value"),
                    is_significant=data["comparison"].get("is_significant"),
                    weight=data["comparison"].get("weight", 1.0)
                )
                metric_differences.append(metric_diff)

        # Sort by absolute difference (largest absolute change first)
        metric_differences = sorted(
            metric_differences,
            key=lambda x: abs(x.absolute_difference),
            reverse=True
        )

        return metric_differences

    async def generate_comparison_report(
            self, comparison_id: UUID, format: str = "json", user_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Generate a downloadable report for the comparison."""
        # Get comparison with user verification if user_id provided
        if user_id:
            comparison = await self.comparison_repo.get_user_owned(comparison_id, user_id)
            if not comparison:
                logger.warning(f"Comparison with ID {comparison_id} not found or user {user_id} doesn't have access")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Comparison with ID {comparison_id} not found or you don't have access to it"
                )
        else:
            comparison = await self.comparison_repo.get(comparison_id)
            if not comparison:
                logger.warning(f"Comparison with ID {comparison_id} not found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Comparison with ID {comparison_id} not found"
                )

        # Check if comparison has results
        if not comparison.comparison_results:
            logger.warning(f"No comparison results found for comparison {comparison_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Comparison has not been calculated yet. Please run the comparison first."
            )

        # Get related evaluations
        evaluation_a = await self.evaluation_repo.get(comparison.evaluation_a_id)
        evaluation_b = await self.evaluation_repo.get(comparison.evaluation_b_id)

        if not evaluation_a or not evaluation_b:
            logger.error(f"One or both evaluations not found for comparison {comparison_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or both referenced evaluations not found"
            )

        # Generate narrative insights if not already present
        narrative_insights = comparison.narrative_insights
        if not narrative_insights and comparison.summary and comparison.comparison_results:
            narrative_insights = _generate_enhanced_insights(
                comparison.summary,
                comparison.comparison_results.get("compatibility_warnings", [])
            )

        # Format report data
        report_data = {
            "report_title": f"Comparison Report: {comparison.name}",
            "generated_at": datetime.now().isoformat(),
            "evaluations": {
                "evaluation_a": {
                    "id": str(evaluation_a.id),
                    "name": evaluation_a.name,
                    "method": evaluation_a.method.value if evaluation_a.method else None
                },
                "evaluation_b": {
                    "id": str(evaluation_b.id),
                    "name": evaluation_b.name,
                    "method": evaluation_b.method.value if evaluation_b.method else None
                }
            },
            "summary": comparison.summary,
            "narrative_insights": narrative_insights,
            "comparison_results": comparison.comparison_results,
            "compatibility_warnings": comparison.comparison_results.get("compatibility_warnings", [])
        }

        if format.lower() == "pdf":
            # Generate PDF
            try:
                pdf_buffer = self.pdf_generator.generate_comparison_pdf(report_data)
                return {
                    "format": "pdf",
                    "content": pdf_buffer.getvalue(),
                    "filename": f"comparison_report_{comparison_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    "content_type": "application/pdf"
                }
            except Exception as e:
                logger.error(f"Failed to generate PDF report: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to generate PDF report: {str(e)}"
                )
        else:
            # Return JSON format
            return {
                "format": "json",
                "content": report_data,
                "filename": f"comparison_report_{comparison_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "content_type": "application/json"
            }

    async def get_comparison_visualizations(
            self, comparison_id: UUID, visualization_type: str = "radar", user_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Generate visualization data for charts."""
        # Get comparison with user verification if user_id provided
        if user_id:
            comparison = await self.comparison_repo.get_user_owned(comparison_id, user_id)
            if not comparison:
                logger.warning(f"Comparison with ID {comparison_id} not found or user {user_id} doesn't have access")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Comparison with ID {comparison_id} not found or you don't have access to it"
                )
        else:
            comparison = await self.comparison_repo.get(comparison_id)
            if not comparison:
                logger.warning(f"Comparison with ID {comparison_id} not found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Comparison with ID {comparison_id} not found"
                )

        # Check if comparison has results
        if not comparison.comparison_results or "metric_comparison" not in comparison.comparison_results:
            logger.warning(f"No metric comparison results found for comparison {comparison_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Comparison has not been calculated yet. Please run the comparison first."
            )

        # Get related evaluations for labels
        evaluation_a = await self.evaluation_repo.get(comparison.evaluation_a_id)
        evaluation_b = await self.evaluation_repo.get(comparison.evaluation_b_id)

        if not evaluation_a or not evaluation_b:
            logger.error(f"One or both evaluations not found for comparison {comparison_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or both referenced evaluations not found"
            )

        # Get metric comparisons
        metric_comparison = comparison.comparison_results["metric_comparison"]

        # Generate visualization data based on type
        if visualization_type == "radar":
            return self._generate_radar_chart_data(metric_comparison, evaluation_a.name, evaluation_b.name)
        elif visualization_type == "bar":
            return self._generate_bar_chart_data(metric_comparison, evaluation_a.name, evaluation_b.name)
        elif visualization_type == "line":
            return self._generate_line_chart_data(metric_comparison, evaluation_a.name, evaluation_b.name)
        elif visualization_type == "significance":
            return self._generate_significance_chart_data(metric_comparison, evaluation_a.name, evaluation_b.name)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported visualization type: {visualization_type}"
            )

    @staticmethod
    def _generate_radar_chart_data(
            metric_comparison: Dict[str, Any], eval_a_name: str, eval_b_name: str
    ) -> Dict[str, Any]:
        """Generate data for radar chart visualization."""
        labels = []
        series = [
            {
                "name": eval_a_name,
                "data": []
            },
            {
                "name": eval_b_name,
                "data": []
            }
        ]

        # Track which metrics need to be inverted for display (lower is better)
        metric_is_inverted = []

        for metric_name, data in metric_comparison.items():
            if "evaluation_a" in data and "evaluation_b" in data:
                avg_a = data["evaluation_a"].get("average")
                avg_b = data["evaluation_b"].get("average")

                # Get if higher is better for this metric
                higher_is_better = data.get("config", {}).get("higher_is_better", True)

                if avg_a is not None and avg_b is not None:
                    labels.append(metric_name)

                    # For metrics where lower is better, we invert for display
                    # so that "better" is always further out on the radar
                    if not higher_is_better:
                        # Invert by calculating 1 - value (assuming values are normalized 0-1)
                        # If values might be outside 0-1, normalize first
                        max_val = max(avg_a, avg_b, 1.0)  # Ensure minimum scale of 1.0
                        series[0]["data"].append(max_val - avg_a)
                        series[1]["data"].append(max_val - avg_b)
                        metric_is_inverted.append(True)
                    else:
                        series[0]["data"].append(avg_a)
                        series[1]["data"].append(avg_b)
                        metric_is_inverted.append(False)

        return {
            "type": "radar",
            "labels": labels,
            "series": series,
            "is_inverted": metric_is_inverted  # Include inversion information
        }

    @staticmethod
    def _generate_bar_chart_data(
            metric_comparison: Dict[str, Any], eval_a_name: str, eval_b_name: str
    ) -> Dict[str, Any]:
        """Generate data for bar chart visualization."""
        categories = []
        series = [{
            "name": eval_a_name,
            "data": []
        }, {
            "name": eval_b_name,
            "data": []
        }, {
            "name": "Change",
            "data": [],
            "type": "line"
        }]

        # Track significance for styling
        significance = []
        higher_is_better = []

        for metric_name, data in metric_comparison.items():
            if "evaluation_a" in data and "evaluation_b" in data and "comparison" in data:
                avg_a = data["evaluation_a"].get("average")
                avg_b = data["evaluation_b"].get("average")

                if avg_a is not None and avg_b is not None:
                    categories.append(metric_name)
                    series[0]["data"].append(avg_a)
                    series[1]["data"].append(avg_b)

                    # Add percentage change
                    pct_change = data["comparison"].get("percentage_change", 0)
                    series[2]["data"].append(pct_change)

                    # Track if change is significant
                    significance.append(data["comparison"].get("is_significant", False))

                    # Track if higher is better
                    higher_is_better.append(data.get("config", {}).get("higher_is_better", True))

        return {
            "type": "bar",
            "categories": categories,
            "series": series,
            "is_significant": significance,
            "higher_is_better": higher_is_better
        }

    @staticmethod
    def _generate_line_chart_data(
            metric_comparison: Dict[str, Any], eval_a_name: str, eval_b_name: str
    ) -> Dict[str, Any]:
        """Generate data for line chart visualization."""
        # For line chart, we'll use the actual values rather than averages
        metrics = []

        for metric_name, data in metric_comparison.items():
            if "evaluation_a" in data and "evaluation_b" in data:
                if "values" in data["evaluation_a"] and "values" in data["evaluation_b"]:
                    values_a = data["evaluation_a"]["values"]
                    values_b = data["evaluation_b"]["values"]

                    if values_a and values_b:
                        # Get if higher is better
                        higher_is_better = data.get("config", {}).get("higher_is_better", True)

                        # Only include metrics with actual values
                        metrics.append({
                            "name": metric_name,
                            "higher_is_better": higher_is_better,
                            "is_significant": data["comparison"].get("is_significant", False),
                            "evaluation_a": {
                                "name": eval_a_name,
                                "values": values_a,
                                "min": data["evaluation_a"].get("min"),
                                "max": data["evaluation_a"].get("max"),
                                "q1": data["evaluation_a"].get("q1"),
                                "q3": data["evaluation_a"].get("q3"),
                                "median": data["evaluation_a"].get("median")
                            },
                            "evaluation_b": {
                                "name": eval_b_name,
                                "values": values_b,
                                "min": data["evaluation_b"].get("min"),
                                "max": data["evaluation_b"].get("max"),
                                "q1": data["evaluation_b"].get("q1"),
                                "q3": data["evaluation_b"].get("q3"),
                                "median": data["evaluation_b"].get("median")
                            }
                        })

        return {
            "type": "line",
            "metrics": metrics
        }

    @staticmethod
    def _generate_significance_chart_data(
            metric_comparison: Dict[str, Any], eval_a_name: str, eval_b_name: str
    ) -> Dict[str, Any]:
        """Generate data for significance visualization."""
        metrics = []

        for metric_name, data in metric_comparison.items():
            if "comparison" in data and data["comparison"].get("p_value") is not None:
                # Extract data
                p_value = data["comparison"].get("corrected_p_value") or data["comparison"]["p_value"]
                percentage_change = data["comparison"].get("percentage_change", 0)
                is_improvement = data["comparison"].get("is_improvement", False)
                is_significant = data["comparison"].get("is_significant", False)
                higher_is_better = data.get("config", {}).get("higher_is_better", True)
                effect_size = data["comparison"].get("effect_size")

                # Add to metrics list
                metrics.append({
                    "name": metric_name,
                    "p_value": p_value,
                    "percentage_change": percentage_change,
                    "is_improvement": is_improvement,
                    "is_significant": is_significant,
                    "higher_is_better": higher_is_better,
                    "effect_size": effect_size
                })

        # Sort by p-value
        metrics.sort(key=lambda x: x["p_value"])

        return {
            "type": "significance",
            "metrics": metrics,
            "evaluation_a_name": eval_a_name,
            "evaluation_b_name": eval_b_name
        }