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


# SIMPLIFIED UTILITY FUNCTIONS
def _get_config_value(config: Union[Dict, MetricConfig], key: str, default: Any = None) -> Any:
    """Simplified config value extraction."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _calculate_average_score(results: List[EvaluationResult]) -> Optional[float]:
    """Calculate average overall score from results."""
    if not results:
        return None
    valid_scores = [r.overall_score for r in results if r.overall_score is not None]
    return sum(valid_scores) / len(valid_scores) if valid_scores else None


def _assess_statistical_power(metric_comparison: Dict[str, Any], sample_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Assess the statistical power of the comparison."""
    sample_size = sample_stats.get("matched_count", 0)

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


def _check_evaluation_compatibility(evaluation_a: Evaluation, evaluation_b: Evaluation) -> Dict[str, List[str]]:
    """Check compatibility between two evaluations."""
    errors = []
    warnings = []

    # Check dataset compatibility
    if evaluation_a.dataset_id != evaluation_b.dataset_id:
        warnings.append(f"Different datasets: {evaluation_a.dataset_id} vs {evaluation_b.dataset_id}")

    # Check method compatibility
    method_a = evaluation_a.method.value if evaluation_a.method else "unknown"
    method_b = evaluation_b.method.value if evaluation_b.method else "unknown"

    if method_a != method_b:
        warnings.append(f"Different methods: {method_a} vs {method_b}. Normalization will be applied.")

    # Check metric compatibility
    if evaluation_a.metrics and evaluation_b.metrics:
        metrics_a = set(evaluation_a.metrics)
        metrics_b = set(evaluation_b.metrics)
        common_metrics = metrics_a & metrics_b

        if len(common_metrics) == 0:
            errors.append("No common metrics found between evaluations")
        elif metrics_a != metrics_b:
            only_in_a = metrics_a - metrics_b
            only_in_b = metrics_b - metrics_a
            if only_in_a:
                warnings.append(f"Metrics only in evaluation A: {', '.join(only_in_a)}")
            if only_in_b:
                warnings.append(f"Metrics only in evaluation B: {', '.join(only_in_b)}")

    # Check results
    if len(evaluation_a.results) == 0 or len(evaluation_b.results) == 0:
        errors.append(f"Missing results: {len(evaluation_a.results)} vs {len(evaluation_b.results)}")

    # Check status
    if evaluation_a.status.value != "completed":
        errors.append(f"Evaluation A not completed: {evaluation_a.status.value}")
    if evaluation_b.status.value != "completed":
        errors.append(f"Evaluation B not completed: {evaluation_b.status.value}")

    return {"errors": errors, "warnings": warnings}


class ComparisonCalculator:
    """Dedicated class for comparison calculations."""

    def __init__(self, evaluation_a: Evaluation, evaluation_b: Evaluation, metric_configs: Dict[str, Dict[str, Any]]):
        self.evaluation_a = evaluation_a
        self.evaluation_b = evaluation_b
        self.metric_configs = metric_configs

    async def calculate_metric_comparison(self) -> Dict[str, Any]:
        """Calculate metric-level differences."""
        # Get all metric scores for each evaluation
        metric_scores_a = self._extract_metric_scores(self.evaluation_a.results)
        metric_scores_b = self._extract_metric_scores(self.evaluation_b.results)

        metric_comparison = {}
        all_metrics = set(list(metric_scores_a.keys()) + list(metric_scores_b.keys()))

        # Collect p-values for correction
        p_values_for_correction = []
        metric_names_with_p_values = []

        for metric_name in all_metrics:
            if not metric_scores_a.get(metric_name) and not metric_scores_b.get(metric_name):
                continue

            config = self.metric_configs.get(metric_name, {
                "higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "unknown"
            })

            # Calculate statistics for both evaluations
            stats_a = self._calculate_metric_stats(metric_scores_a.get(metric_name, []))
            stats_b = self._calculate_metric_stats(metric_scores_b.get(metric_name, []))

            # Calculate comparison if both have data
            comparison_stats = {}
            if stats_a["values"] and stats_b["values"]:
                comparison_stats = self._calculate_metric_comparison_stats(
                    stats_a, stats_b, config, p_values_for_correction, metric_names_with_p_values, metric_name
                )

            metric_comparison[metric_name] = {
                "evaluation_a": stats_a,
                "evaluation_b": stats_b,
                "comparison": comparison_stats,
                "config": config
            }

        # Apply multiple comparison correction
        if p_values_for_correction:
            self._apply_p_value_correction(metric_comparison, p_values_for_correction, metric_names_with_p_values)

        return metric_comparison

    def _extract_metric_scores(self, results: List[EvaluationResult]) -> Dict[str, List[float]]:
        """Extract metric scores from results."""
        metric_scores = {}
        for result in results:
            for metric_score in result.metric_scores:
                metric_name = metric_score.name
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                metric_scores[metric_name].append(metric_score.value)
        return metric_scores

    def _calculate_metric_stats(self, values: List[float]) -> Dict[str, Any]:
        """Calculate statistics for a metric."""
        if not values:
            return {
                "average": None, "median": None, "std_dev": None,
                "min": None, "max": None, "q1": None, "q3": None,
                "sample_count": 0, "values": []
            }

        values_np = np.array(values)
        return {
            "average": float(np.mean(values_np)),
            "median": float(np.median(values_np)),
            "std_dev": float(np.std(values_np)) if len(values) > 1 else None,
            "min": float(np.min(values_np)),
            "max": float(np.max(values_np)),
            "q1": float(np.percentile(values_np, 25)) if len(values) > 1 else None,
            "q3": float(np.percentile(values_np, 75)) if len(values) > 1 else None,
            "sample_count": len(values),
            "values": values
        }

    def _calculate_metric_comparison_stats(
            self, stats_a: Dict, stats_b: Dict, config: Dict,
            p_values: List, metric_names: List, metric_name: str
    ) -> Dict[str, Any]:
        """Calculate comparison statistics between two metrics."""
        avg_a = stats_a["average"]
        avg_b = stats_b["average"]
        higher_is_better = config.get("higher_is_better", True)

        # Calculate differences
        absolute_diff = StatisticsUtils.safe_absolute_difference(avg_b, avg_a)
        percentage_diff = StatisticsUtils.safe_percentage_change(avg_b, avg_a)

        # Determine improvement
        is_improvement = None
        if absolute_diff is not None:
            is_improvement = (absolute_diff > 0) if higher_is_better else (absolute_diff < 0)

        # Statistical tests
        p_value = None
        effect_size = None
        if len(stats_a["values"]) > 1 and len(stats_b["values"]) > 1:
            try:
                _, p_value = stats.ttest_ind(stats_a["values"], stats_b["values"], equal_var=False)
                p_values.append(p_value)
                metric_names.append(metric_name)
                effect_size = StatisticsUtils.calculate_effect_size(stats_a["values"], stats_b["values"])
            except Exception as e:
                logger.warning(f"Statistical test failed for {metric_name}: {e}")

        return {
            "absolute_difference": absolute_diff,
            "percentage_change": percentage_diff,
            "is_improvement": is_improvement,
            "p_value": p_value,
            "is_significant": None,  # Updated after correction
            "effect_size": effect_size,
            "weight": config.get("weight", 1.0)
        }

    def _apply_p_value_correction(
            self, metric_comparison: Dict, p_values: List, metric_names: List
    ) -> None:
        """Apply multiple comparison correction."""
        corrected_p_values = StatisticsUtils.apply_multiple_comparison_correction(p_values, method='bonferroni')

        for i, metric_name in enumerate(metric_names):
            if metric_name in metric_comparison and "comparison" in metric_comparison[metric_name]:
                corrected_p = corrected_p_values[i]
                metric_comparison[metric_name]["comparison"]["corrected_p_value"] = corrected_p
                metric_comparison[metric_name]["comparison"][
                    "is_significant"] = corrected_p < 0.05 if corrected_p is not None else None

    async def calculate_sample_comparison(self) -> Dict[str, Any]:
        """Calculate sample-level differences."""
        # Match results by dataset_sample_id
        results_a_by_sample = {r.dataset_sample_id: r for r in self.evaluation_a.results if r.dataset_sample_id}
        results_b_by_sample = {r.dataset_sample_id: r for r in self.evaluation_b.results if r.dataset_sample_id}

        all_sample_ids = set(list(results_a_by_sample.keys()) + list(results_b_by_sample.keys()))
        if not all_sample_ids:
            return self._empty_sample_comparison()

        matched_results = {}
        sample_improvements = 0
        sample_regressions = 0

        for sample_id in all_sample_ids:
            result_a = results_a_by_sample.get(sample_id)
            result_b = results_b_by_sample.get(sample_id)

            if result_a and result_b and result_a.overall_score is not None and result_b.overall_score is not None:
                comparison_data = self._compare_sample_results(result_a, result_b)
                matched_results[sample_id] = comparison_data

                if comparison_data["comparison"]["is_improvement"] is True:
                    sample_improvements += 1
                elif comparison_data["comparison"]["is_improvement"] is False:
                    sample_regressions += 1
            else:
                matched_results[sample_id] = self._handle_missing_sample(result_a, result_b)

        return self._build_sample_comparison_results(matched_results, sample_improvements, sample_regressions,
                                                     all_sample_ids)

    def _empty_sample_comparison(self) -> Dict[str, Any]:
        """Return empty sample comparison results."""
        return {
            "matched_results": {},
            "stats": {
                "matched_count": 0, "only_in_evaluation_a": 0, "only_in_evaluation_b": 0,
                "total_samples": 0, "improved_samples": 0, "regressed_samples": 0,
                "improvement_rate": 0, "significant_improvements": 0, "significant_regressions": 0
            }
        }

    def _compare_sample_results(self, result_a: EvaluationResult, result_b: EvaluationResult) -> Dict[str, Any]:
        """Compare two sample results."""
        score_a = result_a.overall_score
        score_b = result_b.overall_score

        absolute_diff = StatisticsUtils.safe_absolute_difference(score_b, score_a)
        percentage_diff = StatisticsUtils.safe_percentage_change(score_b, score_a)
        is_improvement = absolute_diff > 0 if absolute_diff is not None else None

        return {
            "evaluation_a": {"overall_score": score_a, "result_id": str(result_a.id)},
            "evaluation_b": {"overall_score": score_b, "result_id": str(result_b.id)},
            "comparison": {
                "absolute_difference": absolute_diff,
                "percentage_change": percentage_diff,
                "is_improvement": is_improvement
            }
        }

    def _handle_missing_sample(self, result_a: Optional[EvaluationResult], result_b: Optional[EvaluationResult]) -> \
            Dict[str, Any]:
        """Handle missing sample data."""
        return {
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

    def _build_sample_comparison_results(
            self, matched_results: Dict, improvements: int, regressions: int, all_sample_ids: set
    ) -> Dict[str, Any]:
        """Build the final sample comparison results."""
        matched_count = len([s for s in matched_results.values()
                             if
                             s.get("evaluation_a", {}).get("result_id") and s.get("evaluation_b", {}).get("result_id")])

        only_in_a = len([s for s in matched_results.values()
                         if
                         s.get("evaluation_a", {}).get("result_id") and not s.get("evaluation_b", {}).get("result_id")])

        only_in_b = len([s for s in matched_results.values()
                         if
                         not s.get("evaluation_a", {}).get("result_id") and s.get("evaluation_b", {}).get("result_id")])

        improvement_rate = (improvements / matched_count) * 100 if matched_count > 0 else 0

        return {
            "matched_results": matched_results,
            "stats": {
                "matched_count": matched_count,
                "only_in_evaluation_a": only_in_a,
                "only_in_evaluation_b": only_in_b,
                "total_samples": len(all_sample_ids),
                "improved_samples": improvements,
                "regressed_samples": regressions,
                "improvement_rate": improvement_rate,
                "significant_improvements": 0,  # Could be enhanced
                "significant_regressions": 0,  # Could be enhanced
                "significant_improvement_rate": 0
            }
        }

    def calculate_overall_comparison(self, metric_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall comparison statistics."""
        # Calculate overall scores
        overall_score_a = _calculate_average_score(self.evaluation_a.results)
        overall_score_b = _calculate_average_score(self.evaluation_b.results)

        overall_diff = StatisticsUtils.safe_absolute_difference(overall_score_b, overall_score_a)
        overall_pct = StatisticsUtils.safe_percentage_change(overall_score_b, overall_score_a)
        is_improvement = overall_diff > 0 if overall_diff is not None else None

        # Analyze metric improvements
        metric_stats = self._analyze_metric_improvements(metric_comparison)

        return {
            "overall_scores": {
                "evaluation_a": overall_score_a,
                "evaluation_b": overall_score_b,
                "absolute_difference": overall_diff,
                "percentage_change": overall_pct,
                "is_improvement": is_improvement
            },
            "metric_stats": metric_stats
        }

    def _analyze_metric_improvements(self, metric_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metric-level improvements and regressions."""
        metric_improvements = 0
        metric_regressions = 0
        significant_improvements = 0
        significant_regressions = 0
        weighted_sum = 0
        total_weight = 0
        normalized_scores = []

        for metric_data in metric_comparison.values():
            if "comparison" not in metric_data or metric_data["comparison"].get("is_improvement") is None:
                continue

            weight = metric_data["comparison"].get("weight", 1.0)
            is_improvement = metric_data["comparison"]["is_improvement"]
            percentage_change = metric_data["comparison"].get("percentage_change", 0)

            # Count improvements/regressions
            if is_improvement:
                metric_improvements += 1
            else:
                metric_regressions += 1

            # Count significant changes
            if metric_data["comparison"].get("is_significant") is True:
                if is_improvement:
                    significant_improvements += 1
                else:
                    significant_regressions += 1

            # Calculate weighted score
            if percentage_change is not None:
                normalized_change = max(-1.0, min(1.0, percentage_change / 100.0))
                if not is_improvement:
                    normalized_change = -abs(normalized_change)
                else:
                    normalized_change = abs(normalized_change)

                weighted_sum += normalized_change * weight
                total_weight += weight
                normalized_scores.append(normalized_change)

        # Calculate rates and consistency
        total_metrics = len(metric_comparison)
        significance_rate = (
                (significant_improvements + significant_regressions) / total_metrics * 100) if total_metrics else 0
        improvement_rate = (metric_improvements / total_metrics * 100) if total_metrics else 0
        weighted_score = weighted_sum / total_weight if total_weight > 0 else None
        consistency_score = max(0, 1 - np.var(normalized_scores)) if normalized_scores else None

        return {
            "total_metrics": total_metrics,
            "improved_metrics": metric_improvements,
            "regressed_metrics": metric_regressions,
            "significant_improvements": significant_improvements,
            "significant_regressions": significant_regressions,
            "significance_rate": significance_rate,
            "metric_improvement_rate": improvement_rate,
            "weighted_improvement_score": weighted_score,
            "consistency_score": consistency_score
        }


class InsightGenerator:
    """Generate natural language insights from comparison data."""

    @staticmethod
    def generate_insights(summary: Dict, compatibility_warnings: List[str] = None) -> str:
        """Generate enhanced natural language insights."""
        insights = []

        # Compatibility warnings
        if compatibility_warnings:
            insights.extend([
                "⚠️ **Compatibility Warnings:**",
                *[f"- {warning}" for warning in compatibility_warnings],
                ""
            ])

        # Overall assessment
        overall_result = summary.get("overall_result", "inconclusive")
        percentage_change = summary.get("percentage_change", 0)
        eval_a_name = summary.get("evaluation_a_name", "Evaluation A")
        eval_b_name = summary.get("evaluation_b_name", "Evaluation B")

        # Cross-method comparison note
        if summary.get("cross_method_comparison", False):
            method_a = summary.get("evaluation_a_method", "unknown")
            method_b = summary.get("evaluation_b_method", "unknown")
            insights.extend([
                f"🔄 **Cross-Method Comparison:** Comparing {method_a} vs {method_b} evaluations. "
                f"Metric normalization has been applied for fair comparison.",
                ""
            ])

            # Main assessment
            if overall_result == "improved":
                # Safe handling of None percentage_change
                if percentage_change is not None:
                    insights.append(f"📈 **Overall Assessment:** {eval_b_name} shows a "
                                    f"{percentage_change:.1f}% improvement over {eval_a_name}.")
                else:
                    insights.append(f"📈 **Overall Assessment:** {eval_b_name} shows improvement over {eval_a_name}.")
            elif overall_result == "regressed":
                # Safe handling of None percentage_change
                if percentage_change is not None:
                    insights.append(f"📉 **Overall Assessment:** {eval_b_name} shows a "
                                    f"{abs(percentage_change):.1f}% regression compared to {eval_a_name}.")
                else:
                    insights.append(
                        f"📉 **Overall Assessment:** {eval_b_name} shows regression compared to {eval_a_name}.")
            else:
                insights.append(f"⚖️ **Overall Assessment:** The comparison between {eval_a_name} and "
                                f"{eval_b_name} is inconclusive due to insufficient data.")

        # Sample analysis
        if summary.get("matched_samples", 0) > 0:
            InsightGenerator._add_sample_analysis(insights, summary)

        # Statistical power
        if summary.get("statistical_power"):
            power_info = summary["statistical_power"]
            insights.append(f"\n**Statistical Power:** {power_info['power_category'].title()} power "
                            f"(sample size: {power_info['sample_size']})")

        # Top changes
        InsightGenerator._add_top_changes(insights, summary)

        # Significance analysis
        InsightGenerator._add_significance_analysis(insights, summary)

        # Consistency and conclusion
        InsightGenerator._add_conclusion(insights, summary, overall_result)

        return "\n".join(insights)

    @staticmethod
    def _add_sample_analysis(insights: List[str], summary: Dict) -> None:
        """Add sample analysis section."""
        total_compared = summary.get("matched_samples", 0)
        improved_samples = summary.get("improved_samples", 0) or 0  # Handle None
        regressed_samples = summary.get("regressed_samples", 0) or 0  # Handle None

        if total_compared > 0:
            imp_rate = (improved_samples / total_compared) * 100
            insights.append(f"\n**Sample Analysis:** Of {total_compared} samples analyzed, "
                            f"{improved_samples} ({imp_rate:.1f}%) improved and "
                            f"{regressed_samples} ({100 - imp_rate:.1f}%) regressed.")

    @staticmethod
    def _add_top_changes(insights: List[str], summary: Dict) -> None:
        """Add top improvements and regressions."""
        if summary.get("top_improvements"):
            insights.append("\n**Top Improvements:**")
            for m in summary["top_improvements"][:3]:
                sig_marker = "* " if m.get("is_significant", False) else ""
                effect_info = f" (Effect: {m['effect_magnitude']})" if m.get("effect_magnitude") else ""
                # Safe handling of None percentage_change
                pct_change = m.get('percentage_change')
                if pct_change is not None:
                    insights.append(f"- {sig_marker}{m['metric_name']}: +{pct_change:.1f}%{effect_info}")
                else:
                    insights.append(f"- {sig_marker}{m['metric_name']}: improved{effect_info}")

    @staticmethod
    def _add_significance_analysis(insights: List[str], summary: Dict) -> None:
        """Add statistical significance analysis."""
        total_significant = summary.get("significant_improvements", 0) + summary.get("significant_regressions", 0)
        if total_significant > 0:
            insights.append(f"\n**Statistical Significance:** {total_significant} of {summary.get('total_metrics', 0)} "
                            f"metrics show statistically significant changes (marked with *).")

    @staticmethod
    def _add_conclusion(insights: List[str], summary: Dict, overall_result: str) -> None:
        """Add conclusion section."""
        if overall_result == "improved":
            # Safe handling of None consistency_score
            consistency_score = summary.get("consistency_score")
            if consistency_score is not None and consistency_score > 0.7:
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


class ComparisonService:
    """Streamlined service for handling comparison operations."""

    def __init__(self, db_session: AsyncSession):
        """Initialize the comparison service."""
        self.db_session = db_session
        self.comparison_repo = ComparisonRepository(db_session)
        self.evaluation_repo = EvaluationRepository(db_session)
        self.result_repo = BaseRepository(EvaluationResult, db_session)
        self.metric_repo = BaseRepository(MetricScore, db_session)
        self.pdf_generator = PDFReportGenerator()

    async def create_comparison(self, comparison_data: ComparisonCreate) -> EvaluationComparison:
        """Create a new comparison with user attribution."""
        # Verify evaluations exist
        evaluation_a = await self.evaluation_repo.get_evaluation_with_details(comparison_data.evaluation_a_id)
        evaluation_b = await self.evaluation_repo.get_evaluation_with_details(comparison_data.evaluation_b_id)

        if not evaluation_a:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Evaluation A with ID {comparison_data.evaluation_a_id} not found")
        if not evaluation_b:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Evaluation B with ID {comparison_data.evaluation_b_id} not found")

        # Check for existing comparison
        existing_comparison = await self.comparison_repo.get_by_evaluations(
            comparison_data.evaluation_a_id, comparison_data.evaluation_b_id, comparison_data.created_by_id
        )
        if existing_comparison:
            return existing_comparison

        # Compatibility check
        compatibility_issues = _check_evaluation_compatibility(evaluation_a, evaluation_b)
        if compatibility_issues.get("errors"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Evaluations incompatible: {'; '.join(compatibility_issues['errors'])}"
            )

        # Use defaults if no configs provided
        if not comparison_data.metric_configs:
            comparison_data.metric_configs = DEFAULT_METRIC_CONFIGS

        # Create comparison
        comparison_dict = comparison_data.model_dump()
        comparison_dict["status"] = "pending"

        try:
            comparison = await self.comparison_repo.create(comparison_dict)
            logger.info(f"Created comparison {comparison.id}")
            return comparison
        except Exception as e:
            logger.error(f"Failed to create comparison: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Failed to create comparison: {str(e)}")

    async def run_comparison_calculation(self, comparison_id: UUID,
                                         user_id: Optional[UUID] = None) -> EvaluationComparison:
        """Perform the comparison calculation."""
        # Get comparison with user verification
        comparison = await self._get_comparison_with_auth(comparison_id, user_id)

        try:
            # Get evaluations
            evaluation_a = await self.evaluation_repo.get_evaluation_with_details(comparison.evaluation_a_id)
            evaluation_b = await self.evaluation_repo.get_evaluation_with_details(comparison.evaluation_b_id)

            if not evaluation_a or not evaluation_b:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                    detail="One or both referenced evaluations not found")

            # Compatibility check
            compatibility_check = _check_evaluation_compatibility(evaluation_a, evaluation_b)
            if compatibility_check.get("errors"):
                error_msg = f"Evaluations incompatible: {'; '.join(compatibility_check['errors'])}"
                # Use a new transaction for error update
                try:
                    await self.comparison_repo.update(comparison_id, {"status": "failed", "error": error_msg})
                except Exception as update_error:
                    logger.error(f"Failed to update comparison status after compatibility error: {update_error}")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)

            # Update status to running
            try:
                await self.comparison_repo.update(comparison_id, {"status": "running"})
            except Exception as update_error:
                logger.error(f"Failed to update comparison status to running: {update_error}")
                # Continue anyway as this is not critical

            # Calculate comparison
            comparison_results = await self._calculate_comparison(evaluation_a, evaluation_b, comparison.metric_configs)
            comparison_results["compatibility_warnings"] = compatibility_check.get("warnings", [])

            # Generate insights
            narrative_insights = InsightGenerator.generate_insights(
                comparison_results["summary"], compatibility_check.get("warnings", [])
            )

            # Ensure all data is JSON serializable before database update
            serializable_comparison_results = self._ensure_json_serializable(comparison_results["detailed_results"])
            serializable_summary = self._ensure_json_serializable(comparison_results["summary"])

            # Validate serialized data before saving
            try:
                import json
                json.dumps(serializable_comparison_results)
                json.dumps(serializable_summary)
            except (TypeError, ValueError) as json_error:
                logger.error(f"Data still not JSON serializable after conversion: {json_error}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to serialize comparison results for storage"
                )

            # Update with results
            update_data = {
                "status": "completed",
                "comparison_results": serializable_comparison_results,
                "summary": serializable_summary,
                "narrative_insights": narrative_insights
            }

            updated_comparison = await self.comparison_repo.update(comparison_id, update_data)
            logger.info(f"Completed comparison calculation for {comparison_id}")
            return updated_comparison

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Error calculating comparison {comparison_id}: {str(e)}", exc_info=True)

            # Try to update status to failed, but handle transaction errors
            try:
                # Use a simplified error message to avoid serialization issues
                error_message = f"Calculation failed: {type(e).__name__}"
                if hasattr(e, 'message'):
                    error_message += f" - {e.message[:200]}"  # Truncate long messages
                elif str(e):
                    error_message += f" - {str(e)[:200]}"  # Truncate long error strings

                await self.comparison_repo.update(comparison_id, {
                    "status": "failed",
                    "error": error_message
                })
            except Exception as update_error:
                logger.error(f"Failed to update comparison status after calculation error: {update_error}")
                # If we can't update the database, at least log the original error
                logger.error(f"Original calculation error for {comparison_id}: {str(e)}")

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error calculating comparison: {type(e).__name__}"
            )

    async def _calculate_comparison(
            self, evaluation_a: Evaluation, evaluation_b: Evaluation,
            metric_configs: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Calculate comparison using the dedicated calculator."""
        # Prepare metric configurations
        final_configs = self._prepare_metric_configs(evaluation_a, evaluation_b, metric_configs)

        # Initialize calculator
        calculator = ComparisonCalculator(evaluation_a, evaluation_b, final_configs)

        # Perform calculations
        metric_comparison = await calculator.calculate_metric_comparison()
        sample_comparison = await calculator.calculate_sample_comparison()
        overall_comparison = calculator.calculate_overall_comparison(metric_comparison)

        # Generate statistical summary
        statistical_summary = self._generate_statistical_summary(metric_comparison, sample_comparison,
                                                                 overall_comparison)

        # Build detailed results
        detailed_results = {
            "metric_comparison": metric_comparison,
            "sample_comparison": sample_comparison,
            "overall_comparison": overall_comparison,
            "statistical_summary": statistical_summary
        }

        # Generate summary
        summary = self._generate_summary(evaluation_a, evaluation_b, metric_comparison,
                                         sample_comparison, overall_comparison, statistical_summary)

        return {"detailed_results": detailed_results, "summary": summary}

    def _prepare_metric_configs(
            self, evaluation_a: Evaluation, evaluation_b: Evaluation, metric_configs: Optional[Dict] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Prepare metric configurations."""
        final_configs = {}

        # Get all metrics
        all_metrics = set()
        if evaluation_a.metrics:
            all_metrics.update(evaluation_a.metrics)
        if evaluation_b.metrics:
            all_metrics.update(evaluation_b.metrics)

        for metric_name in all_metrics:
            # Start with defaults
            config = DEFAULT_METRIC_CONFIGS.get(metric_name, {
                "higher_is_better": True, "weight": 1.0, "scale": (0, 1), "method": "unknown"
            }).copy()

            # Override with user configs
            if metric_configs and metric_name in metric_configs:
                user_config = metric_configs[metric_name]
                higher_is_better = _get_config_value(user_config, "higher_is_better")
                weight = _get_config_value(user_config, "weight")

                if higher_is_better is not None:
                    config["higher_is_better"] = higher_is_better
                if weight is not None:
                    config["weight"] = weight

            final_configs[metric_name] = config

        return final_configs

    def _generate_statistical_summary(
            self, metric_comparison: Dict, sample_comparison: Dict, overall_comparison: Dict
    ) -> Dict[str, Any]:
        """Generate statistical summary."""
        # Collect effect sizes
        effect_sizes = []
        for metric_name, data in metric_comparison.items():
            if "comparison" in data and data["comparison"].get("effect_size") is not None:
                effect_size = data["comparison"]["effect_size"]
                effect_sizes.append({
                    "metric": metric_name,
                    "effect_size": effect_size,
                    "magnitude": StatisticsUtils.interpret_effect_size(effect_size)
                })

        return {
            "sample_statistics": sample_comparison.get("stats", {}),
            "metric_statistics": overall_comparison.get("metric_stats", {}),
            "overall_statistics": overall_comparison.get("overall_scores", {}),
            "effect_sizes": {
                "metric_effect_sizes": effect_sizes,
                "average_effect_size": np.mean([es["effect_size"] for es in effect_sizes]) if effect_sizes else None
            }
        }

    def _generate_summary(
            self, evaluation_a: Evaluation, evaluation_b: Evaluation, metric_comparison: Dict,
            sample_comparison: Dict, overall_comparison: Dict, statistical_summary: Dict
    ) -> Dict[str, Any]:
        """Generate comparison summary."""
        overall_stats = overall_comparison["overall_scores"]
        metric_stats = overall_comparison["metric_stats"]
        sample_stats = sample_comparison["stats"]

        is_improvement = overall_stats.get("is_improvement", False)
        percentage_change = overall_stats.get("percentage_change", 0)

        # Identify top changes
        top_improvements, top_regressions = self._identify_top_changes(metric_comparison)

        return {
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

    def _identify_top_changes(self, metric_comparison: Dict[str, Any]) -> tuple:
        """Identify top improvements and regressions."""
        improvements = []
        regressions = []

        for metric_name, data in metric_comparison.items():
            if "comparison" not in data or data["comparison"].get("absolute_difference") is None:
                continue

            diff_data = {
                "metric_name": metric_name,
                "absolute_difference": data["comparison"]["absolute_difference"],
                "percentage_change": data["comparison"]["percentage_change"],
                "is_significant": data["comparison"].get("is_significant", False),
                "effect_size": data["comparison"].get("effect_size"),
                "effect_magnitude": StatisticsUtils.interpret_effect_size(data["comparison"]["effect_size"])
                if data["comparison"].get("effect_size") else None
            }

            if data["comparison"]["is_improvement"]:
                improvements.append(diff_data)
            else:
                regressions.append(diff_data)

        # Sort by effect size or absolute difference
        def sort_key(x):
            return abs(x["effect_size"]) if x["effect_size"] is not None else abs(x["absolute_difference"]) if x[
                                                                                                                   "absolute_difference"] is not None else 0

        return sorted(improvements, key=sort_key, reverse=True)[:3], sorted(regressions, key=sort_key, reverse=True)[:3]

    async def _get_comparison_with_auth(self, comparison_id: UUID,
                                        user_id: Optional[UUID] = None) -> EvaluationComparison:
        """Get comparison with user authorization check."""
        if user_id:
            comparison = await self.comparison_repo.get_user_owned(comparison_id, user_id)
            if not comparison:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                    detail=f"Comparison with ID {comparison_id} not found or access denied")
        else:
            comparison = await self.comparison_repo.get(comparison_id)
            if not comparison:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                    detail=f"Comparison with ID {comparison_id} not found")
        return comparison

    # SIMPLIFIED CRUD OPERATIONS
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
        """Update comparison by ID."""
        update_data = {k: v for k, v in comparison_data.model_dump().items() if v is not None}
        if not update_data:
            return await self.comparison_repo.get_with_evaluations(comparison_id)

        if user_id:
            comparison = await self.comparison_repo.get_user_owned(comparison_id, user_id)
            if not comparison:
                raise AuthorizationException(detail="You don't have permission to update this comparison")

        try:
            comparison = await self.comparison_repo.update(comparison_id, update_data)
            return await self.comparison_repo.get_with_evaluations(comparison_id) if comparison else None
        except Exception as e:
            logger.error(f"Error updating comparison {comparison_id}: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Failed to update comparison: {str(e)}")

    async def delete_comparison(self, comparison_id: UUID, user_id: Optional[UUID] = None) -> bool:
        """Delete comparison by ID."""
        try:
            comparison = await self.comparison_repo.get(comparison_id)
            if not comparison:
                return False

            if user_id and comparison.created_by_id and comparison.created_by_id != user_id:
                raise AuthorizationException(detail="You don't have permission to delete this comparison")

            return await self.comparison_repo.delete(comparison_id)
        except AuthorizationException:
            raise
        except Exception as e:
            logger.error(f"Error deleting comparison {comparison_id}: {str(e)}")
            return False

    async def get_comparison_metrics(
            self, comparison_id: UUID, user_id: Optional[UUID] = None
    ) -> List[MetricDifferenceResponse]:
        """Get detailed metric differences."""
        comparison = await self._get_comparison_with_auth(comparison_id, user_id)

        if not comparison.comparison_results or "metric_comparison" not in comparison.comparison_results:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Comparison has not been calculated yet.")

        metric_differences = []
        for metric_name, data in comparison.comparison_results["metric_comparison"].items():
            if "comparison" in data and data["comparison"].get("absolute_difference") is not None:
                metric_differences.append(MetricDifferenceResponse(
                    metric_name=metric_name,
                    evaluation_a_value=data["evaluation_a"]["average"],
                    evaluation_b_value=data["evaluation_b"]["average"],
                    absolute_difference=data["comparison"]["absolute_difference"],
                    percentage_change=data["comparison"]["percentage_change"],
                    is_improvement=data["comparison"]["is_improvement"],
                    p_value=data["comparison"].get("corrected_p_value") or data["comparison"].get("p_value"),
                    is_significant=data["comparison"].get("is_significant"),
                    weight=data["comparison"].get("weight", 1.0)
                ))

        return sorted(metric_differences,
                      key=lambda x: abs(x.absolute_difference) if x.absolute_difference is not None else 0,
                      reverse=True)

        # Add these methods to your ComparisonService class:

    async def generate_comparison_report(
            self, comparison_id: UUID, format: str = "json", user_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Generate a downloadable report for the comparison."""
        comparison = await self._get_comparison_with_auth(comparison_id, user_id)

        if not comparison.comparison_results:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Comparison has not been calculated yet."
            )

        # Get related evaluations
        evaluation_a = await self.evaluation_repo.get(comparison.evaluation_a_id)
        evaluation_b = await self.evaluation_repo.get(comparison.evaluation_b_id)

        if not evaluation_a or not evaluation_b:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or both referenced evaluations not found"
            )

        # Generate narrative insights if not present
        narrative_insights = comparison.narrative_insights
        if not narrative_insights and comparison.summary and comparison.comparison_results:
            narrative_insights = InsightGenerator.generate_insights(
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
        comparison = await self._get_comparison_with_auth(comparison_id, user_id)

        if not comparison.comparison_results or "metric_comparison" not in comparison.comparison_results:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Comparison has not been calculated yet."
            )

        # Get related evaluations for labels
        evaluation_a = await self.evaluation_repo.get(comparison.evaluation_a_id)
        evaluation_b = await self.evaluation_repo.get(comparison.evaluation_b_id)

        if not evaluation_a or not evaluation_b:
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
        series = [{"name": eval_a_name, "data": []}, {"name": eval_b_name, "data": []}]
        metric_is_inverted = []

        for metric_name, data in metric_comparison.items():
            if "evaluation_a" in data and "evaluation_b" in data:
                avg_a = data["evaluation_a"].get("average")
                avg_b = data["evaluation_b"].get("average")
                higher_is_better = data.get("config", {}).get("higher_is_better", True)

                if avg_a is not None and avg_b is not None:
                    labels.append(metric_name)

                    if not higher_is_better:
                        max_val = max(avg_a, avg_b, 1.0)
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
            "is_inverted": metric_is_inverted
        }

    @staticmethod
    def _generate_bar_chart_data(
            metric_comparison: Dict[str, Any], eval_a_name: str, eval_b_name: str
    ) -> Dict[str, Any]:
        """Generate data for bar chart visualization."""
        categories = []
        series = [
            {"name": eval_a_name, "data": []},
            {"name": eval_b_name, "data": []},
            {"name": "Change", "data": [], "type": "line"}
        ]
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

                    pct_change = data["comparison"].get("percentage_change", 0)
                    series[2]["data"].append(pct_change if pct_change is not None else 0)

                    significance.append(data["comparison"].get("is_significant", False))
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
        metrics = []

        for metric_name, data in metric_comparison.items():
            if "evaluation_a" in data and "evaluation_b" in data:
                if "values" in data["evaluation_a"] and "values" in data["evaluation_b"]:
                    values_a = data["evaluation_a"]["values"]
                    values_b = data["evaluation_b"]["values"]

                    if values_a and values_b:
                        higher_is_better = data.get("config", {}).get("higher_is_better", True)

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

        return {"type": "line", "metrics": metrics}

    @staticmethod
    def _generate_significance_chart_data(
            metric_comparison: Dict[str, Any], eval_a_name: str, eval_b_name: str
    ) -> Dict[str, Any]:
        """Generate data for significance visualization."""
        metrics = []

        for metric_name, data in metric_comparison.items():
            if "comparison" in data and data["comparison"].get("p_value") is not None:
                p_value = data["comparison"].get("corrected_p_value") or data["comparison"]["p_value"]
                percentage_change = data["comparison"].get("percentage_change", 0)
                is_improvement = data["comparison"].get("is_improvement", False)
                is_significant = data["comparison"].get("is_significant", False)
                higher_is_better = data.get("config", {}).get("higher_is_better", True)
                effect_size = data["comparison"].get("effect_size")

                metrics.append({
                    "name": metric_name,
                    "p_value": p_value,
                    "percentage_change": percentage_change,
                    "is_improvement": is_improvement,
                    "is_significant": is_significant,
                    "higher_is_better": higher_is_better,
                    "effect_size": effect_size
                })

        metrics.sort(key=lambda x: x["p_value"] if x["p_value"] is not None else float('inf'))

        return {
            "type": "significance",
            "metrics": metrics,
            "evaluation_a_name": eval_a_name,
            "evaluation_b_name": eval_b_name
        }

    def _ensure_json_serializable(self, data: Any) -> Any:
        """
        Recursively ensure all data is JSON serializable.
        Converts numpy types, NaN, infinity, and other non-serializable types to Python native types.
        """
        import json
        import numpy as np
        import math

        if isinstance(data, dict):
            return {key: self._ensure_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._ensure_json_serializable(item) for item in data]
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(data)
        elif isinstance(data, (np.float64, np.float16, np.float32, np.float64)):
            # Handle NaN and infinity values
            if np.isnan(data):
                return None  # Convert NaN to null
            elif np.isinf(data):
                return None  # Convert infinity to null
            else:
                return float(data)
        elif isinstance(data, (float, np.floating)):
            # Handle Python float NaN and infinity
            if math.isnan(data):
                return None
            elif math.isinf(data):
                return None
            else:
                return float(data)
        elif isinstance(data, np.ndarray):
            # Convert array and handle NaN/inf values
            converted_list = []
            for item in data.tolist():
                converted_list.append(self._ensure_json_serializable(item))
            return converted_list
        elif data is None or isinstance(data, (bool, int, str)):
            return data
        else:
            # Try to convert to string as fallback
            try:
                # First check if it's already JSON serializable
                json.dumps(data)
                return data
            except (TypeError, ValueError):
                # Convert to string as last resort
                str_value = str(data)
                # Handle string representations of NaN/inf
                if str_value.lower() in ['nan', 'inf', '-inf', 'infinity', '-infinity']:
                    return None
                return str_value
