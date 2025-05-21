import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

import numpy as np
from fastapi import HTTPException, status
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

logger = logging.getLogger(__name__)

# Default metric configurations with directions
DEFAULT_METRIC_CONFIGS = {
    "faithfulness": {"higher_is_better": True, "weight": 1.0},
    "response_relevancy": {"higher_is_better": True, "weight": 1.0},
    "context_precision": {"higher_is_better": True, "weight": 1.0},
    "context_recall": {"higher_is_better": True, "weight": 1.0},
    "context_entity_recall": {"higher_is_better": True, "weight": 1.0},
    "noise_sensitivity": {"higher_is_better": False, "weight": 1.0},  # Lower is better for noise sensitivity
    "answer_correctness": {"higher_is_better": True, "weight": 1.0},
    "answer_similarity": {"higher_is_better": True, "weight": 1.0},
    "answer_relevancy": {"higher_is_better": True, "weight": 1.0},
    "factual_correctness": {"higher_is_better": True, "weight": 1.0}
}


class ComparisonService:
    """Service for handling comparison operations."""

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
        evaluation_a = await self.evaluation_repo.get(comparison_data.evaluation_a_id)
        if not evaluation_a:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation A with ID {comparison_data.evaluation_a_id} not found"
            )

        evaluation_b = await self.evaluation_repo.get(comparison_data.evaluation_b_id)
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

            # Validate compatibility between evaluations
            compatibility_warnings = self._check_evaluation_compatibility(evaluation_a, evaluation_b)
            if compatibility_warnings:
                logger.warning(f"Compatibility warnings for comparison {comparison_id}: {compatibility_warnings}")

            # Update status to "running"
            await self.comparison_repo.update(comparison_id, {"status": "running"})

            # Calculate comparisons
            comparison_results = await self._calculate_comparison(
                evaluation_a, evaluation_b, comparison.metric_configs)

            # Add compatibility warnings
            comparison_results["compatibility_warnings"] = compatibility_warnings

            # Generate natural language insights
            narrative_insights = self._generate_natural_language_insights(
                comparison_results["summary"],
                compatibility_warnings
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

    @staticmethod
    def _check_evaluation_compatibility(
            evaluation_a: Evaluation, evaluation_b: Evaluation
    ) -> List[str]:
        """
        Check compatibility between two evaluations.

        Args:
            evaluation_a: First evaluation
            evaluation_b: Second evaluation

        Returns:
            List[str]: List of compatibility warnings
        """
        warnings = []

        # Check dataset compatibility
        if evaluation_a.dataset_id != evaluation_b.dataset_id:
            warnings.append(
                f"Evaluations use different datasets: {evaluation_a.dataset_id} vs {evaluation_b.dataset_id}")

        # Check method compatibility
        if evaluation_a.method != evaluation_b.method:
            warnings.append(f"Evaluations use different methods: {evaluation_a.method} vs {evaluation_b.method}")

        # Check metric compatibility
        if evaluation_a.metrics and evaluation_b.metrics:
            metrics_a = set(evaluation_a.metrics)
            metrics_b = set(evaluation_b.metrics)
            if metrics_a != metrics_b:
                only_in_a = metrics_a - metrics_b
                only_in_b = metrics_b - metrics_a
                if only_in_a:
                    warnings.append(f"Metrics only in evaluation A: {', '.join(only_in_a)}")
                if only_in_b:
                    warnings.append(f"Metrics only in evaluation B: {', '.join(only_in_b)}")

        # Check sample size compatibility
        if len(evaluation_a.results) == 0 or len(evaluation_b.results) == 0:
            warnings.append(
                f"One or both evaluations have no results: {len(evaluation_a.results)} vs {len(evaluation_b.results)}")
        elif abs(len(evaluation_a.results) - len(evaluation_b.results)) > 0.2 * max(len(evaluation_a.results),
                                                                                    len(evaluation_b.results)):
            warnings.append(
                f"Large difference in sample sizes: {len(evaluation_a.results)} vs {len(evaluation_b.results)}")

        return warnings

    async def _calculate_comparison(
            self,
            evaluation_a: Evaluation,
            evaluation_b: Evaluation,
            metric_configs: Optional[Dict[str, MetricConfig]] = None
    ) -> Dict[str, Any]:
        """
        Calculate detailed comparison between two evaluations.

        Args:
            evaluation_a: First evaluation
            evaluation_b: Second evaluation
            metric_configs: Optional configuration for metrics

        Returns:
            Dict with comparison results and summary
        """
        # Initialize results
        detailed_results = {
            "metric_comparison": {},
            "sample_comparison": {},
            "overall_comparison": {}
        }

        # Prepare metric configurations - use defaults for any missing metrics
        final_metric_configs = {}
        for metric_name in set(DEFAULT_METRIC_CONFIGS.keys()):
            config = DEFAULT_METRIC_CONFIGS[metric_name].copy()
            # Override with user-provided config if available
            if metric_configs and metric_name in metric_configs:
                user_config = metric_configs[metric_name]
                if user_config.higher_is_better is not None:
                    config["higher_is_better"] = user_config.higher_is_better
                if user_config.weight is not None:
                    config["weight"] = user_config.weight
            final_metric_configs[metric_name] = config

        # Calculate metric-level comparison
        metric_comparison = await self._calculate_metric_comparison(
            evaluation_a, evaluation_b, final_metric_configs)
        detailed_results["metric_comparison"] = metric_comparison

        # Calculate sample-level comparison (matching results by dataset item)
        sample_comparison = await self._calculate_sample_comparison(
            evaluation_a, evaluation_b, final_metric_configs)
        detailed_results["sample_comparison"] = sample_comparison

        # Calculate overall evaluation comparison
        overall_comparison = self._calculate_overall_comparison(
            evaluation_a, evaluation_b, metric_comparison)
        detailed_results["overall_comparison"] = overall_comparison

        # Generate summary
        summary = self._generate_comparison_summary(
            evaluation_a,
            evaluation_b,
            metric_comparison,
            sample_comparison,
            overall_comparison
        )

        return {
            "detailed_results": detailed_results,
            "summary": summary
        }

    @staticmethod
    async def _calculate_metric_comparison(
            evaluation_a: Evaluation,
            evaluation_b: Evaluation,
            metric_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate detailed metric comparison between two evaluations.

        Args:
            evaluation_a: First evaluation
            evaluation_b: Second evaluation
            metric_configs: Configuration for metrics

        Returns:
            Dict with metric comparison details
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

        # Calculate averages, differences, and improvements
        metric_comparison = {}
        all_metrics = set(list(metric_scores_a.keys()) + list(metric_scores_b.keys()))

        for metric_name in all_metrics:
            # Get metric configuration
            metric_config = metric_configs.get(metric_name, {
                "higher_is_better": True,
                "weight": 1.0
            })
            higher_is_better = metric_config.get("higher_is_better", True)
            metric_weight = metric_config.get("weight", 1.0)

            # Calculate averages
            values_a = metric_scores_a.get(metric_name, [])
            values_b = metric_scores_b.get(metric_name, [])

            # Skip if no values
            if not values_a and not values_b:
                continue

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

            # If both have values, calculate comparison statistics
            comparison_stats = {}
            if values_a and values_b:
                avg_a = stats_a["average"]
                avg_b = stats_b["average"]

                # Calculate absolute difference
                absolute_diff = avg_b - avg_a

                # Calculate percentage change safely
                if abs(avg_a) > 1e-10:  # Avoid division by very small numbers
                    percentage_diff = (absolute_diff / avg_a) * 100
                else:
                    # Handle division by zero or very small numbers
                    if absolute_diff > 0:
                        percentage_diff = float('inf')
                    elif absolute_diff < 0:
                        percentage_diff = float('-inf')
                    else:
                        percentage_diff = 0

                # Determine if this is an improvement based on metric direction
                is_improvement = (absolute_diff > 0) if higher_is_better else (absolute_diff < 0)

                # Calculate statistical significance
                p_value = None
                is_significant = None
                if len(values_a) > 1 and len(values_b) > 1:
                    try:
                        from scipy import stats
                        # Use Welch's t-test (unequal variances)
                        t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
                        is_significant = p_value < 0.05  # Standard threshold
                    except (ImportError, Exception) as e:
                        logger.warning(f"Failed to calculate statistical significance: {e}")

                comparison_stats = {
                    "absolute_difference": absolute_diff,
                    "percentage_change": percentage_diff,
                    "is_improvement": is_improvement,
                    "p_value": p_value,
                    "is_significant": is_significant,
                    "weight": metric_weight
                }

            # Add metric data to comparison results
            metric_comparison[metric_name] = {
                "evaluation_a": stats_a,
                "evaluation_b": stats_b,
                "comparison": comparison_stats,
                "config": {
                    "higher_is_better": higher_is_better,
                    "weight": metric_weight
                }
            }

        return metric_comparison

    @staticmethod
    async def _calculate_sample_comparison(
            evaluation_a: Evaluation,
            evaluation_b: Evaluation,
            metric_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate comparison for matching samples between two evaluations.

        Args:
            evaluation_a: First evaluation
            evaluation_b: Second evaluation
            metric_configs: Configuration for metrics

        Returns:
            Dict with sample comparison details
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
                    "improvement_rate": 0
                }
            }

        # Compare matched samples
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
                    # Calculate difference
                    absolute_diff = score_b - score_a

                    # Calculate percentage change safely
                    if abs(score_a) > 1e-10:
                        percentage_diff = (absolute_diff / score_a) * 100
                    else:
                        # Handle division by zero or very small numbers
                        if absolute_diff > 0:
                            percentage_diff = float('inf')
                        elif absolute_diff < 0:
                            percentage_diff = float('-inf')
                        else:
                            percentage_diff = 0

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

                    # Add metric-level comparison for this sample
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

                            # Calculate percentage change safely
                            if abs(value_a) > 1e-10:
                                metric_percent = (metric_diff / value_a) * 100
                            else:
                                # Handle division by zero or very small numbers
                                if metric_diff > 0:
                                    metric_percent = float('inf')
                                elif metric_diff < 0:
                                    metric_percent = float('-inf')
                                else:
                                    metric_percent = 0

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

        # Calculate overall sample statistics
        matched_count = len([s for s in matched_results.values() if
                             s.get("evaluation_a", {}).get("result_id") and
                             s.get("evaluation_b", {}).get("result_id")])

        only_in_a = len([s for s in matched_results.values() if
                         s.get("evaluation_a", {}).get("result_id") and
                         not s.get("evaluation_b", {}).get("result_id")])

        only_in_b = len([s for s in matched_results.values() if
                         not s.get("evaluation_a", {}).get("result_id") and
                         s.get("evaluation_b", {}).get("result_id")])

        # Calculate improvement rate safely
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

    @staticmethod
    def _calculate_overall_comparison(
            evaluation_a: Evaluation,
            evaluation_b: Evaluation,
            metric_comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate overall comparison statistics between two evaluations.

        Args:
            evaluation_a: First evaluation
            evaluation_b: Second evaluation
            metric_comparison: Metric comparison data

        Returns:
            Dict with overall comparison statistics
        """
        # Calculate overall score comparison using weighted metrics
        if not evaluation_a.results:
            logger.warning(f"Evaluation A ({evaluation_a.id}) has no results")
            overall_score_a = None
        else:
            # Get all valid overall scores
            valid_scores_a = [r.overall_score for r in evaluation_a.results if r.overall_score is not None]
            overall_score_a = sum(valid_scores_a) / len(valid_scores_a) if valid_scores_a else None

        if not evaluation_b.results:
            logger.warning(f"Evaluation B ({evaluation_b.id}) has no results")
            overall_score_b = None
        else:
            # Get all valid overall scores
            valid_scores_b = [r.overall_score for r in evaluation_b.results if r.overall_score is not None]
            overall_score_b = sum(valid_scores_b) / len(valid_scores_b) if valid_scores_b else None

        # Calculate difference only if both scores exist
        overall_diff = None
        overall_pct = None
        is_improvement = None

        if overall_score_a is not None and overall_score_b is not None:
            overall_diff = overall_score_b - overall_score_a

            # Calculate percentage change safely
            if abs(overall_score_a) > 1e-10:
                overall_pct = (overall_diff / overall_score_a) * 100
            else:
                # Handle division by zero or very small numbers
                if overall_diff > 0:
                    overall_pct = float('inf')
                elif overall_diff < 0:
                    overall_pct = float('-inf')
                else:
                    overall_pct = 0

            is_improvement = overall_diff > 0  # For overall scores, higher is always better

        # Calculate metric improvements
        metric_improvements = 0
        metric_regressions = 0
        significant_improvements = 0
        significant_regressions = 0
        weighted_improvement_score = 0
        total_weights = 0

        for metric_name, metric_data in metric_comparison.items():
            if "comparison" in metric_data and metric_data["comparison"].get("is_improvement") is not None:
                # Get weight from config
                weight = metric_data["comparison"].get("weight", 1.0)
                total_weights += weight

                # Check if this is an improvement
                if metric_data["comparison"]["is_improvement"]:
                    metric_improvements += 1
                    # Add weighted improvement
                    improvement_value = abs(metric_data["comparison"]["absolute_difference"])
                    weighted_improvement_score += improvement_value * weight
                else:
                    metric_regressions += 1
                    # Subtract weighted regression
                    regression_value = abs(metric_data["comparison"]["absolute_difference"])
                    weighted_improvement_score -= regression_value * weight

                # Check if significant
                if metric_data["comparison"].get("is_significant") is True:
                    if metric_data["comparison"]["is_improvement"]:
                        significant_improvements += 1
                    else:
                        significant_regressions += 1

        # Calculate weighted improvement score
        normalized_weighted_score = None
        if total_weights > 0:
            normalized_weighted_score = weighted_improvement_score / total_weights

        # Calculate significance rate
        significant_metrics = significant_improvements + significant_regressions
        significance_rate = (significant_metrics / len(metric_comparison)) * 100 if metric_comparison else 0

        # Return overall statistics
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
                "metric_improvement_rate": (metric_improvements / len(
                    metric_comparison)) * 100 if metric_comparison else 0,
                "weighted_improvement_score": normalized_weighted_score
            }
        }

    @staticmethod
    def _generate_comparison_summary(
            evaluation_a: Evaluation,
            evaluation_b: Evaluation,
            metric_comparison: Dict[str, Any],
            sample_comparison: Dict[str, Any],
            overall_comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a summary of the comparison with key insights.

        Args:
            evaluation_a: First evaluation
            evaluation_b: Second evaluation
            metric_comparison: Metric comparison data
            sample_comparison: Sample comparison data
            overall_comparison: Overall comparison data

        Returns:
            Dict with comparison summary
        """
        # Extract key information
        overall_stats = overall_comparison["overall_scores"]
        metric_stats = overall_comparison["metric_stats"]
        sample_stats = sample_comparison["stats"]

        # Determine if there's an overall improvement
        is_improvement = overall_stats.get("is_improvement", False)
        percentage_change = overall_stats.get("percentage_change", 0)

        # Identify top improvements and regressions
        top_improvements = []
        top_regressions = []

        for metric_name, data in metric_comparison.items():
            if "comparison" in data and data["comparison"].get("absolute_difference") is not None:
                diff_data = {
                    "metric_name": metric_name,
                    "absolute_difference": data["comparison"]["absolute_difference"],
                    "percentage_change": data["comparison"]["percentage_change"],
                    "is_significant": data["comparison"].get("is_significant", False),
                    "p_value": data["comparison"].get("p_value")
                }

                if data["comparison"]["is_improvement"]:
                    top_improvements.append(diff_data)
                else:
                    top_regressions.append(diff_data)

        # Sort by absolute difference
        top_improvements = sorted(top_improvements, key=lambda x: abs(x["absolute_difference"]), reverse=True)[:3]
        top_regressions = sorted(top_regressions, key=lambda x: abs(x["absolute_difference"]), reverse=True)[:3]

        # Generate summary
        summary = {
            "evaluation_a_name": evaluation_a.name,
            "evaluation_b_name": evaluation_b.name,
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
            "weighted_improvement_score": metric_stats.get("weighted_improvement_score")
        }

        return summary

    @staticmethod
    def _generate_natural_language_insights(
            summary: Dict,
            compatibility_warnings: List[str] = None
    ) -> str:
        """
        Generate natural language insights from comparison data.

        Args:
            summary: Comparison summary
            compatibility_warnings: Any compatibility warnings

        Returns:
            str: Natural language insights
        """
        insights = []

        # Start with compatibility warnings if any
        if compatibility_warnings:
            insights.append("⚠️ **Compatibility Warnings:**")
            for warning in compatibility_warnings:
                insights.append(f"- {warning}")
            insights.append("")  # Add blank line

        # Overall assessment
        if summary["overall_result"] == "improved":
            insights.append(f"📈 **Overall Assessment:** {summary['evaluation_b_name']} shows a "
                            f"{summary['percentage_change']:.1f}% improvement over {summary['evaluation_a_name']}.")
        elif summary["overall_result"] == "regressed":
            insights.append(f"📉 **Overall Assessment:** {summary['evaluation_b_name']} shows a "
                            f"{abs(summary['percentage_change']):.1f}% regression compared to {summary['evaluation_a_name']}.")
        else:
            insights.append(f"⚖️ **Overall Assessment:** The comparison between {summary['evaluation_a_name']} and "
                            f"{summary['evaluation_b_name']} is inconclusive due to insufficient data.")

        # Add sample statistics
        total_compared = summary["matched_samples"]
        if total_compared > 0:
            imp_rate = (summary["improved_samples"] / total_compared) * 100
            insights.append(f"\n**Sample Analysis:** Of {total_compared} samples analyzed, "
                            f"{summary['improved_samples']} ({imp_rate:.1f}%) improved and "
                            f"{summary['regressed_samples']} ({100 - imp_rate:.1f}%) regressed.")

        # Key metrics
        if summary["top_improvements"]:
            insights.append("\n**Top Improvements:**")
            for m in summary["top_improvements"][:3]:
                sig_marker = "* " if m.get("is_significant", False) else ""
                insights.append(f"- {sig_marker}{m['metric_name']}: +{m['percentage_change']:.1f}%")

        if summary["top_regressions"]:
            insights.append("\n**Areas for Attention:**")
            for m in summary["top_regressions"][:3]:
                sig_marker = "* " if m.get("is_significant", False) else ""
                insights.append(f"- {sig_marker}{m['metric_name']}: {m['percentage_change']:.1f}%")

        # Add significance note
        if summary.get("significant_improvements", 0) > 0 or summary.get("significant_regressions", 0) > 0:
            total_significant = summary.get("significant_improvements", 0) + summary.get("significant_regressions", 0)
            insights.append(f"\n**Statistical Significance:** {total_significant} of {summary['total_metrics']} "
                            f"metrics show statistically significant changes (p < 0.05, marked with *).")

        # Add weighted score if available
        if summary.get("weighted_improvement_score") is not None:
            weighted_score = summary["weighted_improvement_score"]
            if weighted_score > 0:
                insights.append(
                    f"\n**Weighted Analysis:** Considering metric weights, the overall improvement is positive ({weighted_score:.3f}).")
            else:
                insights.append(
                    f"\n**Weighted Analysis:** Considering metric weights, the overall impact is negative ({weighted_score:.3f}).")

        # Conclusion
        if summary["overall_result"] == "improved":
            insights.append(
                "\n**Conclusion:** This change represents a meaningful improvement and should be considered for adoption.")
        elif summary["overall_result"] == "regressed":
            insights.append(
                "\n**Conclusion:** This change shows regression in overall performance and requires further investigation before adoption.")
        else:
            insights.append("\n**Conclusion:** More data is needed to make a definitive assessment of this change.")

        return "\n".join(insights)

    async def get_comparison(self, comparison_id: UUID) -> Optional[EvaluationComparison]:
        """
        Get comparison by ID.

        Args:
            comparison_id: Comparison ID

        Returns:
            Optional[EvaluationComparison]: Comparison if found, None otherwise
        """
        try:
            return await self.comparison_repo.get_with_evaluations(comparison_id)
        except Exception as e:
            logger.error(f"Error retrieving comparison {comparison_id}: {str(e)}")
            return None

    async def get_user_comparison(self, comparison_id: UUID, user_id: UUID) -> Optional[EvaluationComparison]:
        """
        Get comparison by ID with user ownership check.

        Args:
            comparison_id: Comparison ID
            user_id: User ID for ownership verification

        Returns:
            Optional[EvaluationComparison]: Comparison if found and owned by user, None otherwise
        """
        try:
            comparison = await self.comparison_repo.get_user_owned(comparison_id, user_id)
            if comparison:
                # Load evaluations
                return await self.comparison_repo.get_with_evaluations(comparison_id)
            return None
        except Exception as e:
            logger.error(f"Error retrieving user comparison {comparison_id}: {str(e)}")
            return None

    async def update_comparison(
            self, comparison_id: UUID, comparison_data: ComparisonUpdate, user_id: Optional[UUID] = None
    ) -> Optional[EvaluationComparison]:
        """
        Update comparison by ID with optional user ownership check.

        Args:
            comparison_id: Comparison ID
            comparison_data: Comparison update data
            user_id: Optional user ID for ownership verification

        Returns:
            Optional[EvaluationComparison]: Updated comparison if found, None otherwise

        Raises:
            HTTPException: If update fails or user doesn't have permission
        """
        # Filter out None values
        update_data = {
            k: v for k, v in comparison_data.model_dump().items() if v is not None
        }

        if not update_data:
            # No update needed
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
        """
        Delete comparison by ID with optional user ownership check.

        Args:
            comparison_id: Comparison ID
            user_id: Optional user ID for ownership verification

        Returns:
            bool: True if deleted, False if not found

        Raises:
            AuthorizationException: If user doesn't have permission
        """
        try:
            # First, check if comparison exists and user has permission
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

            # Delete the comparison
            success = await self.comparison_repo.delete(comparison_id)
            if success:
                logger.info(f"Deleted comparison {comparison_id}")
            return success

        except AuthorizationException:
            # Re-raise authorization exceptions
            raise
        except Exception as e:
            logger.error(f"Error deleting comparison {comparison_id}: {str(e)}")
            return False

    async def get_comparison_metrics(
            self, comparison_id: UUID, user_id: Optional[UUID] = None
    ) -> List[MetricDifferenceResponse]:
        """
        Get detailed metric differences for a comparison.

        Args:
            comparison_id: Comparison ID
            user_id: Optional user ID for ownership verification

        Returns:
            List[MetricDifferenceResponse]: List of metric differences

        Raises:
            HTTPException: If comparison not found or user doesn't have access
        """
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
                    p_value=data["comparison"].get("p_value"),
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
            self, comparison_id: UUID, user_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """
        Generate a downloadable report for the comparison.

        Args:
            comparison_id: Comparison ID
            user_id: Optional user ID for ownership verification

        Returns:
            Dict with report data and metadata

        Raises:
            HTTPException: If comparison not found or user doesn't have access
        """
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
            narrative_insights = self._generate_natural_language_insights(
                comparison.summary
            )

        # Format report
        report = {
            "report_title": f"Comparison Report: {comparison.name}",
            "generated_at": str(comparison.updated_at),
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

        return report

    async def get_comparison_visualizations(
            self, comparison_id: UUID, visualization_type: str = "radar", user_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """
        Generate visualization data for charts.

        Args:
            comparison_id: Comparison ID
            visualization_type: Type of visualization ("radar", "bar", "line")
            user_id: Optional user ID for ownership verification

        Returns:
            Dict with visualization data

        Raises:
            HTTPException: If comparison not found or user doesn't have access
        """
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
        """
        Generate data for radar chart visualization.

        Args:
            metric_comparison: Metric comparison data
            eval_a_name: Name of evaluation A
            eval_b_name: Name of evaluation B

        Returns:
            Dict with radar chart data
        """
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
        """
        Generate data for bar chart visualization.

        Args:
            metric_comparison: Metric comparison data
            eval_a_name: Name of evaluation A
            eval_b_name: Name of evaluation B

        Returns:
            Dict with bar chart data
        """
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
        """
        Generate data for line chart visualization.

        Args:
            metric_comparison: Metric comparison data
            eval_a_name: Name of evaluation A
            eval_b_name: Name of evaluation B

        Returns:
            Dict with line chart data
        """
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
        """
        Generate data for significance visualization.

        Args:
            metric_comparison: Metric comparison data
            eval_a_name: Name of evaluation A
            eval_b_name: Name of evaluation B

        Returns:
            Dict with significance chart data
        """
        metrics = []

        for metric_name, data in metric_comparison.items():
            if "comparison" in data and data["comparison"].get("p_value") is not None:
                # Extract data
                p_value = data["comparison"]["p_value"]
                percentage_change = data["comparison"].get("percentage_change", 0)
                is_improvement = data["comparison"].get("is_improvement", False)
                is_significant = data["comparison"].get("is_significant", False)
                higher_is_better = data.get("config", {}).get("higher_is_better", True)

                # Add to metrics list
                metrics.append({
                    "name": metric_name,
                    "p_value": p_value,
                    "percentage_change": percentage_change,
                    "is_improvement": is_improvement,
                    "is_significant": is_significant,
                    "higher_is_better": higher_is_better
                })

        # Sort by p-value
        metrics.sort(key=lambda x: x["p_value"])

        return {
            "type": "significance",
            "metrics": metrics,
            "evaluation_a_name": eval_a_name,
            "evaluation_b_name": eval_b_name
        }
