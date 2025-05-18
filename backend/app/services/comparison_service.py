import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.exceptions import AuthorizationException
from backend.app.db.models.orm import EvaluationComparison, Evaluation, EvaluationResult, MetricScore
from backend.app.db.repositories.base import BaseRepository
from backend.app.db.repositories.comparison_repository import ComparisonRepository
from backend.app.db.repositories.evaluation_repository import EvaluationRepository
from backend.app.db.schema.comparison_schema import ComparisonCreate, ComparisonUpdate, MetricDifferenceResponse

logger = logging.getLogger(__name__)


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

            # Update status to "running"
            await self.comparison_repo.update(comparison_id, {"status": "running"})

            # Calculate comparisons
            comparison_results = await self._calculate_comparison(evaluation_a, evaluation_b)

            # Update comparison with results
            update_data = {
                "status": "completed",
                "comparison_results": comparison_results["detailed_results"],
                "summary": comparison_results["summary"]
            }

            updated_comparison = await self.comparison_repo.update(comparison_id, update_data)
            logger.info(f"Completed comparison calculation for comparison {comparison_id}")

            return updated_comparison

        except Exception as e:
            logger.error(f"Error calculating comparison {comparison_id}: {str(e)}")
            # Update status to "failed"
            await self.comparison_repo.update(comparison_id, {"status": "failed"})
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error calculating comparison: {str(e)}"
            )

    async def _calculate_comparison(
            self, evaluation_a: Evaluation, evaluation_b: Evaluation
    ) -> Dict[str, Any]:
        """
        Calculate detailed comparison between two evaluations.

        Args:
            evaluation_a: First evaluation
            evaluation_b: Second evaluation

        Returns:
            Dict with comparison results and summary
        """
        # Initialize results
        detailed_results = {
            "metric_comparison": {},
            "sample_comparison": {},
            "overall_comparison": {}
        }

        # Calculate metric-level comparison
        metric_comparison = await self._calculate_metric_comparison(evaluation_a, evaluation_b)
        detailed_results["metric_comparison"] = metric_comparison

        # Calculate sample-level comparison (matching results by dataset item)
        sample_comparison = await self._calculate_sample_comparison(evaluation_a, evaluation_b)
        detailed_results["sample_comparison"] = sample_comparison

        # Calculate overall evaluation comparison
        overall_comparison = self._calculate_overall_comparison(evaluation_a, evaluation_b, metric_comparison)
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

    async def _calculate_metric_comparison(
            self, evaluation_a: Evaluation, evaluation_b: Evaluation
    ) -> Dict[str, Any]:
        """
        Calculate detailed metric comparison between two evaluations.

        Args:
            evaluation_a: First evaluation
            evaluation_b: Second evaluation

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
            # Calculate averages
            values_a = metric_scores_a.get(metric_name, [])
            values_b = metric_scores_b.get(metric_name, [])

            avg_a = sum(values_a) / len(values_a) if values_a else None
            avg_b = sum(values_b) / len(values_b) if values_b else None

            # Calculate difference if both have values
            if avg_a is not None and avg_b is not None:
                absolute_diff = avg_b - avg_a
                percentage_diff = (absolute_diff / avg_a) * 100 if avg_a != 0 else 0
                is_improvement = absolute_diff > 0  # Higher scores are better

                metric_comparison[metric_name] = {
                    "evaluation_a": {
                        "average": avg_a,
                        "sample_count": len(values_a),
                        "values": values_a
                    },
                    "evaluation_b": {
                        "average": avg_b,
                        "sample_count": len(values_b),
                        "values": values_b
                    },
                    "comparison": {
                        "absolute_difference": absolute_diff,
                        "percentage_change": percentage_diff,
                        "is_improvement": is_improvement
                    }
                }
            else:
                # Handle case where one evaluation is missing this metric
                metric_comparison[metric_name] = {
                    "evaluation_a": {
                        "average": avg_a,
                        "sample_count": len(values_a) if values_a else 0,
                        "values": values_a
                    },
                    "evaluation_b": {
                        "average": avg_b,
                        "sample_count": len(values_b) if values_b else 0,
                        "values": values_b
                    },
                    "comparison": {
                        "absolute_difference": None,
                        "percentage_change": None,
                        "is_improvement": None,
                        "missing_in": "evaluation_a" if not values_a else ("evaluation_b" if not values_b else None)
                    }
                }

        return metric_comparison

    async def _calculate_sample_comparison(
            self, evaluation_a: Evaluation, evaluation_b: Evaluation
    ) -> Dict[str, Any]:
        """
        Calculate comparison for matching samples between two evaluations.

        Args:
            evaluation_a: First evaluation
            evaluation_b: Second evaluation

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

        # Compare matched samples
        sample_improvements = 0
        sample_regressions = 0

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
                    percentage_diff = (absolute_diff / score_a) * 100 if score_a != 0 else 0
                    is_improvement = absolute_diff > 0  # Higher scores are better

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
                            metric_diff = value_b - value_a
                            metric_percent = (metric_diff / value_a) * 100 if value_a != 0 else 0
                            metric_improvement = metric_diff > 0

                            metric_diffs[metric_name] = {
                                "evaluation_a_value": value_a,
                                "evaluation_b_value": value_b,
                                "absolute_difference": metric_diff,
                                "percentage_change": metric_percent,
                                "is_improvement": metric_improvement
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
                             s["evaluation_a"]["result_id"] and s["evaluation_b"]["result_id"]])

        only_in_a = len([s for s in matched_results.values() if
                         s["evaluation_a"]["result_id"] and not s["evaluation_b"]["result_id"]])

        only_in_b = len([s for s in matched_results.values() if
                         not s["evaluation_a"]["result_id"] and s["evaluation_b"]["result_id"]])

        return {
            "matched_results": matched_results,
            "stats": {
                "matched_count": matched_count,
                "only_in_evaluation_a": only_in_a,
                "only_in_evaluation_b": only_in_b,
                "total_samples": len(all_sample_ids),
                "improved_samples": sample_improvements,
                "regressed_samples": sample_regressions,
                "improvement_rate": (sample_improvements / matched_count) * 100 if matched_count > 0 else 0
            }
        }

    def _calculate_overall_comparison(
            self, evaluation_a: Evaluation, evaluation_b: Evaluation, metric_comparison: Dict[str, Any]
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
        # Calculate overall score comparison
        overall_score_a = 0
        overall_score_b = 0

        if evaluation_a.results:
            overall_score_a = sum(r.overall_score or 0 for r in evaluation_a.results) / len(evaluation_a.results)

        if evaluation_b.results:
            overall_score_b = sum(r.overall_score or 0 for r in evaluation_b.results) / len(evaluation_b.results)

        overall_diff = overall_score_b - overall_score_a
        overall_pct = (overall_diff / overall_score_a) * 100 if overall_score_a != 0 else 0

        # Calculate metric improvements
        metric_improvements = 0
        metric_regressions = 0

        for metric_data in metric_comparison.values():
            if "comparison" in metric_data and metric_data["comparison"].get("is_improvement") is not None:
                if metric_data["comparison"]["is_improvement"]:
                    metric_improvements += 1
                else:
                    metric_regressions += 1

        # Return overall statistics
        return {
            "overall_scores": {
                "evaluation_a": overall_score_a,
                "evaluation_b": overall_score_b,
                "absolute_difference": overall_diff,
                "percentage_change": overall_pct,
                "is_improvement": overall_diff > 0
            },
            "metric_stats": {
                "total_metrics": len(metric_comparison),
                "improved_metrics": metric_improvements,
                "regressed_metrics": metric_regressions,
                "metric_improvement_rate": (metric_improvements / len(
                    metric_comparison)) * 100 if metric_comparison else 0
            }
        }

    def _generate_comparison_summary(
            self,
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
        is_improvement = overall_stats["is_improvement"]
        percentage_change = overall_stats["percentage_change"]

        # Identify top improvements and regressions
        top_improvements = []
        top_regressions = []

        for metric_name, data in metric_comparison.items():
            if "comparison" in data and data["comparison"].get("absolute_difference") is not None:
                diff_data = {
                    "metric_name": metric_name,
                    "absolute_difference": data["comparison"]["absolute_difference"],
                    "percentage_change": data["comparison"]["percentage_change"]
                }

                if data["comparison"]["is_improvement"]:
                    top_improvements.append(diff_data)
                else:
                    top_regressions.append(diff_data)

        # Sort by absolute difference
        top_improvements = sorted(top_improvements, key=lambda x: x["absolute_difference"], reverse=True)[:3]
        top_regressions = sorted(top_regressions, key=lambda x: x["absolute_difference"])[:3]

        # Generate summary
        summary = {
            "evaluation_a_name": evaluation_a.name,
            "evaluation_b_name": evaluation_b.name,
            "overall_result": "improved" if is_improvement else "regressed",
            "percentage_change": percentage_change,
            "total_metrics": metric_stats["total_metrics"],
            "improved_metrics": metric_stats["improved_metrics"],
            "regressed_metrics": metric_stats["regressed_metrics"],
            "improved_samples": sample_stats["improved_samples"],
            "regressed_samples": sample_stats["regressed_samples"],
            "matched_samples": sample_stats["matched_count"],
            "top_improvements": top_improvements,
            "top_regressions": top_regressions
        }

        return summary

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
                    is_improvement=data["comparison"]["is_improvement"]
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
        """
        Generate a downloadable report for the comparison.

        Args:
            comparison_id: Comparison ID
            format: Report format ("json", "html", "pdf")
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
            "comparison_results": comparison.comparison_results
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
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported visualization type: {visualization_type}"
            )

    def _generate_radar_chart_data(
            self, metric_comparison: Dict[str, Any], eval_a_name: str, eval_b_name: str
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

        for metric_name, data in metric_comparison.items():
            if "evaluation_a" in data and "evaluation_b" in data:
                avg_a = data["evaluation_a"].get("average")
                avg_b = data["evaluation_b"].get("average")

                if avg_a is not None and avg_b is not None:
                    labels.append(metric_name)
                    series[0]["data"].append(avg_a)
                    series[1]["data"].append(avg_b)

        return {
            "type": "radar",
            "labels": labels,
            "series": series
        }

    def _generate_bar_chart_data(
            self, metric_comparison: Dict[str, Any], eval_a_name: str, eval_b_name: str
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

        # Add delta/improvement series
        series.append({
            "name": "Improvement",
            "data": [],
            "type": "line"
        })

        for metric_name, data in metric_comparison.items():
            if "evaluation_a" in data and "evaluation_b" in data and "comparison" in data:
                avg_a = data["evaluation_a"].get("average")
                avg_b = data["evaluation_b"].get("average")

                if avg_a is not None and avg_b is not None:
                    categories.append(metric_name)
                    series[0]["data"].append(avg_a)
                    series[1]["data"].append(avg_b)

                    # Add improvement value (percentage change)
                    pct_change = data["comparison"].get("percentage_change", 0)
                    series[2]["data"].append(pct_change)

        return {
            "type": "bar",
            "categories": categories,
            "series": series
        }

    def _generate_line_chart_data(
            self, metric_comparison: Dict[str, Any], eval_a_name: str, eval_b_name: str
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
                        # Only include metrics with actual values
                        metrics.append({
                            "name": metric_name,
                            "evaluation_a": {
                                "name": eval_a_name,
                                "values": values_a
                            },
                            "evaluation_b": {
                                "name": eval_b_name,
                                "values": values_b
                            }
                        })

        return {
            "type": "line",
            "metrics": metrics
        }
