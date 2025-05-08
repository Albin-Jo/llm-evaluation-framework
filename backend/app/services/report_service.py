import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, BinaryIO
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.core.config import settings
from backend.app.db.models.orm import Report, ReportStatus, ReportFormat, Evaluation, EvaluationResult
from backend.app.db.repositories.base import BaseRepository
from backend.app.db.repositories.report_repository import ReportRepository
from backend.app.db.schema.report_schema import ReportCreate, ReportUpdate, EmailRecipient
from backend.app.services.storage import get_storage_service

# Configure logging
logger = logging.getLogger(__name__)


class ReportService:
    """Service for report operations."""

    def __init__(self, db_session: AsyncSession):
        """
        Initialize the report service.

        Args:
            db_session: Database session
        """
        self.db_session = db_session
        self.report_repo = ReportRepository(db_session)
        self.evaluation_repo = BaseRepository(Evaluation, db_session)
        self.storage_service = get_storage_service()

    async def create_report(self, report_data: ReportCreate) -> Report:
        """
        Create a new report.

        Args:
            report_data: Report creation data

        Returns:
            Report: Created report

        Raises:
            HTTPException: If validation fails
        """
        logger.info(f"Creating new report for evaluation: {report_data}")

        # Check if evaluation exists
        evaluation = await self.evaluation_repo.get(report_data.evaluation_id)
        if not evaluation:
            logger.warning(f"Evaluation {report_data.evaluation_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation with ID {report_data.evaluation_id} not found"
            )

        # Create config based on include flags
        config = report_data.config or {}
        config.update({
            "include_executive_summary": report_data.include_executive_summary,
            "include_evaluation_details": report_data.include_evaluation_details,
            "include_metrics_overview": report_data.include_metrics_overview,
            "include_detailed_results": report_data.include_detailed_results,
            "include_agent_responses": report_data.include_agent_responses,
            "max_examples": report_data.max_examples
        })

        # Create report
        report_dict = report_data.model_dump(exclude={
            "include_executive_summary",
            "include_evaluation_details",
            "include_metrics_overview",
            "include_detailed_results",
            "include_agent_responses",
            "max_examples"
        })
        report_dict["config"] = config
        report_dict["status"] = ReportStatus.DRAFT

        try:
            report = await self.report_repo.create(report_dict)
            logger.info(f"Report created successfully: {report.id}")
            return report
        except Exception as e:
            logger.error(f"Error creating report: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating report: {str(e)}"
            )

    async def get_report(self, report_id: UUID) -> Report:
        """
        Get a report by ID.

        Args:
            report_id: Report ID

        Returns:
            Report: Retrieved report

        Raises:
            HTTPException: If report not found
        """
        logger.debug(f"Getting report with ID: {report_id}")

        report = await self.report_repo.get(report_id)
        if not report:
            logger.warning(f"Report with ID {report_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report with ID {report_id} not found"
            )

        return report

    async def list_reports(
            self,
            skip: int = 0,
            limit: int = 100,
            evaluation_id: Optional[UUID] = None,
            status: Optional[ReportStatus] = None,
            is_public: Optional[bool] = None,
            name: Optional[str] = None,
            user_id: Optional[UUID] = None
    ) -> List[Report]:
        """
        List reports with optional filtering and user-based access control.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            evaluation_id: Optional evaluation ID filter
            status: Optional status filter
            is_public: Optional public/private filter
            name: Optional name filter (partial match)
            user_id: Optional user ID for ownership filtering

        Returns:
            List[Report]: List of reports
        """
        logger.debug(
            f"Listing reports with filters: evaluation_id={evaluation_id}, "
            f"status={status}, is_public={is_public}, name={name}, user_id={user_id}"
        )

        filters = {}

        # Add filters if provided
        if evaluation_id:
            filters["evaluation_id"] = evaluation_id
        if status:
            filters["status"] = status
        if is_public is not None:
            filters["is_public"] = is_public

        # Add user filter if provided
        if user_id:
            filters["created_by_id"] = user_id

        # Get Reports
        if name:
            # Use custom search method for name partial match
            reports = await self.report_repo.search_by_name(
                name, skip=skip, limit=limit, additional_filters=filters, user_id=user_id
            )
        else:
            # Use standard get_multi for exact filters
            reports = await self.report_repo.get_multi(skip=skip, limit=limit, filters=filters)

        return reports

    async def update_report(self, report_id: UUID, report_data: ReportUpdate) -> Report:
        """
        Update a report by ID.

        Args:
            report_id: Report ID
            report_data: Updated report data

        Returns:
            Report: Updated report

        Raises:
            HTTPException: If report not found or validation fails
        """
        logger.info(f"Updating report with ID: {report_id}")

        report = await self.report_repo.get(report_id)
        if not report:
            logger.warning(f"Report with ID {report_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report with ID {report_id} not found"
            )

        # Update the Report
        update_data = {
            k: v for k, v in report_data.model_dump().items() if v is not None
        }

        if not update_data:
            return report

        updated_report = await self.report_repo.update(report_id, update_data)
        logger.info(f"Report updated successfully: {report_id}")

        return updated_report

    async def delete_report(self, report_id: UUID) -> bool:
        """
        Delete a report by ID.

        Args:
            report_id: Report ID

        Returns:
            bool: True if deleted successfully

        Raises:
            HTTPException: If report not found or deletion fails
        """
        logger.info(f"Deleting report with ID: {report_id}")

        report = await self.report_repo.get(report_id)
        if not report:
            logger.warning(f"Report with ID {report_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report with ID {report_id} not found"
            )

        # Delete report file if it exists
        if report.file_path:
            try:
                await self.storage_service.delete_file(report.file_path)
            except Exception as e:
                logger.warning(f"Error deleting report file {report.file_path}: {str(e)}")

        # Delete the Report
        success = await self.report_repo.delete(report_id)
        if not success:
            logger.error(f"Failed to delete report: {report_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete Report"
            )

        logger.info(f"Report deleted successfully: {report_id}")
        return True

    async def fetch_evaluation_data(
            self,
            evaluation_id: UUID,
            user_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """
        Fetch all necessary evaluation data for report generation in a single query.
        Optionally verifies user access to the evaluation.

        This method eagerly loads all relationships to avoid lazy loading.

        Args:
            evaluation_id: The evaluation ID
            user_id: Optional user ID for ownership verification

        Returns:
            Dict containing the evaluation, results, and related data

        Raises:
            HTTPException: If evaluation not found or user doesn't have access
        """
        # Create query that joins all necessary tables and loads them eagerly
        query = (
            select(Evaluation)
            .options(
                selectinload(Evaluation.agent),
                selectinload(Evaluation.dataset),
                selectinload(Evaluation.prompt),
                selectinload(Evaluation.results).selectinload(EvaluationResult.metric_scores)
            )
            .where(Evaluation.id == evaluation_id)
        )

        # Add user filter if provided
        if user_id:
            query = query.where(Evaluation.created_by_id == user_id)

        # Execute query
        result = await self.db_session.execute(query)
        evaluation = result.scalar_one_or_none()

        if not evaluation:
            logger.error(f"Evaluation {evaluation_id} not found or user {user_id} doesn't have access")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation with ID {evaluation_id} not found or you don't have access to it"
            )

        # Extract all data needed for report in dictionary format
        # This removes all SQLAlchemy dependency for later processing

        # Process results
        results_data = []
        for result in evaluation.results:
            # Process metric scores
            metric_scores_data = []
            for metric in result.metric_scores:
                metric_scores_data.append({
                    "name": metric.name,
                    "value": metric.value,
                    "weight": metric.weight,
                    "meta_info": metric.meta_info
                })

            # Add result
            results_data.append({
                "id": str(result.id),
                "overall_score": result.overall_score,
                "input_data": result.input_data,
                "output_data": result.output_data,
                "processing_time_ms": result.processing_time_ms,
                "metric_scores": metric_scores_data
            })

        # Get agent, dataset, prompt data
        agent_data = None
        if evaluation.agent:
            agent_data = {
                "id": str(evaluation.agent.id),
                "name": evaluation.agent.name,
                "domain": evaluation.agent.domain,
                "description": evaluation.agent.description
            }

        dataset_data = None
        if evaluation.dataset:
            dataset_data = {
                "id": str(evaluation.dataset.id),
                "name": evaluation.dataset.name,
                "type": evaluation.dataset.type.value if evaluation.dataset.type else None,
                "row_count": evaluation.dataset.row_count
            }

        prompt_data = None
        if evaluation.prompt:
            prompt_data = {
                "id": str(evaluation.prompt.id),
                "name": evaluation.prompt.name,
                "content": evaluation.prompt.content
            }

        # Create evaluation data
        evaluation_data = {
            "id": str(evaluation.id),
            "name": evaluation.name,
            "description": evaluation.description,
            "method": evaluation.method.value if evaluation.method else None,
            "status": evaluation.status.value if evaluation.status else None,
            "config": evaluation.config,
            "metrics": evaluation.metrics,
            "start_time": evaluation.start_time,
            "end_time": evaluation.end_time,
            "agent": agent_data,
            "dataset": dataset_data,
            "prompt": prompt_data
        }

        return {
            "evaluation": evaluation_data,
            "results": results_data
        }

    async def generate_report(
            self,
            report_id: UUID,
            force_regenerate: bool = False,
            user_id: Optional[UUID] = None
    ) -> Report:
        """
        Generate a report file based on the report configuration with user verification.

        Args:
            report_id: Report ID
            force_regenerate: Whether to force regeneration even if already generated
            user_id: Optional user ID for ownership verification

        Returns:
            Report: Updated report with file path

        Raises:
            HTTPException: If report not found, user doesn't have access, or generation fails
        """
        logger.info(f"Generating report file for report ID: {report_id}")

        # Get report with user verification if user_id provided
        if user_id:
            report = await self.report_repo.get_user_report(report_id, user_id)
            if not report:
                logger.warning(f"Report with ID {report_id} not found or user {user_id} doesn't have access")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report with ID {report_id} not found or you don't have permission to generate it"
                )
        else:
            report = await self.report_repo.get(report_id)
            if not report:
                logger.warning(f"Report with ID {report_id} not found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report with ID {report_id} not found"
                )

        # Check if report is already generated and force_regenerate is False
        if report.status == ReportStatus.GENERATED and not force_regenerate and report.file_path:
            logger.info(f"Report {report_id} already generated and force_regenerate=False")
            return report

        try:
            # Fetch all evaluation data in a single query to avoid lazy loading later
            # Pass user_id for ownership verification of the evaluation
            data = await self.fetch_evaluation_data(report.evaluation_id, user_id)
            evaluation_data = data["evaluation"]
            results_data = data["results"]

            # Format report content
            report_content = self._format_report_content(
                evaluation=evaluation_data,
                results=results_data,
                config=report.config or {}
            )

            # Generate file based on format
            file_path = await self._generate_report_file(
                report_id=report_id,
                content=report_content,
                format=report.format
            )

            # Update report status and file path
            update_data = {
                "status": ReportStatus.GENERATED,
                "file_path": file_path,
                "content": report_content
            }

            updated_report = await self.report_repo.update(report_id, update_data)
            logger.info(f"Report file generated successfully: {file_path}")

            return updated_report

        except Exception as e:
            logger.error(f"Error generating report file: {str(e)}")

            # Update report status to failed
            update_data = {
                "status": ReportStatus.FAILED
            }
            await self.report_repo.update(report_id, update_data)

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error generating report file: {str(e)}"
            )

    @staticmethod
    def _format_report_content(
            evaluation: Dict[str, Any], results: List[Dict[str, Any]], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format report content based on evaluation data and config.

        This is now a synchronous function that works with pre-fetched data
        to avoid any SQLAlchemy operations.

        Args:
            evaluation: Evaluation data dictionary
            results: List of result data dictionaries
            config: Report configuration

        Returns:
            Dict[str, Any]: Formatted report content
        """
        content = {
            "report_title": "Evaluation Report",
            "generated_at": datetime.now().isoformat(),
            "evaluation_id": evaluation["id"],
            "sections": []
        }

        # Include executive summary if configured
        if config.get("include_executive_summary", True):
            # Calculate overall metrics
            overall_score = 0
            metric_scores = {}

            if results:
                # Calculate average overall score
                overall_score = sum(r["overall_score"] or 0 for r in results) / len(results)

                # Calculate average metric scores
                for result in results:
                    for metric_score in result["metric_scores"]:
                        if not metric_score["name"] in metric_scores:
                            metric_scores[metric_score["name"]] = []
                        metric_scores[metric_score["name"]].append(metric_score["value"])

                # Calculate averages
                metric_averages = {
                    name: sum(scores) / len(scores) for name, scores in metric_scores.items()
                }

                # Format dates
                completion_date = None
                if evaluation["end_time"]:
                    if isinstance(evaluation["end_time"], datetime):
                        completion_date = evaluation["end_time"].isoformat()
                    else:
                        completion_date = evaluation["end_time"]

                # Calculate duration
                duration = None
                if evaluation["start_time"] and evaluation["end_time"]:
                    if isinstance(evaluation["start_time"], datetime) and isinstance(evaluation["end_time"], datetime):
                        duration = (evaluation["end_time"] - evaluation["start_time"]).total_seconds()

                content["sections"].append({
                    "title": "Executive Summary",
                    "content": {
                        "overall_score": overall_score,
                        "metric_averages": metric_averages,
                        "total_samples": len(results),
                        "completion_date": completion_date,
                        "evaluation_duration": duration
                    }
                })

        # Include evaluation details if configured
        if config.get("include_evaluation_details", True):
            # Get start and end times in ISO format if they're datetime objects
            start_time = None
            if evaluation["start_time"]:
                if isinstance(evaluation["start_time"], datetime):
                    start_time = evaluation["start_time"].isoformat()
                else:
                    start_time = evaluation["start_time"]

            end_time = None
            if evaluation["end_time"]:
                if isinstance(evaluation["end_time"], datetime):
                    end_time = evaluation["end_time"].isoformat()
                else:
                    end_time = evaluation["end_time"]

            content["sections"].append({
                "title": "Evaluation Details",
                "content": {
                    "name": evaluation["name"],
                    "description": evaluation["description"],
                    "method": evaluation["method"],
                    "status": evaluation["status"],
                    "config": evaluation["config"],
                    "agent": evaluation["agent"],
                    "dataset": evaluation["dataset"],
                    "prompt": evaluation["prompt"],
                    "metrics": evaluation["metrics"],
                    "start_time": start_time,
                    "end_time": end_time
                }
            })

        # Include metrics overview if configured
        if config.get("include_metrics_overview", True) and results:
            # Get metrics distribution
            metrics_distribution = {}

            for result in results:
                for metric_score in result["metric_scores"]:
                    if not metric_score["name"] in metrics_distribution:
                        metrics_distribution[metric_score["name"]] = {
                            "values": [],
                            "min": float('inf'),
                            "max": float('-inf'),
                            "avg": 0,
                            "description": metric_score.get("meta_info", {}).get("description", "")
                            if isinstance(metric_score.get("meta_info"), dict) else ""
                        }

                    value = metric_score["value"]
                    metrics_distribution[metric_score["name"]]["values"].append(value)
                    metrics_distribution[metric_score["name"]]["min"] = min(
                        metrics_distribution[metric_score["name"]]["min"], value
                    )
                    metrics_distribution[metric_score["name"]]["max"] = max(
                        metrics_distribution[metric_score["name"]]["max"], value
                    )

            # Calculate averages
            for metric, data in metrics_distribution.items():
                data["avg"] = sum(data["values"]) / len(data["values"])

            content["sections"].append({
                "title": "Metrics Overview",
                "content": {
                    "metrics_distribution": metrics_distribution
                }
            })

        # Include detailed results if configured
        if config.get("include_detailed_results", True) and results:
            # Limit number of examples if configured
            max_examples = config.get("max_examples")
            filtered_results = results[:max_examples] if max_examples else results

            # Format results data - use the results we already processed
            detailed_results = []
            for result in filtered_results:
                result_dict = {
                    "id": result["id"],
                    "overall_score": result["overall_score"],
                    "input_data": result["input_data"],
                    "output_data": result["output_data"] if config.get("include_agent_responses", True) else None,
                    "processing_time_ms": result["processing_time_ms"],
                    "metric_scores": result["metric_scores"]
                }
                detailed_results.append(result_dict)

            content["sections"].append({
                "title": "Detailed Results",
                "content": {
                    "results": detailed_results
                }
            })

        return content

    async def _generate_report_file(
            self, report_id: UUID, content: Dict[str, Any], format: ReportFormat
    ) -> str:
        """
        Generate a report file in the specified format.

        Args:
            report_id: Report ID
            content: Report content
            format: Report format

        Returns:
            str: File path to the generated report

        Raises:
            Exception: If file generation fails
        """
        # Create a unique filename
        file_ext = format.value
        filename = f"report_{report_id}.{file_ext}"

        # Generate file path in storage
        environment = settings.APP_ENV
        dir_path = f"{environment}/reports"
        file_path = f"{dir_path}/{filename}"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(f"{settings.STORAGE_LOCAL_PATH}/{file_path}"), exist_ok=True)

        if format == ReportFormat.JSON:
            # Write JSON file
            with open(f"{settings.STORAGE_LOCAL_PATH}/{file_path}", "w") as f:
                json.dump(content, f, indent=2)
        elif format == ReportFormat.HTML:
            # Generate HTML report - Fixed to call without await
            html_content = self.generate_html_report(content)

            with open(f"{settings.STORAGE_LOCAL_PATH}/{file_path}", "w") as f:
                f.write(html_content)
        elif format == ReportFormat.PDF:
            # For PDF, we'll first generate HTML, then convert to PDF
            # Fixed to call without await
            html_content = self.generate_html_report(content)

            # Write HTML to temporary file
            temp_html_path = f"{dir_path}/temp_{report_id}.html"
            with open(f"{settings.STORAGE_LOCAL_PATH}/{temp_html_path}", "w") as f:
                f.write(html_content)

            # In a real implementation, convert HTML to PDF
            # For now, we'll just generate a simple PDF with the content
            with open(f"{settings.STORAGE_LOCAL_PATH}/{file_path}", "w") as f:
                f.write("PDF CONTENT: " + json.dumps(content))

            # Clean up temporary HTML file
            try:
                os.remove(f"{settings.STORAGE_LOCAL_PATH}/{temp_html_path}")
            except:
                pass

        return file_path

    def generate_html_report(self, content: Dict[str, Any]) -> str:
        """
        Generate an HTML report from the content.

        Args:
            content: Report content

        Returns:
            str: HTML report content
        """
        # This is a simplified HTML report template
        # In a real implementation, you might use a template engine like Jinja2

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{content.get('report_title', 'Evaluation Report')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
                .section-title {{ border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }}
                .metric {{ margin-bottom: 15px; }}
                .metric-label {{ font-weight: bold; }}
                .result-item {{ margin-bottom: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #ffffcc; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{content.get('report_title', 'Evaluation Report')}</h1>
                    <p>Generated on: {datetime.fromisoformat(content.get('generated_at', datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Evaluation ID: {content.get('evaluation_id', 'N/A')}</p>
                </div>
        """

        # Add sections
        for section in content.get('sections', []):
            section_title = section.get('title', '')
            section_content = section.get('content', {})

            html += f"""
                <div class="section">
                    <h2 class="section-title">{section_title}</h2>
            """

            if section_title == "Executive Summary":
                html += self._generate_executive_summary_html(section_content)
            elif section_title == "Evaluation Details":
                html += self._generate_evaluation_details_html(section_content)
            elif section_title == "Metrics Overview":
                html += self._generate_metrics_overview_html(section_content)
            elif section_title == "Detailed Results":
                html += self._generate_detailed_results_html(section_content)

            html += """
                </div>
            """

        html += """
            </div>
        </body>
        </html>
        """

        return html

    @staticmethod
    def _generate_executive_summary_html(content: Dict[str, Any]) -> str:
        """Generate HTML for executive summary section."""
        overall_score = content.get('overall_score', 0)
        metric_averages = content.get('metric_averages', {})
        total_samples = content.get('total_samples', 0)
        completion_date = content.get('completion_date')
        duration = content.get('evaluation_duration')

        formatted_completion = "N/A"
        formatted_duration = "N/A"

        if completion_date:
            try:
                formatted_completion = datetime.fromisoformat(completion_date).strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_completion = completion_date

        if duration is not None:
            hours, remainder = divmod(duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_duration = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

        html = f"""
            <div class="summary">
                <div class="metric">
                    <span class="metric-label">Overall Score:</span> {overall_score:.2f}
                </div>
                <div class="metric">
                    <span class="metric-label">Total Samples:</span> {total_samples}
                </div>
                <div class="metric">
                    <span class="metric-label">Completion Date:</span> {formatted_completion}
                </div>
                <div class="metric">
                    <span class="metric-label">Evaluation Duration:</span> {formatted_duration}
                </div>

                <h3>Metric Averages</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Average Score</th>
                    </tr>
        """

        for name, score in metric_averages.items():
            html += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{score:.2f}</td>
                    </tr>
            """

        html += """
                </table>
            </div>
        """

        return html

    @staticmethod
    def _generate_evaluation_details_html(content: Dict[str, Any]) -> str:
        """Generate HTML for evaluation details section."""
        name = content.get('name', 'N/A')
        description = content.get('description', 'N/A')
        method = content.get('method', 'N/A')
        status = content.get('status', 'N/A')

        agent = content.get('agent', {})
        dataset = content.get('dataset', {})
        prompt = content.get('prompt', {})

        html = f"""
            <div class="evaluation-details">
                <div class="metric">
                    <span class="metric-label">Name:</span> {name}
                </div>
                <div class="metric">
                    <span class="metric-label">Description:</span> {description}
                </div>
                <div class="metric">
                    <span class="metric-label">Method:</span> {method}
                </div>
                <div class="metric">
                    <span class="metric-label">Status:</span> {status}
                </div>

                <h3>Agent</h3>
                <div class="metric">
                    <span class="metric-label">Name:</span> {agent.get('name', 'N/A')}
                </div>
                <div class="metric">
                    <span class="metric-label">Domain:</span> {agent.get('domain', 'N/A')}
                </div>

                <h3>Dataset</h3>
                <div class="metric">
                    <span class="metric-label">Name:</span> {dataset.get('name', 'N/A')}
                </div>
                <div class="metric">
                    <span class="metric-label">Type:</span> {dataset.get('type', 'N/A')}
                </div>
                <div class="metric">
                    <span class="metric-label">Row Count:</span> {dataset.get('row_count', 'N/A')}
                </div>

                <h3>Prompt</h3>
                <div class="metric">
                    <span class="metric-label">Name:</span> {prompt.get('name', 'N/A')}
                </div>
                <div class="prompt-content">
                    <pre>{prompt.get('content', 'N/A')}</pre>
                </div>
            </div>
        """

        return html

    @staticmethod
    def _generate_metrics_overview_html(content: Dict[str, Any]) -> str:
        """Generate HTML for metrics overview section."""
        metrics_distribution = content.get('metrics_distribution', {})

        html = """
            <div class="metrics-overview">
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Average</th>
                        <th>Description</th>
                    </tr>
        """

        for name, data in metrics_distribution.items():
            html += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{data.get('min', 'N/A'):.2f}</td>
                        <td>{data.get('max', 'N/A'):.2f}</td>
                        <td>{data.get('avg', 'N/A'):.2f}</td>
                        <td>{data.get('description', 'N/A')}</td>
                    </tr>
            """

        html += """
                </table>
            </div>
        """

        return html

    @staticmethod
    def _generate_detailed_results_html(content: Dict[str, Any]) -> str:
        """Generate HTML for detailed results section."""
        results = content.get('results', [])

        if not results:
            return "<p>No detailed results available.</p>"

        html = f"""
            <div class="detailed-results">
                <p>Showing {len(results)} result(s)</p>
        """

        for i, result in enumerate(results):
            input_data = result.get('input_data', {})
            output_data = result.get('output_data', {})
            metric_scores = result.get('metric_scores', [])

            query = input_data.get('query', 'N/A')
            context = input_data.get('context', 'N/A')
            ground_truth = input_data.get('ground_truth', '')
            answer = output_data.get('answer', 'N/A') if output_data else 'N/A'

            # Truncate long text for display
            if len(context) > 300:
                context = context[:300] + "..."

            html += f"""
                <div class="result-item">
                    <h4>Example {i + 1} (Score: {result.get('overall_score', 0):.2f})</h4>

                    <div class="metric">
                        <span class="metric-label">Query:</span> {query}
                    </div>

                    <div class="metric">
                        <span class="metric-label">Context:</span>
                        <pre>{context}</pre>
                    </div>
            """

            if ground_truth:
                html += f"""
                    <div class="metric">
                        <span class="metric-label">Ground Truth:</span> {ground_truth}
                    </div>
                """

            html += f"""
                    <div class="metric">
                        <span class="metric-label">Answer:</span> {answer}
                    </div>

                    <h5>Metrics</h5>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Score</th>
                        </tr>
            """

            for score in metric_scores:
                html += f"""
                        <tr>
                            <td>{score.get('name', 'N/A')}</td>
                            <td>{score.get('value', 0):.2f}</td>
                        </tr>
                """

            html += """
                    </table>
                </div>
            """

        html += """
            </div>
        """

        return html

    async def send_report_email(
            self,
            report_id: UUID,
            recipients: List[EmailRecipient],
            subject: Optional[str] = None,
            message: Optional[str] = None,
            include_pdf: bool = True,
            user_id: Optional[UUID] = None
    ) -> bool:
        """
        Send a report via email with user verification.

        Args:
            report_id: Report ID
            recipients: List of email recipients
            subject: Email subject
            message: Email message
            include_pdf: Whether to include the PDF attachment
            user_id: Optional user ID for ownership verification

        Returns:
            bool: True if email sent successfully

        Raises:
            HTTPException: If report not found, user doesn't have access, or email sending fails
        """
        logger.info(f"Sending report {report_id} via email to {len(recipients)} recipients")

        # Get report with user verification if user_id provided
        if user_id:
            report = await self.report_repo.get_user_report(report_id, user_id)
            if not report:
                logger.warning(f"Report with ID {report_id} not found or user {user_id} doesn't have access")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report with ID {report_id} not found or you don't have permission to send it"
                )
        else:
            report = await self.report_repo.get(report_id)
            if not report:
                logger.warning(f"Report with ID {report_id} not found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report with ID {report_id} not found"
                )

        # Ensure report is generated
        if report.status != ReportStatus.GENERATED or not report.file_path:
            logger.warning(f"Report {report_id} is not generated yet")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Report is not generated yet. Please generate the report first."
            )

        # Get evaluation for report context - using regular repo to avoid lazy loading issues
        from sqlalchemy import select
        query = select(Evaluation).where(Evaluation.id == report.evaluation_id)
        result = await self.db_session.execute(query)
        evaluation = result.scalar_one_or_none()

        # Default subject and message if not provided
        if not subject:
            subject = f"Evaluation Report: {report.name}"

        if not message:
            message = f"""
            Please find attached the evaluation report for {evaluation.name if evaluation else 'the evaluation'}.

            This report was generated on {datetime.now().strftime('%Y-%m-%d')}.

            For any questions regarding this report, please contact the system administrator.
            """

        try:
            # In a real implementation, use an email service here
            # For now, we'll just log the email details

            logger.info(f"Would send email: {subject}")
            logger.info(f"To: {', '.join([r.email for r in recipients])}")
            logger.info(f"Message: {message}")
            logger.info(f"Attachment: {report.file_path if include_pdf else 'None'}")

            # Update report status and last_sent_at
            update_data = {
                "status": ReportStatus.SENT,
                "last_sent_at": datetime.now()
            }

            await self.report_repo.update(report_id, update_data)

            return True

        except Exception as e:
            logger.error(f"Error sending report email: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error sending report email: {str(e)}"
            )

    async def get_report_file(self, report_id: UUID, user_id: Optional[UUID] = None) -> BinaryIO:
        """
        Get the generated report file with optional user verification.

        Args:
            report_id: Report ID
            user_id: Optional user ID for ownership verification

        Returns:
            BinaryIO: File object

        Raises:
            HTTPException: If report not found, user doesn't have access, or file not generated
        """
        logger.info(f"Getting report file for report ID: {report_id}")

        # Get report with user verification if user_id provided
        if user_id:
            report = await self.report_repo.get_user_report(report_id, user_id)
            if not report:
                # Check if it's a public report
                report = await self.report_repo.get_accessible_report(report_id, user_id)

                if not report:
                    logger.warning(f"Report with ID {report_id} not found or user {user_id} doesn't have access")
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Report with ID {report_id} not found or you don't have access to it"
                    )
        else:
            report = await self.report_repo.get(report_id)
            if not report:
                logger.warning(f"Report with ID {report_id} not found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report with ID {report_id} not found"
                )

        # Check if report is generated
        if report.status != ReportStatus.GENERATED or not report.file_path:
            logger.warning(f"Report {report_id} file is not generated yet")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Report file is not generated yet"
            )

        try:
            # In a real implementation, fetch the file from storage service
            file_path = f"{settings.STORAGE_LOCAL_PATH}/{report.file_path}"

            if not os.path.exists(file_path):
                logger.error(f"Report file not found at {file_path}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Report file not found"
                )

            # Open file in binary mode for download
            return open(file_path, "rb")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting report file: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting report file: {str(e)}"
            )

    async def count_reports_by_status(self, user_id: Optional[UUID] = None) -> Dict[str, int]:
        """
        Count reports grouped by status, optionally filtering by user ownership.

        Args:
            user_id: Optional user ID to filter by ownership

        Returns:
            Dictionary mapping status to count
        """
        # Use the repository method with user filtering
        return await self.report_repo.count_reports_by_status(user_id)

    async def get_user_report(self, report_id: UUID, user_id: Optional[UUID] = None) -> Report:
        """
        Get a report by ID with user verification.

        Args:
            report_id: Report ID
            user_id: Optional user ID for ownership verification

        Returns:
            Report: Retrieved report

        Raises:
            HTTPException: If report not found or user doesn't have access
        """
        logger.debug(f"Getting report with ID: {report_id} for user: {user_id}")

        if user_id:
            # Get report owned by this user
            report = await self.report_repo.get_user_report(report_id, user_id)

            if not report:
                # Check if it's a public report
                report = await self.report_repo.get_accessible_report(report_id, user_id)

                if not report:
                    logger.warning(f"Report with ID {report_id} not found or user {user_id} doesn't have access")
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Report with ID {report_id} not found or you don't have access to it"
                    )
        else:
            # Without user_id, just get the report
            report = await self.report_repo.get(report_id)
            if not report:
                logger.warning(f"Report with ID {report_id} not found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report with ID {report_id} not found"
                )

        return report
