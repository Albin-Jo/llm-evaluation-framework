# File: backend/app/services/report_service.py
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, BinaryIO
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.db.models.orm import Report, ReportStatus, ReportFormat, Evaluation
from backend.app.db.repositories.base import BaseRepository
from backend.app.db.repositories.report_repository import ReportRepository
from backend.app.db.schema.report_schema import ReportCreate, ReportUpdate, EmailRecipient
from backend.app.services.evaluation_service import EvaluationService
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
        self.evaluation_service = EvaluationService(db_session)
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
        logger.info(f"Creating new report for evaluation: {report_data.evaluation_id}")

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
            self, skip: int = 0, limit: int = 100,
            evaluation_id: Optional[UUID] = None,
            status: Optional[ReportStatus] = None,
            is_public: Optional[bool] = None,
            name: Optional[str] = None,
    ) -> List[Report]:
        """
        List reports with optional filtering.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            evaluation_id: Optional evaluation ID filter
            status: Optional status filter
            is_public: Optional public/private filter
            name: Optional name filter (partial match)

        Returns:
            List[Report]: List of reports
        """
        logger.debug(
            f"Listing reports with filters: evaluation_id={evaluation_id}, status={status}, is_public={is_public}, name={name}")

        filters = {}

        # Add filters if provided
        if evaluation_id:
            filters["evaluation_id"] = evaluation_id
        if status:
            filters["status"] = status
        if is_public is not None:
            filters["is_public"] = is_public

        # Get Reports
        if name:
            # Use custom search method for name partial match
            reports = await self.report_repo.search_by_name(name, skip=skip, limit=limit, additional_filters=filters)
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

    async def generate_report(self, report_id: UUID, force_regenerate: bool = False) -> Report:
        """
        Generate a report file based on the report configuration.

        Args:
            report_id: Report ID
            force_regenerate: Whether to force regeneration even if already generated

        Returns:
            Report: Updated report with file path

        Raises:
            HTTPException: If report not found or generation fails
        """
        logger.info(f"Generating report file for report ID: {report_id}")

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

        # Get evaluation data
        evaluation = await self.evaluation_repo.get(report.evaluation_id)
        if not evaluation:
            logger.error(f"Evaluation {report.evaluation_id} not found for report {report_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation with ID {report.evaluation_id} not found"
            )

        # Get evaluation results and format report content
        try:
            # Get evaluation detail with results
            evaluation_detail = await self.evaluation_service.get_evaluation(report.evaluation_id)

            # Get evaluation results
            results = await self.evaluation_service.get_evaluation_results(report.evaluation_id)

            # Format report content based on config
            report_content = await self._format_report_content(
                evaluation=evaluation_detail,
                results=results,
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

    async def _format_report_content(
            self, evaluation: Evaluation, results: List[Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format report content based on evaluation data and config.

        Args:
            evaluation: Evaluation detail
            results: Evaluation results
            config: Report configuration

        Returns:
            Dict[str, Any]: Formatted report content
        """
        content = {
            "report_title": "Evaluation Report",
            "generated_at": datetime.now().isoformat(),
            "evaluation_id": str(evaluation.id),
            "sections": []
        }

        # Include executive summary if configured
        if config.get("include_executive_summary", True):
            # Calculate overall metrics
            overall_score = 0
            metric_scores = {}

            if results:
                # Calculate average overall score
                overall_score = sum(r.overall_score or 0 for r in results) / len(results)

                # Calculate average metric scores
                for result in results:
                    for metric_score in getattr(result, "metric_scores", []):
                        if not metric_score.name in metric_scores:
                            metric_scores[metric_score.name] = []
                        metric_scores[metric_score.name].append(metric_score.value)

                # Calculate averages
                metric_averages = {
                    name: sum(scores) / len(scores) for name, scores in metric_scores.items()
                }

                content["sections"].append({
                    "title": "Executive Summary",
                    "content": {
                        "overall_score": overall_score,
                        "metric_averages": metric_averages,
                        "total_samples": len(results),
                        "completion_date": evaluation.end_time.isoformat() if evaluation.end_time else None,
                        "evaluation_duration": (
                            (evaluation.end_time - evaluation.start_time).total_seconds()
                            if evaluation.start_time and evaluation.end_time else None
                        )
                    }
                })

        # Include evaluation details if configured
        if config.get("include_evaluation_details", True):
            # Get agent details
            agent = getattr(evaluation, "agent", None)
            agent_details = {
                "id": str(agent.id) if agent else None,
                "name": agent.name if agent else None,
                "domain": agent.domain if agent else None,
                "description": agent.description if agent else None
            }

            # Get dataset details
            dataset = getattr(evaluation, "dataset", None)
            dataset_details = {
                "id": str(dataset.id) if dataset else None,
                "name": dataset.name if dataset else None,
                "type": dataset.type.value if dataset and dataset.type else None,
                "row_count": dataset.row_count if dataset else None
            }

            # Get prompt details
            prompt = getattr(evaluation, "prompt", None)
            prompt_details = {
                "id": str(prompt.id) if prompt else None,
                "name": prompt.name if prompt else None,
                "content": prompt.content if prompt else None
            }

            content["sections"].append({
                "title": "Evaluation Details",
                "content": {
                    "name": evaluation.name,
                    "description": evaluation.description,
                    "method": evaluation.method.value,
                    "status": evaluation.status.value,
                    "config": evaluation.config,
                    "agent": agent_details,
                    "dataset": dataset_details,
                    "prompt": prompt_details,
                    "metrics": evaluation.metrics,
                    "start_time": evaluation.start_time.isoformat() if evaluation.start_time else None,
                    "end_time": evaluation.end_time.isoformat() if evaluation.end_time else None
                }
            })

        # Include metrics overview if configured
        if config.get("include_metrics_overview", True) and results:
            # Get metrics distribution
            metrics_distribution = {}

            for result in results:
                for metric_score in getattr(result, "metric_scores", []):
                    if not metric_score.name in metrics_distribution:
                        metrics_distribution[metric_score.name] = {
                            "values": [],
                            "min": float('inf'),
                            "max": float('-inf'),
                            "avg": 0,
                            "description": getattr(metric_score, "meta_info", {}).get("description", "")
                        }

                    value = metric_score.value
                    metrics_distribution[metric_score.name]["values"].append(value)
                    metrics_distribution[metric_score.name]["min"] = min(
                        metrics_distribution[metric_score.name]["min"], value
                    )
                    metrics_distribution[metric_score.name]["max"] = max(
                        metrics_distribution[metric_score.name]["max"], value
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

            # Format results data
            detailed_results = []
            for result in filtered_results:
                result_dict = {
                    "id": str(result.id),
                    "overall_score": result.overall_score,
                    "input_data": result.input_data,
                    "output_data": result.output_data if config.get("include_agent_responses", True) else None,
                    "processing_time_ms": result.processing_time_ms,
                    "metric_scores": [
                        {
                            "name": metric_score.name,
                            "value": metric_score.value,
                            "weight": metric_score.weight
                        }
                        for metric_score in getattr(result, "metric_scores", [])
                    ]
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
            # Generate HTML report
            html_content = await self._generate_html_report(content)

            with open(f"{settings.STORAGE_LOCAL_PATH}/{file_path}", "w") as f:
                f.write(html_content)
        elif format == ReportFormat.PDF:
            # For PDF, we'll first generate HTML, then convert to PDF
            # For implementation simplicity, we'll use a placeholder PDF generation
            # In a real implementation, you'd use a library like WeasyPrint or a service like wkhtmltopdf

            html_content = await self._generate_html_report(content)

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

    async def _generate_html_report(self, content: Dict[str, Any]) -> str:
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

    def _generate_executive_summary_html(self, content: Dict[str, Any]) -> str:
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

    def _generate_evaluation_details_html(self, content: Dict[str, Any]) -> str:
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

    def _generate_metrics_overview_html(self, content: Dict[str, Any]) -> str:
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

    def _generate_detailed_results_html(self, content: Dict[str, Any]) -> str:
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
            self, report_id: UUID, recipients: List[EmailRecipient],
            subject: Optional[str] = None, message: Optional[str] = None,
            include_pdf: bool = True
    ) -> bool:
        """
        Send a report via email.

        Args:
            report_id: Report ID
            recipients: List of email recipients
            subject: Email subject
            message: Email message
            include_pdf: Whether to include the PDF attachment

        Returns:
            bool: True if email sent successfully

        Raises:
            HTTPException: If report not found or email sending fails
        """
        logger.info(f"Sending report {report_id} via email to {len(recipients)} recipients")

        # Get report
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

        # Get evaluation for report context
        evaluation = await self.evaluation_repo.get(report.evaluation_id)

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

    async def get_report_file(self, report_id: UUID) -> BinaryIO:
        """
        Get the generated report file.

        Args:
            report_id: Report ID

        Returns:
            BinaryIO: File object

        Raises:
            HTTPException: If report not found or file not generated
        """
        logger.info(f"Getting report file for report ID: {report_id}")

        # Get report
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