# File: backend/app/workers/tasks.py
import asyncio
import logging
from typing import Optional
from uuid import UUID

from celery import Celery
from celery.signals import worker_ready

from backend.app.core.config import settings
from backend.app.db.models.orm import EvaluationStatus
from backend.app.db.session import db_session
from backend.app.services.evaluation_service import EvaluationService

# Configure Celery
celery_app = Celery(
    "evaluation_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Configure logging
logger = logging.getLogger(__name__)


@celery_app.task(name="run_evaluation", bind=True, max_retries=3)
def run_evaluation_task(self, evaluation_id: str, jwt_token: Optional[str] = None) -> str:
    """
    Run an evaluation as a background task.

    Args:
        evaluation_id: Evaluation ID
        jwt_token: Optional JWT token for MCP agent authentication

    Returns:
        str: Task result
    """
    # Convert string to UUID
    evaluation_uuid = UUID(evaluation_id)

    # Run the evaluation in an asyncio event loop
    loop = asyncio.get_event_loop()
    try:
        return loop.run_until_complete(_run_evaluation(evaluation_uuid, jwt_token))
    except Exception as exc:
        logger.exception(f"Error running evaluation {evaluation_id}")
        # Retry with exponential backoff
        retry_in = 2 ** self.request.retries
        self.retry(exc=exc, countdown=retry_in)


async def _run_evaluation(evaluation_id: UUID, jwt_token: Optional[str] = None) -> str:
    """
    Internal async function to run an evaluation.

    Args:
        evaluation_id: Evaluation ID
        jwt_token: Optional JWT token for MCP agent authentication

    Returns:
        str: Task result
    """
    async with db_session() as session:
        evaluation_service = EvaluationService(session)

        # Get the evaluation
        evaluation = await evaluation_service.get_evaluation(evaluation_id)
        if not evaluation:
            logger.error(f"Evaluation {evaluation_id} not found")
            return f"Evaluation {evaluation_id} not found"

        # Check if the evaluation is already in progress or completed
        if evaluation.status not in (EvaluationStatus.PENDING, EvaluationStatus.RUNNING):
            logger.warning(
                f"Evaluation {evaluation_id} is already in {evaluation.status} status"
            )
            return f"Evaluation {evaluation_id} is already in {evaluation.status} status"

        # Start the evaluation
        if evaluation.status == EvaluationStatus.PENDING:
            evaluation = await evaluation_service.start_evaluation(evaluation_id)

        try:
            # Get the appropriate evaluation method handler
            method_handler = await evaluation_service.get_evaluation_method_handler(
                evaluation.method
            )

            # Log JWT token availability
            if jwt_token:
                logger.info(f"Running evaluation {evaluation_id} with JWT token provided")
            else:
                logger.info(f"Running evaluation {evaluation_id} without JWT token")

            # Run the evaluation with batch processing, passing the JWT token
            batch_size = evaluation.config.get("batch_size", 10) if evaluation.config else 10
            logger.info(f"Processing evaluation {evaluation_id} with batch size {batch_size}")
            results = await method_handler.run_evaluation(evaluation, jwt_token=jwt_token)

            # Process results
            for result_data in results:
                await evaluation_service.create_evaluation_result(result_data)

            # Mark the evaluation as completed
            await evaluation_service.complete_evaluation(evaluation_id, success=True)

            logger.info(f"Evaluation {evaluation_id} completed successfully with {len(results)} results")
            return f"Evaluation {evaluation_id} completed successfully with {len(results)} results"

        except Exception as e:
            # Log the error
            logger.exception(f"Error running evaluation {evaluation_id}: {e}")

            # Mark the evaluation as failed
            await evaluation_service.complete_evaluation(evaluation_id, success=False)

            return f"Error running evaluation {evaluation_id}: {str(e)}"


@worker_ready.connect
def at_start(sender, **kwargs):
    """
    Function to run when the worker starts.
    """
    logger.info("Evaluation worker is ready!")