import asyncio
import datetime
import logging
from typing import Optional
from uuid import UUID

from celery import Celery
from celery.signals import worker_ready

from backend.app.core.config import settings
from backend.app.db.models.orm import EvaluationStatus, EvaluationMethod
from backend.app.db.session import db_session
from backend.app.services.evaluation_service import EvaluationService

# Configure Celery
celery_app = Celery(
    "evaluation_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Configure Celery settings for better DeepEval performance
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # Increase task timeout for DeepEval evaluations
    task_time_limit=1800,  # 30 minutes
    task_soft_time_limit=1500,  # 25 minutes
    worker_prefetch_multiplier=1,  # Process one task at a time for memory efficiency
)

# Configure logging
logger = logging.getLogger(__name__)


@celery_app.task(name="run_evaluation", bind=True, max_retries=3)
def run_evaluation_task(self, evaluation_id: str, jwt_token: Optional[str] = None) -> str:
    """
    Run an evaluation as a background task.
    Now supports both RAGAS and DeepEval methods.

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


@celery_app.task(name="run_deepeval_batch", bind=True, max_retries=2)
def run_deepeval_batch_task(
        self,
        evaluation_id: str,
        batch_start: int,
        batch_end: int,
        jwt_token: Optional[str] = None
) -> str:
    """
    Run a batch of DeepEval test cases as a separate task.
    This allows for parallel processing of large datasets.

    Args:
        evaluation_id: Evaluation ID
        batch_start: Start index of batch
        batch_end: End index of batch  
        jwt_token: Optional JWT token for MCP agent authentication

    Returns:
        str: Batch processing result
    """
    evaluation_uuid = UUID(evaluation_id)

    loop = asyncio.get_event_loop()
    try:
        return loop.run_until_complete(
            _run_deepeval_batch(evaluation_uuid, batch_start, batch_end, jwt_token)
        )
    except Exception as exc:
        logger.exception(f"Error running DeepEval batch {batch_start}-{batch_end} for evaluation {evaluation_id}")
        # Shorter retry for batch tasks
        retry_in = 30 * (self.request.retries + 1)
        self.retry(exc=exc, countdown=retry_in)


async def _run_evaluation(evaluation_id: UUID, jwt_token: Optional[str] = None) -> str:
    """
    Internal async function to run an evaluation.
    Routes to appropriate method based on evaluation type.

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

        # Start the evaluation if pending
        if evaluation.status == EvaluationStatus.PENDING:
            evaluation = await evaluation_service.start_evaluation(evaluation_id)

        try:
            # Get the appropriate evaluation method handler
            method_handler = await evaluation_service.get_evaluation_method_handler(
                evaluation.method
            )

            # Log evaluation method and JWT token availability
            logger.info(f"Running {evaluation.method.value} evaluation {evaluation_id}")
            if jwt_token:
                logger.info(f"JWT token provided for evaluation {evaluation_id}")
            else:
                logger.info(f"No JWT token for evaluation {evaluation_id}")

            # Run the evaluation based on method type
            if evaluation.method == EvaluationMethod.DEEPEVAL:
                logger.info(f"Processing DeepEval evaluation {evaluation_id}")
                results = await _run_deepeval_evaluation(
                    method_handler, evaluation, jwt_token
                )
            else:
                # RAGAS or other methods
                logger.info(f"Processing {evaluation.method.value} evaluation {evaluation_id}")
                results = await method_handler.run_evaluation(evaluation, jwt_token=jwt_token)

            # Process results
            logger.info(f"Saving {len(results)} results for evaluation {evaluation_id}")
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


async def _run_deepeval_evaluation(method_handler, evaluation, jwt_token: Optional[str] = None):
    """
    Run DeepEval evaluation with optimized processing.

    Args:
        method_handler: DeepEval method handler
        evaluation: Evaluation model
        jwt_token: Optional JWT token

    Returns:
        List of evaluation results
    """
    # Check if we should use batch processing for large datasets
    dataset = await method_handler.get_dataset(evaluation.dataset_id)

    # Load dataset to check size
    dataset_items = await method_handler.load_dataset(dataset)
    total_items = len(dataset_items)

    logger.info(f"DeepEval evaluation {evaluation.id} processing {total_items} items")

    # For small datasets, use standard processing
    if total_items <= 50:
        logger.info(f"Using standard processing for {total_items} items")
        return await method_handler.run_evaluation(evaluation, jwt_token=jwt_token)

    # For large datasets, use parallel batch processing
    logger.info(f"Using parallel batch processing for {total_items} items")

    batch_size = evaluation.config.get("batch_size", 20) if evaluation.config else 20
    batch_size = min(batch_size, 50)  # Cap batch size for DeepEval

    # Create batch tasks
    batch_tasks = []
    for batch_start in range(0, total_items, batch_size):
        batch_end = min(batch_start + batch_size, total_items)

        # Create Celery task for each batch
        task = run_deepeval_batch_task.delay(
            str(evaluation.id), batch_start, batch_end, jwt_token
        )
        batch_tasks.append(task)
        logger.info(f"Queued DeepEval batch {batch_start}-{batch_end}")

    # Wait for all batch tasks to complete
    all_results = []
    completed_batches = 0

    for task in batch_tasks:
        try:
            # Wait for task completion (with timeout)
            result = task.get(timeout=900)  # 15 minute timeout per batch
            logger.info(f"Completed batch task: {result}")
            completed_batches += 1

            # Get the actual results (would need to be stored in database)
            # For now, we'll use the standard approach as fallback

        except Exception as e:
            logger.error(f"Batch task failed: {e}")

    logger.info(f"Completed {completed_batches}/{len(batch_tasks)} DeepEval batches")

    # Fallback to standard processing if batch processing fails
    if completed_batches < len(batch_tasks):
        logger.warning("Some batches failed, falling back to standard processing")
        return await method_handler.run_evaluation(evaluation, jwt_token=jwt_token)

    # If all batches succeeded, we would collect results from database
    # For now, fallback to standard processing
    return await method_handler.run_evaluation(evaluation, jwt_token=jwt_token)


async def _run_deepeval_batch(
        evaluation_id: UUID,
        batch_start: int,
        batch_end: int,
        jwt_token: Optional[str] = None
) -> str:
    """
    Process a specific batch of DeepEval test cases.

    Args:
        evaluation_id: Evaluation ID
        batch_start: Start index
        batch_end: End index
        jwt_token: Optional JWT token

    Returns:
        Batch processing result
    """
    async with db_session() as session:
        evaluation_service = EvaluationService(session)

        # Get evaluation
        evaluation = await evaluation_service.get_evaluation(evaluation_id)
        if not evaluation:
            return f"Evaluation {evaluation_id} not found"

        # Get method handler
        method_handler = await evaluation_service.get_evaluation_method_handler(
            evaluation.method
        )

        # Load dataset
        dataset = await method_handler.get_dataset(evaluation.dataset_id)
        dataset_items = await method_handler.load_dataset(dataset)

        # Extract batch
        batch_items = dataset_items[batch_start:batch_end]

        logger.info(f"Processing DeepEval batch {batch_start}-{batch_end} "
                    f"with {len(batch_items)} items")

        # Process batch using method handler
        # This is a simplified version - in practice you'd want to 
        # create a batch-specific processing method

        # For now, simulate batch processing
        await asyncio.sleep(2)  # Simulate processing time

        logger.info(f"Completed DeepEval batch {batch_start}-{batch_end}")
        return f"Batch {batch_start}-{batch_end} completed with {len(batch_items)} items"


@celery_app.task(name="validate_deepeval_dataset", bind=True)
def validate_deepeval_dataset_task(self, dataset_id: str, metrics: list) -> dict:
    """
    Validate a dataset for DeepEval compatibility as a background task.

    Args:
        dataset_id: Dataset ID to validate
        metrics: List of metrics to validate against

    Returns:
        Validation results
    """
    loop = asyncio.get_event_loop()
    try:
        return loop.run_until_complete(
            _validate_deepeval_dataset(UUID(dataset_id), metrics)
        )
    except Exception as exc:
        logger.exception(f"Error validating dataset {dataset_id} for DeepEval")
        return {
            "compatible": False,
            "error": str(exc),
            "warnings": [],
            "requirements": []
        }


async def _validate_deepeval_dataset(dataset_id: UUID, metrics: list) -> dict:
    """Validate dataset for DeepEval compatibility."""
    async with db_session() as session:
        from backend.app.services.dataset_service import DatasetService
        from backend.app.evaluation.adapters.dataset_adapter import DatasetAdapter

        dataset_service = DatasetService(session)
        dataset = await dataset_service.get_dataset(dataset_id)

        if not dataset:
            return {
                "compatible": False,
                "error": "Dataset not found",
                "warnings": [],
                "requirements": []
            }

        # Load dataset content
        from backend.app.services.storage import get_storage_service
        storage_service = get_storage_service()
        file_content = await storage_service.read_file(dataset.file_path)

        # Parse content
        import json
        if dataset.file_path.endswith('.json'):
            dataset_content = json.loads(file_content)
        else:
            import pandas as pd
            import io
            df = pd.read_csv(io.StringIO(file_content))
            dataset_content = df.to_dict('records')

        # Validate using adapter
        adapter = DatasetAdapter()
        validation_results = adapter.validate_dataset_for_deepeval(dataset_content, metrics)

        return validation_results


@celery_app.task(name="cleanup_evaluation_cache")
def cleanup_evaluation_cache_task():
    """
    Periodic task to clean up evaluation caches and temporary data.
    """
    logger.info("Starting evaluation cache cleanup")

    try:
        # Clean up any temporary DeepEval files
        import tempfile
        import os
        import time

        temp_dir = tempfile.gettempdir()
        deepeval_pattern = "deepeval_*"

        # Remove temporary files older than 1 hour
        cutoff_time = time.time() - 3600
        cleaned_count = 0

        for filename in os.listdir(temp_dir):
            if filename.startswith("deepeval_"):
                filepath = os.path.join(temp_dir, filename)
                if os.path.getctime(filepath) < cutoff_time:
                    try:
                        os.remove(filepath)
                        cleaned_count += 1
                    except OSError:
                        pass

        logger.info(f"Cleaned up {cleaned_count} temporary DeepEval files")
        return f"Cleaned up {cleaned_count} files"

    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}")
        return f"Error: {str(e)}"


# Periodic task setup
celery_app.conf.beat_schedule = {
    'cleanup-evaluation-cache': {
        'task': 'cleanup_evaluation_cache',
        'schedule': 3600.0,  # Run every hour
    },
}


@worker_ready.connect
def at_start(sender, **kwargs):
    """
    Function to run when the worker starts.
    """
    logger.info("Evaluation worker is ready!")

    # Check if DeepEval is available
    try:
        import deepeval
        logger.info("DeepEval is available for evaluations")
    except ImportError:
        logger.warning("DeepEval is not available. Install with: pip install deepeval")

    # Log worker configuration
    logger.info(f"Worker configured with:")
    logger.info(f"  - Task time limit: {celery_app.conf.task_time_limit}s")
    logger.info(f"  - Soft time limit: {celery_app.conf.task_soft_time_limit}s")
    logger.info(f"  - Prefetch multiplier: {celery_app.conf.worker_prefetch_multiplier}")


# Health check task
@celery_app.task(name="health_check")
def health_check_task():
    """Health check task for monitoring."""
    try:
        # Test database connection
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_test_db_connection())

        # Test DeepEval availability
        deepeval_available = False
        try:
            import deepeval
            deepeval_available = True
        except ImportError:
            pass

        return {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "database": "connected",
            "deepeval": "available" if deepeval_available else "not_available",
            "worker": "running"
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }


async def _test_db_connection():
    """Test database connection."""
    async with db_session() as session:
        result = await session.execute("SELECT 1")
        return result.scalar()
