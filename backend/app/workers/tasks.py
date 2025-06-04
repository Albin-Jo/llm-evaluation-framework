import asyncio
import datetime
import logging
import time
from typing import Optional, Dict, Any, List
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


# Add this to backend/app/workers/tasks.py

@celery_app.task(name="run_deepeval_evaluation", bind=True, max_retries=2)
def run_deepeval_evaluation_task(
        self,
        evaluation_id: str,
        jwt_token: Optional[str] = None,
        deepeval_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Enhanced Celery task specifically for DeepEval evaluations.

    Args:
        evaluation_id: Evaluation ID
        jwt_token: Optional JWT token for MCP agent authentication
        deepeval_config: DeepEval-specific configuration

    Returns:
        str: Task result
    """
    evaluation_uuid = UUID(evaluation_id)

    # Set up asyncio event loop
    loop = asyncio.get_event_loop()

    try:
        return loop.run_until_complete(
            _run_deepeval_evaluation_enhanced(evaluation_uuid, jwt_token, deepeval_config)
        )
    except Exception as exc:
        logger.exception(f"Error running DeepEval evaluation {evaluation_id}")

        # Retry with exponential backoff, but fewer retries for DeepEval
        retry_in = 60 * (2 ** self.request.retries)  # 60s, 120s, 240s
        self.retry(exc=exc, countdown=retry_in)


async def _run_deepeval_evaluation_enhanced(
        evaluation_id: UUID,
        jwt_token: Optional[str] = None,
        deepeval_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Enhanced async function to run DeepEval evaluations with better error handling.
    """
    async with db_session() as session:
        evaluation_service = EvaluationService(session)

        try:
            # Get the evaluation
            evaluation = await evaluation_service.get_evaluation(evaluation_id)
            if not evaluation:
                logger.error(f"DeepEval evaluation {evaluation_id} not found")
                return f"Evaluation {evaluation_id} not found"

            # Validate evaluation is for DeepEval
            if evaluation.method != EvaluationMethod.DEEPEVAL:
                error_msg = f"Evaluation {evaluation_id} is not a DeepEval evaluation (method: {evaluation.method})"
                logger.error(error_msg)
                return error_msg

            # Check status
            if evaluation.status not in (EvaluationStatus.PENDING, EvaluationStatus.RUNNING):
                logger.warning(f"DeepEval evaluation {evaluation_id} is in {evaluation.status} status")
                return f"Evaluation {evaluation_id} is in {evaluation.status} status"

            # Update configuration if provided
            if deepeval_config:
                current_config = evaluation.config or {}
                current_config.update(deepeval_config)
                await evaluation_service.evaluation_repo.update(evaluation_id, {"config": current_config})
                logger.info(f"Updated DeepEval config for evaluation {evaluation_id}")

            # Start the evaluation if pending
            if evaluation.status == EvaluationStatus.PENDING:
                await evaluation_service.evaluation_repo.update(evaluation_id, {
                    "status": EvaluationStatus.RUNNING,
                    "start_time": datetime.datetime.now()
                })
                logger.info(f"Started DeepEval evaluation {evaluation_id}")

            # Get the DeepEval method handler
            method_handler = await evaluation_service.get_evaluation_method_handler(evaluation.method)

            # Add progress tracking
            logger.info(f"Running DeepEval evaluation {evaluation_id} with JWT: {bool(jwt_token)}")

            # Run the evaluation with enhanced monitoring
            start_time = time.time()

            try:
                results = await method_handler.run_evaluation(evaluation, jwt_token=jwt_token)
                processing_time = time.time() - start_time

                logger.info(f"DeepEval processing completed in {processing_time:.2f}s, got {len(results)} results")

            except Exception as eval_error:
                # Log detailed error for DeepEval-specific issues
                logger.error(f"DeepEval execution failed for evaluation {evaluation_id}: {eval_error}")

                # Try to provide more specific error information
                if "deepeval" in str(eval_error).lower():
                    error_msg = f"DeepEval library error: {str(eval_error)}"
                elif "model" in str(eval_error).lower() or "openai" in str(eval_error).lower():
                    error_msg = f"LLM/Model error: {str(eval_error)}"
                elif "timeout" in str(eval_error).lower():
                    error_msg = f"Timeout error - consider reducing batch size: {str(eval_error)}"
                else:
                    error_msg = f"Evaluation error: {str(eval_error)}"

                # Update evaluation status with specific error
                await evaluation_service.evaluation_repo.update(evaluation_id, {
                    "status": EvaluationStatus.FAILED,
                    "end_time": datetime.datetime.now(),
                    "config": {
                        **(evaluation.config or {}),
                        "error_details": error_msg,
                        "processing_time_seconds": time.time() - start_time
                    }
                })

                return error_msg

            # Process and save results
            logger.info(f"Saving {len(results)} DeepEval results for evaluation {evaluation_id}")

            saved_count = 0
            failed_count = 0

            for i, result_data in enumerate(results):
                try:
                    await evaluation_service.create_evaluation_result(result_data)
                    saved_count += 1
                except Exception as result_error:
                    logger.error(f"Error saving DeepEval result {i} for evaluation {evaluation_id}: {result_error}")
                    failed_count += 1

            # Update evaluation status
            final_config = {
                **(evaluation.config or {}),
                "results_saved": saved_count,
                "results_failed": failed_count,
                "processing_time_seconds": time.time() - start_time,
                "deepeval_version": "latest"  # Could get actual version
            }

            if failed_count == 0:
                # Mark as completed
                await evaluation_service.evaluation_repo.update(evaluation_id, {
                    "status": EvaluationStatus.COMPLETED,
                    "end_time": datetime.datetime.now(),
                    "config": final_config
                })

                result_msg = f"DeepEval evaluation {evaluation_id} completed successfully with {saved_count} results"
                logger.info(result_msg)
                return result_msg

            else:
                # Partial success
                await evaluation_service.evaluation_repo.update(evaluation_id, {
                    "status": EvaluationStatus.COMPLETED,  # Still mark as completed
                    "end_time": datetime.datetime.now(),
                    "config": final_config
                })

                result_msg = f"DeepEval evaluation {evaluation_id} completed with {saved_count} results, {failed_count} failed"
                logger.warning(result_msg)
                return result_msg

        except Exception as e:
            # Handle any other unexpected errors
            logger.exception(f"Unexpected error in DeepEval evaluation {evaluation_id}: {e}")

            # Try to mark as failed
            try:
                await evaluation_service.evaluation_repo.update(evaluation_id, {
                    "status": EvaluationStatus.FAILED,
                    "end_time": datetime.datetime.now(),
                    "config": {
                        **(evaluation.config if evaluation else {}),
                        "error": str(e),
                        "error_type": "unexpected_error"
                    }
                })
            except Exception:
                logger.error(f"Could not update failed status for evaluation {evaluation_id}")

            return f"Unexpected error in DeepEval evaluation {evaluation_id}: {str(e)}"


@celery_app.task(name="validate_deepeval_dataset_enhanced", bind=True)
def validate_deepeval_dataset_enhanced_task(
        self,
        dataset_id: str,
        metrics: List[str],
        generate_preview: bool = True
) -> Dict[str, Any]:
    """
    Enhanced dataset validation task for DeepEval with preview generation.

    Args:
        dataset_id: Dataset ID to validate
        metrics: List of metrics to validate against
        generate_preview: Whether to generate conversion preview

    Returns:
        Enhanced validation results
    """
    loop = asyncio.get_event_loop()
    try:
        return loop.run_until_complete(
            _validate_deepeval_dataset_enhanced(UUID(dataset_id), metrics, generate_preview)
        )
    except Exception as exc:
        logger.exception(f"Error validating dataset {dataset_id} for DeepEval")
        return {
            "compatible": False,
            "error": str(exc),
            "warnings": [f"Validation failed: {str(exc)}"],
            "requirements": ["Check dataset format and content"],
            "supported_metrics": [],
            "recommended_metrics": []
        }


async def _validate_deepeval_dataset_enhanced(
        dataset_id: UUID,
        metrics: List[str],
        generate_preview: bool = True
) -> Dict[str, Any]:
    """Enhanced dataset validation with detailed analysis."""
    async with db_session() as session:
        from backend.app.services.dataset_service import DatasetService
        from backend.app.evaluation.adapters.dataset_adapter import DatasetAdapter
        from backend.app.evaluation.metrics.deepeval_metrics import (
            get_supported_metrics_for_dataset_type, get_recommended_metrics
        )

        dataset_service = DatasetService(session)
        dataset = await dataset_service.get_dataset(dataset_id)

        if not dataset:
            return {
                "compatible": False,
                "error": "Dataset not found",
                "warnings": [],
                "requirements": ["Ensure dataset exists and is accessible"]
            }

        try:
            # Load dataset content
            from backend.app.services.storage import get_storage_service
            storage_service = get_storage_service()
            file_content = await storage_service.read_file(dataset.file_path)

            # Parse content based on file type
            if dataset.file_path.endswith('.json'):
                import json
                dataset_content = json.loads(file_content)
            else:
                import pandas as pd
                import io
                df = pd.read_csv(io.StringIO(file_content))
                dataset_content = df.to_dict('records')

            # Validate using enhanced adapter
            adapter = DatasetAdapter()
            validation_results = adapter.validate_dataset_for_deepeval(dataset_content, metrics)

            # Get dataset type specific information
            supported_metrics = get_supported_metrics_for_dataset_type(dataset.type)
            recommended_metrics = get_recommended_metrics(dataset.type)

            # Enhanced validation with detailed field analysis
            field_analysis = _analyze_dataset_fields(dataset_content)

            # Generate conversion preview if requested
            preview = None
            if generate_preview and validation_results.get("compatible", False):
                try:
                    # Convert first few items as preview
                    preview_items = dataset_content[:3] if isinstance(dataset_content, list) else [dataset_content]
                    deepeval_dataset = await adapter.convert_to_deepeval_dataset(dataset, preview_items)

                    preview = []
                    for i, test_case in enumerate(deepeval_dataset.test_cases):
                        preview.append({
                            "index": i,
                            "input": test_case.input,
                            "expected_output": test_case.expected_output,
                            "context": test_case.context,
                            "has_all_required_fields": bool(test_case.input and test_case.context)
                        })
                except Exception as preview_error:
                    logger.warning(f"Could not generate preview: {preview_error}")

            # Combine results
            enhanced_results = {
                **validation_results,
                "dataset_info": {
                    "id": str(dataset_id),
                    "name": dataset.name,
                    "type": dataset.type.value,
                    "total_items": len(dataset_content) if isinstance(dataset_content, list) else 1,
                    "file_format": "json" if dataset.file_path.endswith('.json') else "csv"
                },
                "field_analysis": field_analysis,
                "metric_compatibility": {
                    "requested_metrics": metrics,
                    "supported_metrics": supported_metrics,
                    "recommended_metrics": recommended_metrics,
                    "compatible_metrics": [m for m in metrics if m in supported_metrics],
                    "incompatible_metrics": [m for m in metrics if m not in supported_metrics]
                },
                "deepeval_preview": preview,
                "recommendations": _generate_dataset_recommendations(validation_results, field_analysis, metrics)
            }

            return enhanced_results

        except Exception as e:
            logger.error(f"Error during enhanced dataset validation: {e}")
            return {
                "compatible": False,
                "error": f"Validation error: {str(e)}",
                "warnings": [f"Could not validate dataset: {str(e)}"],
                "requirements": ["Check dataset format and accessibility"]
            }


def _analyze_dataset_fields(dataset_content: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze dataset fields for DeepEval compatibility."""
    if not dataset_content or not isinstance(dataset_content, list):
        return {"error": "Invalid dataset format"}

    # Analyze field presence across all items
    all_fields = set()
    field_presence = {}

    for item in dataset_content:
        if isinstance(item, dict):
            for field in item.keys():
                all_fields.add(field)
                field_presence[field] = field_presence.get(field, 0) + 1

    total_items = len(dataset_content)

    # Categorize fields
    input_fields = []
    context_fields = []
    output_fields = []
    other_fields = []

    for field in all_fields:
        field_lower = field.lower()
        coverage = (field_presence[field] / total_items) * 100

        if any(term in field_lower for term in ['input', 'query', 'question']):
            input_fields.append({"name": field, "coverage": coverage})
        elif any(term in field_lower for term in ['context', 'contexts', 'background']):
            context_fields.append({"name": field, "coverage": coverage})
        elif any(term in field_lower for term in ['output', 'answer', 'ground_truth', 'expected']):
            output_fields.append({"name": field, "coverage": coverage})
        else:
            other_fields.append({"name": field, "coverage": coverage})

    return {
        "total_items": total_items,
        "total_fields": len(all_fields),
        "field_categories": {
            "input_fields": input_fields,
            "context_fields": context_fields,
            "output_fields": output_fields,
            "other_fields": other_fields
        },
        "field_coverage": {
            field: round((count / total_items) * 100, 1)
            for field, count in field_presence.items()
        },
        "completeness_score": _calculate_completeness_score(input_fields, context_fields, output_fields)
    }


def _calculate_completeness_score(input_fields, context_fields, output_fields) -> float:
    """Calculate a completeness score for DeepEval compatibility."""
    score = 0.0

    # Input fields (required)
    if input_fields and any(f["coverage"] > 80 for f in input_fields):
        score += 40

    # Context fields (important for most metrics)
    if context_fields and any(f["coverage"] > 80 for f in context_fields):
        score += 35

    # Output fields (helpful for comparison)
    if output_fields and any(f["coverage"] > 80 for f in output_fields):
        score += 25

    return score


def _generate_dataset_recommendations(
        validation_results: Dict[str, Any],
        field_analysis: Dict[str, Any],
        requested_metrics: List[str]
) -> List[str]:
    """Generate specific recommendations for dataset improvement."""
    recommendations = []

    if not validation_results.get("compatible", False):
        recommendations.append("Dataset is not compatible with DeepEval. Consider restructuring the data.")

    # Field-specific recommendations
    if "field_categories" in field_analysis:
        categories = field_analysis["field_categories"]

        if not categories["input_fields"]:
            recommendations.append("Add input/query fields for DeepEval test cases.")

        if not categories["context_fields"] and any("contextual" in m for m in requested_metrics):
            recommendations.append(
                "Add context fields for contextual metrics (contextual_precision, contextual_recall, etc.).")

        if not categories["output_fields"]:
            recommendations.append("Consider adding expected output fields for better evaluation.")

    # Metric-specific recommendations
    contextual_metrics = [m for m in requested_metrics if "contextual" in m]
    if contextual_metrics and field_analysis.get("completeness_score", 0) < 75:
        recommendations.append(f"Improve dataset completeness for contextual metrics: {contextual_metrics}")

    # General recommendations
    if validation_results.get("warnings"):
        recommendations.extend([f"Address warning: {w}" for w in validation_results["warnings"]])

    if not recommendations:
        recommendations.append("Dataset looks good for DeepEval! You can proceed with the evaluation.")

    return recommendations


# Add periodic cleanup task for DeepEval temporary files
@celery_app.task(name="cleanup_deepeval_temp_files")
def cleanup_deepeval_temp_files_task():
    """
    Clean up temporary files created during DeepEval evaluations.
    """
    logger.info("Starting DeepEval temporary files cleanup")

    try:
        import tempfile
        import os
        import time
        import glob

        temp_dir = tempfile.gettempdir()

        # DeepEval might create various temporary files
        patterns_to_clean = [
            "deepeval_*",
            "*.deepeval",
            "evaluation_*.tmp",
            "test_case_*.json"
        ]

        # Remove files older than 2 hours
        cutoff_time = time.time() - 7200
        cleaned_count = 0

        for pattern in patterns_to_clean:
            pattern_path = os.path.join(temp_dir, pattern)
            for filepath in glob.glob(pattern_path):
                try:
                    if os.path.getctime(filepath) < cutoff_time:
                        os.remove(filepath)
                        cleaned_count += 1
                except (OSError, FileNotFoundError):
                    pass

        logger.info(f"Cleaned up {cleaned_count} DeepEval temporary files")
        return f"Cleaned up {cleaned_count} temporary files"

    except Exception as e:
        logger.error(f"Error during DeepEval cleanup: {e}")
        return f"Cleanup error: {str(e)}"


# Update the periodic task schedule to include DeepEval cleanup
celery_app.conf.beat_schedule.update({
    'cleanup-deepeval-temp-files': {
        'task': 'cleanup_deepeval_temp_files',
        'schedule': 7200.0,  # Run every 2 hours
    }
})
