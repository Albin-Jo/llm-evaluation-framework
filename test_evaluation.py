# File: debug_evaluation.py
import asyncio
import logging
import os
import sys
import uuid
import traceback
from datetime import datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("debug")



# Import project modules - adjust paths if needed
from backend.app.core.config import settings
from backend.app.db.models.orm.models import User, UserRole, EvaluationStatus
from backend.app.utils.sample_dataset import SampleEvaluationBuilder
from backend.app.services.evaluation_service import EvaluationService
from backend.app.evaluation.methods.ragas import RagasEvaluationMethod

# Ensure storage directory exists with absolute path
storage_path = Path(os.path.abspath("storage/datasets"))
os.makedirs(storage_path, exist_ok=True)
logger.info(f"Using storage path: {storage_path}")

# Override settings storage path to use absolute path
settings.STORAGE_LOCAL_PATH = str(storage_path.parent)
logger.info(f"Set STORAGE_LOCAL_PATH to: {settings.STORAGE_LOCAL_PATH}")


class DirectEvaluationService(EvaluationService):
    """Evaluation service that runs evaluations directly without Celery."""

    async def queue_evaluation_job(self, evaluation_id):
        """Run the evaluation directly instead of using Celery."""
        logger.info(f"Starting direct evaluation for {evaluation_id}")

        # Start the evaluation
        await self.start_evaluation(evaluation_id)

        # Run directly and wait for it to complete
        await self._run_evaluation_directly(evaluation_id)

        logger.info(f"Direct evaluation completed for {evaluation_id}")


async def run_debug_test():
    """Run a debug test of the evaluation system."""
    try:
        logger.info("Starting debug test")

        # Create database connection
        logger.info(f"Connecting to database: {settings.get_masked_db_uri()}")
        engine = create_async_engine(settings.DB_URI)
        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        # Create a test session
        async with async_session() as session:
            # Create a test user
            from backend.app.db.repositories.base import BaseRepository
            user_repo = BaseRepository(User, session)

            # Check if we can create a user to test database connection
            test_user = await user_repo.create({
                "id": uuid.uuid4(),
                "external_id": f"test-user-{uuid.uuid4().hex[:8]}",
                "email": f"test-debug-{uuid.uuid4().hex[:8]}@example.com",
                "display_name": "Debug Test User",
                "role": UserRole.ADMIN,
                "is_active": True
            })
            await session.commit()

            logger.info(f"Created test user: {test_user.id}")

            # First, directly test the RAGAS method to see if it's working
            logger.info("Testing RAGAS method directly...")
            ragas_method = RagasEvaluationMethod(session)

            # Test RAGAS availability
            has_ragas = ragas_method._check_ragas_available()
            logger.info(f"RAGAS library available: {has_ragas}")

            # Test metric calculation
            test_metrics = await ragas_method.calculate_metrics(
                input_data={"query": "What is Python?", "context": "Python is a programming language."},
                output_data={"answer": "Python is a programming language."},
                config={"metrics": ["faithfulness", "answer_relevancy", "context_relevancy"]}
            )
            logger.info(f"Test metrics calculation: {test_metrics}")

            # If metrics seem to be working, try the full evaluation
            if test_metrics:
                logger.info("Metrics calculation successful, proceeding with full evaluation test")

                # Create a sample evaluation setup with minimal data for faster testing
                logger.info("Creating sample evaluation setup...")
                evaluation, dataset, prompt, microagent = await SampleEvaluationBuilder.create_sample_evaluation(
                    db_session=session,
                    user=test_user,
                    method="ragas",
                    num_samples=1,  # Just one sample for debugging
                    domain="general"
                )
                await session.commit()

                logger.info(f"Created evaluation: {evaluation.id}")
                logger.info(f"Using dataset: {dataset.name} ({dataset.id})")
                logger.info(f"Using prompt: {prompt.name} ({prompt.id})")
                logger.info(f"Using micro-agent: {microagent.name} ({microagent.id})")

                # Create direct evaluation service to avoid Celery
                logger.info("Creating direct evaluation service...")
                evaluation_service = DirectEvaluationService(session)

                # Try to load the dataset manually to check if it's accessible
                logger.info("Testing dataset loading...")
                try:
                    dataset_items = await ragas_method.load_dataset(dataset)
                    logger.info(f"Successfully loaded dataset with {len(dataset_items)} items")
                    logger.info(f"Sample dataset item: {dataset_items[0]}")
                except Exception as e:
                    logger.error(f"Error loading dataset: {str(e)}")
                    logger.error(traceback.format_exc())

                # Try to process a single item manually
                if dataset_items:
                    logger.info("Testing single item processing...")
                    try:
                        item = dataset_items[0]
                        formatted_prompt = ragas_method._format_prompt(prompt.content, item)
                        logger.info(f"Formatted prompt: {formatted_prompt}")

                        logger.info("Calling micro-agent API...")
                        response = await ragas_method._call_microagent_api(
                            microagent.api_endpoint,
                            {
                                "prompt": formatted_prompt,
                                "query": item.get("query", ""),
                                "context": item.get("context", "")
                            }
                        )
                        logger.info(f"Micro-agent response: {response}")

                        item_metrics = await ragas_method.calculate_metrics(
                            input_data={
                                "query": item.get("query", ""),
                                "context": item.get("context", ""),
                                "ground_truth": item.get("ground_truth", "")
                            },
                            output_data={"answer": response.get("answer", "")},
                            config=evaluation.config or {}
                        )
                        logger.info(f"Item metrics: {item_metrics}")
                    except Exception as e:
                        logger.error(f"Error processing item: {str(e)}")
                        logger.error(traceback.format_exc())

                # Run the full evaluation
                logger.info("Starting full evaluation...")
                try:
                    await evaluation_service.queue_evaluation_job(evaluation.id)

                    # Wait for the evaluation to complete with timeout
                    max_wait_time = 60  # seconds
                    wait_interval = 5  # seconds
                    elapsed_time = 0

                    while elapsed_time < max_wait_time:
                        # Check evaluation status
                        updated_evaluation = await evaluation_service.get_evaluation(evaluation.id)
                        logger.info(f"Evaluation status: {updated_evaluation.status} (waited {elapsed_time}s)")

                        # If completed or failed, break the loop
                        if updated_evaluation.status in [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED]:
                            break

                        # Wait before checking again
                        await asyncio.sleep(wait_interval)
                        elapsed_time += wait_interval

                    # Check if we timed out
                    if elapsed_time >= max_wait_time:
                        logger.warning("WARNING: Evaluation didn't complete within the timeout period")

                    # Get results regardless of status
                    results = await evaluation_service.get_evaluation_results(evaluation.id)
                    logger.info(f"Got {len(results)} results")

                    if results:
                        # Print first result details
                        result = results[0]
                        logger.info("\nSample result:")
                        logger.info(f"  Query: {result.input_data.get('query')}")
                        logger.info(f"  Answer: {result.output_data.get('answer')}")
                        logger.info(f"  Overall score: {result.overall_score}")

                        # Get metrics for this result
                        metrics = await evaluation_service.get_metric_scores(result.id)
                        logger.info("\nMetrics:")
                        for metric in metrics:
                            logger.info(f"  {metric.name}: {metric.value}")
                    else:
                        logger.warning("No results found for this evaluation")

                    # Get evaluation statistics
                    stats = await evaluation_service.get_evaluation_statistics(evaluation.id)
                    logger.info("\nEvaluation statistics:")
                    logger.info(f"  Total samples: {stats['total_samples']}")
                    logger.info(f"  Average overall score: {stats['avg_overall_score']}")
                    logger.info("  Metrics:")
                    for metric_name, metric_data in stats.get('metrics', {}).items():
                        logger.info(
                            f"    {metric_name}: avg={metric_data['avg']}, min={metric_data['min']}, max={metric_data['max']}")

                except Exception as e:
                    logger.error(f"Error in evaluation process: {str(e)}")
                    logger.error(traceback.format_exc())
            else:
                logger.error("Metrics calculation failed, cannot proceed with evaluation test")

    except Exception as e:
        logger.error(f"Error in debug test: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(run_debug_test())