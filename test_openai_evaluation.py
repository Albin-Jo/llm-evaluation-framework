# File: test_openai_evaluation.py
import asyncio
import os
import uuid
from pathlib import Path
import logging

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project modules
from backend.app.core.config import settings
from backend.app.db.models.orm import User, UserRole, EvaluationStatus
from backend.app.utils.sample_dataset import SampleEvaluationBuilder
from backend.app.services.evaluation_service import EvaluationService


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


async def run_openai_test():
    """Run a test using OpenAI API directly."""
    # Ensure storage directory exists with absolute path
    storage_path = Path(os.path.abspath("storage/datasets"))
    os.makedirs(storage_path, exist_ok=True)
    logger.info(f"Using storage path: {storage_path}")

    # Override settings storage path to use absolute path
    settings.STORAGE_LOCAL_PATH = str(storage_path.parent)

    # Create database connection
    engine = create_async_engine(settings.DB_URI)
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    # Create a test session
    async with async_session() as session:
        # Create a test user
        from backend.app.db.repositories.base import BaseRepository
        user_repo = BaseRepository(User, session)

        test_user = await user_repo.create({
            "id": uuid.uuid4(),
            "external_id": f"test-user-{uuid.uuid4().hex[:8]}",
            "email": f"test-{uuid.uuid4().hex[:8]}@example.com",
            "display_name": "Test User",
            "role": UserRole.ADMIN,
            "is_active": True
        })
        await session.commit()

        logger.info(f"Created test user: {test_user.id}")

        # Create the microagent service for direct OpenAI calls
        microagent_service = MicroAgentService()

        # Test direct OpenAI call
        logger.info("Testing direct OpenAI call...")
        response = await microagent_service.query_openai(
            prompt="You are a helpful assistant. Answer based on the context.",
            query="What is Python?",
            context="Python is a high-level programming language known for its readability."
        )
        logger.info(f"OpenAI response: {response.answer}")

        # Create a sample micro-agent that uses direct OpenAI calls
        agent_repo = BaseRepository(MicroAgent, session)

        # First, check if we already have a test agent
        existing_agents = await agent_repo.get_multi(filters={"name": "OpenAI Test Agent"})
        if existing_agents:
            direct_agent = existing_agents[0]
            logger.info(f"Using existing agent: {direct_agent.id}")
        else:
            # Create a new agent with special config that our evaluation service will recognize
            direct_agent = await agent_repo.create({
                "name": "OpenAI Test Agent",
                "description": "Test agent using direct OpenAI API",
                "api_endpoint": "direct_openai",  # Special marker for direct OpenAI
                "domain": "general",
                "config": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",  # Using a cheaper model for testing
                    "temperature": 0.0,
                    "max_tokens": 200
                },
                "is_active": True
            })
            await session.commit()
            logger.info(f"Created direct OpenAI agent: {direct_agent.id}")

        # Create sample evaluation with our direct agent
        logger.info("Creating sample evaluation...")
        evaluation, dataset, prompt, _ = await SampleEvaluationBuilder.create_sample_evaluation(
            db_session=session,
            user=test_user,
            method="ragas",
            num_samples=1,  # Just one sample for faster testing
            domain="general"
        )

        # Update the evaluation to use our direct agent
        from backend.app.db.repositories.base import BaseRepository
        from backend.app.db.models.orm import Evaluation
        eval_repo = BaseRepository(Evaluation, session)
        await eval_repo.update(evaluation.id, {"micro_agent_id": direct_agent.id})
        await session.commit()

        logger.info(f"Created evaluation: {evaluation.id} with direct OpenAI agent")

        # Run the evaluation using direct service
        evaluation_service = DirectEvaluationService(session)

        # Override the method to handle direct OpenAI calls
        original_method = evaluation_service._run_evaluation_directly

        async def patched_run_directly(evaluation_id):
            """Patched method to handle direct OpenAI calls."""
            from backend.app.evaluation.methods.base import BaseEvaluationMethod

            # Override the call_microagent_api method of BaseEvaluationMethod
            original_call = BaseEvaluationMethod._call_microagent_api

            async def patched_call(self, api_endpoint, payload):
                """Patched method to use direct OpenAI for 'direct_openai' endpoints."""
                if api_endpoint == "direct_openai":
                    logger.info("Using direct OpenAI call instead of API endpoint")
                    response = await microagent_service.query_openai(
                        prompt=payload.get("prompt", ""),
                        query=payload.get("query", ""),
                        context=payload.get("context", "")
                    )
                    return {
                        "answer": response.answer,
                        "processing_time_ms": response.processing_time_ms
                    }
                else:
                    return await original_call(self, api_endpoint, payload)

            # Apply the patch
            BaseEvaluationMethod._call_microagent_api = patched_call

            # Run the original method
            try:
                await original_method(evaluation_id)
            finally:
                # Restore the original method
                BaseEvaluationMethod._call_microagent_api = original_call

        # Replace the method
        evaluation_service._run_evaluation_directly = patched_run_directly

        # Run the evaluation
        logger.info("Starting evaluation...")
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

            # Get evaluation statistics
            stats = await evaluation_service.get_evaluation_statistics(evaluation.id)
            logger.info("\nEvaluation statistics:")
            logger.info(f"  Total samples: {stats['total_samples']}")
            logger.info(f"  Average overall score: {stats['avg_overall_score']}")
            logger.info("  Metrics:")
            for metric_name, metric_data in stats.get('metrics', {}).items():
                logger.info(
                    f"    {metric_name}: avg={metric_data['avg']}, min={metric_data['min']}, max={metric_data['max']}")
        else:
            logger.warning("No results found for this evaluation")


if __name__ == "__main__":
    asyncio.run(run_openai_test())