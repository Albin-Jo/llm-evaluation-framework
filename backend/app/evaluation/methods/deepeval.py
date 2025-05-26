import asyncio
import datetime
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from backend.app.evaluation.adapters.dataset_adapter import DatasetAdapter
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import Evaluation, EvaluationStatus
from backend.app.db.schema.evaluation_schema import EvaluationResultCreate, MetricScoreCreate
from backend.app.evaluation.adapters.prompt_adapter import PromptAdapter
from backend.app.evaluation.methods.base import BaseEvaluationMethod
from backend.app.evaluation.metrics.deepeval_metrics import (
    DEEPEVAL_DATASET_TYPE_METRICS, DEEPEVAL_AVAILABLE
)

# Configure logging
logger = logging.getLogger(__name__)

# Only import deepeval if available
if DEEPEVAL_AVAILABLE:
    try:
        from deepeval import evaluate
        from deepeval.test_case import LLMTestCase
        from deepeval.dataset import EvaluationDataset
        from deepeval.metrics import (
            AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric,
            ContextualRecallMetric, ContextualRelevancyMetric, HallucinationMetric,
            ToxicityMetric, BiasMetric, GEval
        )

        logger.info("DeepEval library successfully imported")
    except ImportError as e:
        logger.error(f"Failed to import DeepEval components: {e}")
        DEEPEVAL_AVAILABLE = False


class DeepEvalMethod(BaseEvaluationMethod):
    """Evaluation method using DeepEval library for LLM evaluation."""

    method_name = "deepeval"

    def __init__(self, db_session: AsyncSession):
        """Initialize the DeepEval evaluation method."""
        super().__init__(db_session)
        self.deepeval_available = DEEPEVAL_AVAILABLE

        if not self.deepeval_available:
            raise ImportError("DeepEval library is not available. Please install it with: pip install deepeval")

        # Initialize adapters
        self.dataset_adapter = DatasetAdapter()
        self.prompt_adapter = PromptAdapter()

        logger.info(f"Initializing DeepEval evaluation method. DeepEval available: {self.deepeval_available}")

    async def run_evaluation(self, evaluation: Evaluation, jwt_token: Optional[str] = None) -> List[
        EvaluationResultCreate]:
        """
        Run evaluation using DeepEval.

        Args:
            evaluation: Evaluation model
            jwt_token: Optional JWT token for MCP agent authentication

        Returns:
            List[EvaluationResultCreate]: List of evaluation results
        """
        logger.info(f"Starting DeepEval evaluation {evaluation.id}")

        try:
            # Get related entities
            agent = await self.get_agent(evaluation.agent_id)
            dataset = await self.get_dataset(evaluation.dataset_id)
            prompt = await self.get_prompt(evaluation.prompt_id)

            if not agent or not dataset or not prompt:
                raise ValueError(f"Missing required entities for evaluation {evaluation.id}")

            # Validate metrics based on dataset type
            if not evaluation.metrics:
                # Default metrics based on dataset type
                evaluation.metrics = DEEPEVAL_DATASET_TYPE_METRICS.get(
                    dataset.type, ["answer_relevancy", "faithfulness"]
                )
                logger.info(f"Using default DeepEval metrics for {dataset.type} dataset: {evaluation.metrics}")
            else:
                # Check that selected metrics are appropriate for this dataset type
                valid_metrics = DEEPEVAL_DATASET_TYPE_METRICS.get(dataset.type, [])
                invalid_metrics = [m for m in evaluation.metrics if m not in valid_metrics]
                if invalid_metrics:
                    logger.warning(
                        f"Metrics {invalid_metrics} may not be appropriate for {dataset.type} datasets. "
                        f"Valid metrics are: {valid_metrics}"
                    )

            # Use enhanced batch processing with DeepEval integration
            results = await self.batch_process_with_deepeval(evaluation, jwt_token=jwt_token)

            logger.info(f"Completed DeepEval evaluation {evaluation.id} with {len(results)} results")
            return results

        except Exception as e:
            # Update evaluation to failed status
            await self._update_evaluation_status(
                evaluation.id,
                EvaluationStatus.FAILED,
                {"end_time": datetime.datetime.now()}
            )

            logger.exception(f"Failed DeepEval evaluation {evaluation.id}: {str(e)}")
            raise

    async def batch_process_with_deepeval(
            self,
            evaluation: Evaluation,
            batch_size: int = 10,
            jwt_token: Optional[str] = None
    ) -> List[EvaluationResultCreate]:
        """
        Process dataset items using DeepEval with your existing batch processing pattern.

        Args:
            evaluation: Evaluation model
            batch_size: Number of items to process in each batch
            jwt_token: Optional JWT token for MCP authentication

        Returns:
            List[EvaluationResultCreate]: List of evaluation results
        """
        # Get related entities
        from backend.app.db.repositories.agent_repository import AgentRepository
        from backend.app.services.agent_clients.factory import AgentClientFactory
        from backend.app.db.models.orm import IntegrationType

        agent_repo = AgentRepository(self.db_session)
        agent = await agent_repo.get_with_decrypted_credentials(evaluation.agent_id)
        dataset = await self.get_dataset(evaluation.dataset_id)
        prompt = await self.get_prompt(evaluation.prompt_id)

        if not agent or not dataset or not prompt:
            logger.error(f"Missing required entities for evaluation {evaluation.id}")
            return []

        # Load and convert dataset to DeepEval format
        dataset_items = await self.load_dataset(dataset)
        deepeval_dataset = await self.dataset_adapter.convert_to_deepeval_dataset(
            dataset, dataset_items
        )

        # Create agent client
        logger.info(f"Creating client for agent type: {agent.integration_type}")
        if agent.integration_type == IntegrationType.MCP and jwt_token:
            logger.info(f"Using JWT token for MCP agent in evaluation {evaluation.id}")
            agent_client = await AgentClientFactory.create_client(agent, jwt_token)
        else:
            agent_client = await AgentClientFactory.create_client(agent)

        # Generate responses for all test cases first
        test_cases_with_outputs = []
        for i, test_case in enumerate(deepeval_dataset.test_cases):
            try:
                # Apply prompt template to generate response
                actual_output = await self.prompt_adapter.apply_prompt_to_agent_client(
                    agent_client, prompt, test_case.input, test_case.context
                )

                # Update test case with actual output
                test_case.actual_output = actual_output
                test_cases_with_outputs.append(test_case)

                # Log progress
                if (i + 1) % 5 == 0:
                    logger.info(f"Generated {i + 1}/{len(deepeval_dataset.test_cases)} responses")

            except Exception as e:
                logger.error(f"Error generating response for test case {i}: {e}")
                # Create test case with error
                test_case.actual_output = f"Error: {str(e)}"
                test_cases_with_outputs.append(test_case)

        # Create evaluation dataset with outputs
        evaluation_dataset = EvaluationDataset(test_cases=test_cases_with_outputs)

        # Initialize DeepEval metrics
        deepeval_metrics = self._initialize_deepeval_metrics(evaluation.metrics, evaluation.config or {})

        # Run DeepEval evaluation in thread pool (since it's synchronous)
        deepeval_results = await self._run_deepeval_async(evaluation_dataset, deepeval_metrics)

        # Convert results to your format
        evaluation_results = await self._convert_deepeval_results(
            evaluation, test_cases_with_outputs, deepeval_results
        )

        return evaluation_results

    def _initialize_deepeval_metrics(self, metric_names: List[str], config: Dict[str, Any]) -> List:
        """Initialize DeepEval metrics based on configuration."""
        metrics = []

        for metric_name in metric_names:
            try:
                metric_config = config.get('deepeval_config', {}).get(metric_name, {})
                threshold = metric_config.get('threshold', 0.7)
                model = metric_config.get('model', 'gpt-4')

                if metric_name == 'answer_relevancy':
                    metrics.append(AnswerRelevancyMetric(
                        threshold=threshold,
                        model=model,
                        include_reason=True
                    ))
                elif metric_name == 'faithfulness':
                    metrics.append(FaithfulnessMetric(
                        threshold=threshold,
                        model=model,
                        include_reason=True
                    ))
                elif metric_name == 'contextual_precision':
                    metrics.append(ContextualPrecisionMetric(
                        threshold=threshold,
                        model=model,
                        include_reason=True
                    ))
                elif metric_name == 'contextual_recall':
                    metrics.append(ContextualRecallMetric(
                        threshold=threshold,
                        model=model,
                        include_reason=True
                    ))
                elif metric_name == 'contextual_relevancy':
                    metrics.append(ContextualRelevancyMetric(
                        threshold=threshold,
                        model=model,
                        include_reason=True
                    ))
                elif metric_name == 'hallucination':
                    metrics.append(HallucinationMetric(
                        threshold=threshold,
                        model=model
                    ))
                elif metric_name == 'toxicity':
                    metrics.append(ToxicityMetric(
                        threshold=threshold,
                        model=model
                    ))
                elif metric_name == 'bias':
                    metrics.append(BiasMetric(
                        threshold=threshold,
                        model=model
                    ))
                elif metric_name.startswith('g_eval_'):
                    # Custom G-Eval metric
                    g_eval_config = metric_config.get('g_eval', {})
                    metrics.append(GEval(
                        name=g_eval_config.get('name', metric_name),
                        criteria=g_eval_config.get('criteria', 'Evaluate the response quality'),
                        evaluation_steps=g_eval_config.get('evaluation_steps', [
                            "Read the input and response carefully",
                            "Evaluate based on the given criteria",
                            "Provide a score from 1-10"
                        ]),
                        evaluation_params=g_eval_config.get('evaluation_params', []),
                        threshold=threshold
                    ))
                else:
                    logger.warning(f"Unknown DeepEval metric: {metric_name}")

            except Exception as e:
                logger.error(f"Error initializing metric {metric_name}: {e}")

        return metrics

    async def _run_deepeval_async(self, dataset, metrics: List) -> Dict:
        """Run DeepEval in async context using thread pool."""
        loop = asyncio.get_event_loop()

        def run_deepeval():
            return evaluate(
                test_cases=dataset.test_cases,
                metrics=metrics,
                hyperparameters={"model": "gpt-4"}
            )

        return await loop.run_in_executor(None, run_deepeval)

    async def _convert_deepeval_results(
            self,
            evaluation: Evaluation,
            test_cases: List[LLMTestCase],
            deepeval_results: Dict
    ) -> List[EvaluationResultCreate]:
        """Convert DeepEval results to your EvaluationResult format."""
        results = []

        for i, test_case in enumerate(test_cases):
            try:
                # Extract metric scores from the test case after evaluation
                metric_scores = []
                total_score = 0
                passed_count = 0

                # Get metric results from test case
                for metric in deepeval_results.get('metrics', []):
                    if hasattr(metric, 'score') and hasattr(metric, 'success'):
                        score_value = getattr(metric, 'score', 0)
                        success = getattr(metric, 'success', False)
                        reason = getattr(metric, 'reason', '')

                        metric_scores.append(MetricScoreCreate(
                            name=metric.__name__ if hasattr(metric, '__name__') else str(metric),
                            value=score_value,
                            weight=1.0,
                            meta_info={
                                'success': success,
                                'reason': reason,
                                'threshold': getattr(metric, 'threshold', 0.7)
                            }
                        ))

                        total_score += score_value
                        if success:
                            passed_count += 1

                # Calculate overall score
                overall_score = total_score / len(metric_scores) if metric_scores else 0
                passed = passed_count == len(metric_scores) if metric_scores else False

                # Create evaluation result
                result = EvaluationResultCreate(
                    evaluation_id=evaluation.id,
                    overall_score=overall_score,
                    raw_results={
                        'deepeval_metrics': [
                            {
                                'name': ms.name,
                                'value': ms.value,
                                'success': ms.meta_info.get('success'),
                                'reason': ms.meta_info.get('reason')
                            } for ms in metric_scores
                        ]
                    },
                    dataset_sample_id=str(i),
                    input_data={
                        'input': test_case.input,
                        'context': test_case.context,
                        'expected_output': test_case.expected_output
                    },
                    output_data={
                        'actual_output': test_case.actual_output
                    },
                    metric_scores=metric_scores,
                    passed=passed,
                    pass_threshold=evaluation.pass_threshold or 0.7
                )

                results.append(result)

            except Exception as e:
                logger.error(f"Error converting result for test case {i}: {e}")
                # Create error result
                error_result = EvaluationResultCreate(
                    evaluation_id=evaluation.id,
                    overall_score=0.0,
                    raw_results={'error': str(e)},
                    dataset_sample_id=str(i),
                    input_data={'error': 'Failed to process'},
                    output_data={'error': str(e)},
                    metric_scores=[],
                    passed=False,
                    pass_threshold=evaluation.pass_threshold or 0.7
                )
                results.append(error_result)

        return results

    async def calculate_metrics(
            self, input_data: Dict[str, Any], output_data: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate DeepEval metrics for a single evaluation item.

        Note: This method is kept for compatibility but DeepEval works better
        with batch processing in run_evaluation.
        """
        logger.info("Individual metric calculation called - DeepEval works best with batch processing")

        # For individual calculation, create a single test case and evaluate
        test_case = LLMTestCase(
            input=input_data.get('query', ''),
            actual_output=output_data.get('answer', ''),
            expected_output=input_data.get('ground_truth', ''),
            context=input_data.get('context', [])
        )

        # Initialize metrics
        metrics = self._initialize_deepeval_metrics(
            config.get('metrics', ['answer_relevancy']),
            config
        )

        try:
            # Run evaluation on single test case
            dataset = EvaluationDataset(test_cases=[test_case])
            await self._run_deepeval_async(dataset, metrics)

            # Extract scores
            scores = {}
            for metric in metrics:
                if hasattr(metric, 'score'):
                    metric_name = metric.__name__ if hasattr(metric, '__name__') else str(metric)
                    scores[metric_name] = getattr(metric, 'score', 0.0)

            return scores

        except Exception as e:
            logger.error(f"Error calculating individual metrics: {e}")
            return {}

    async def _update_evaluation_status(
            self, evaluation_id: UUID, status: EvaluationStatus, additional_data: Dict[str, Any] = None
    ) -> None:
        """Update evaluation status in the database."""
        from backend.app.db.repositories.base import BaseRepository
        from backend.app.db.models.orm import Evaluation

        evaluation_repo = BaseRepository(Evaluation, self.db_session)

        update_data = {"status": status}
        if additional_data:
            update_data.update(additional_data)

        await evaluation_repo.update(evaluation_id, update_data)
        logger.info(f"Updated evaluation {evaluation_id} status to {status}")
