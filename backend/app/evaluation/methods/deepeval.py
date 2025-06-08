import asyncio
import datetime
import logging
import random
import time
import warnings
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import Evaluation, EvaluationStatus
from backend.app.db.schema.evaluation_schema import EvaluationResultCreate, MetricScoreCreate
from backend.app.evaluation.methods.base import BaseEvaluationMethod
from backend.app.evaluation.metrics.deepeval_metrics import (
    DEEPEVAL_AVAILABLE
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepeval_evaluation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Suppress verbose third-party loggers
third_party_loggers = [
    'urllib3.connectionpool',
    'httpx',
    'backoff',
    'posthog',
    'requests.packages.urllib3.connectionpool',
    'deepeval'
]

for logger_name in third_party_loggers:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Suppress warnings
warnings.filterwarnings("ignore")

logger.info("DeepEval method module loaded successfully")

if DEEPEVAL_AVAILABLE:
    try:
        from deepeval import evaluate
        from deepeval.test_case import LLMTestCase
        from deepeval.dataset import EvaluationDataset
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualRecallMetric,
            ContextualPrecisionMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
            BiasMetric,
            ToxicityMetric,
            HallucinationMetric
        )
        from deepeval.models.base_model import DeepEvalBaseLLM

        # Import LangChain Azure OpenAI
        from langchain_openai import AzureChatOpenAI

        logger.info("DeepEval library successfully imported")
    except ImportError as e:
        logger.error(f"Failed to import DeepEval components: {e}")
        DEEPEVAL_AVAILABLE = False


class EnhancedAzureOpenAI(DeepEvalBaseLLM):
    """Enhanced Azure OpenAI model wrapper with proper async/sync handling"""

    def __init__(self, agent):
        """Initialize with agent configuration"""
        self.model = None
        self.agent = agent
        self.semaphore = asyncio.Semaphore(3)
        self.last_request_time = 0
        self.min_request_interval = 0.5
        self._rate_limit_lock = asyncio.Lock()

        super().__init__()

    def load_model(self):
        """Load the model with error handling"""
        try:
            if not self.model:
                from backend.app.core.config import settings

                # Handle case where agent is None (for mock scenarios)
                if self.agent is None:
                    # Use default settings for mock scenarios
                    self.model = AzureChatOpenAI(
                        openai_api_key=settings.AZURE_OPENAI_KEY,
                        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
                        api_version=settings.AZURE_OPENAI_VERSION,
                        request_timeout=60,
                        max_retries=3,
                        temperature=0.0
                    )
                else:
                    # Extract credentials from agent
                    credentials = self.agent.auth_credentials or {}

                    self.model = AzureChatOpenAI(
                        openai_api_key=credentials.get("api_key", settings.AZURE_OPENAI_KEY),
                        azure_endpoint=self.agent.api_endpoint or settings.AZURE_OPENAI_ENDPOINT,
                        azure_deployment=credentials.get("deployment", settings.AZURE_OPENAI_DEPLOYMENT),
                        api_version=credentials.get("api_version", settings.AZURE_OPENAI_VERSION),
                        request_timeout=60,
                        max_retries=3,
                        temperature=0.0
                    )

                logger.info("Successfully loaded Azure OpenAI model")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    async def _apply_rate_limiting(self):
        """Apply rate limiting with async lock"""
        async with self._rate_limit_lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - elapsed)
            self.last_request_time = time.time()

    async def _retry_request_async(self, request_func, max_retries=5):
        """Execute async request with retry logic"""
        for attempt in range(max_retries):
            try:
                return await request_func()
            except Exception as e:
                if self._should_retry(e, attempt, max_retries):
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                raise

    @staticmethod
    def _should_retry(error: Exception, attempt: int, max_retries: int) -> bool:
        """Determine if request should be retried"""
        if attempt >= max_retries - 1:
            return False

        error_str = str(error).lower()
        retry_conditions = [
            "500" in error_str,
            "502" in error_str,
            "503" in error_str,
            "504" in error_str,
            "rate limit" in error_str,
            "timeout" in error_str,
            "connection" in error_str
        ]

        return any(retry_conditions)

    def generate(self, prompt: str) -> str:
        """Generate response with retry logic - FIXED SYNC VERSION"""
        # This needs to be synchronous to match DeepEval expectations
        max_retries = 5

        for attempt in range(max_retries):
            try:
                # Simple synchronous rate limiting
                elapsed = time.time() - self.last_request_time
                if elapsed < self.min_request_interval:
                    time.sleep(self.min_request_interval - elapsed)

                model = self.load_model()
                response = model.invoke(prompt)
                self.last_request_time = time.time()
                return response.content

            except Exception as e:
                if self._should_retry(e, attempt, max_retries):
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Sync retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                    time.sleep(wait_time)
                    continue
                logger.error(f"Generate failed after {attempt + 1} attempts: {e}")
                raise

    async def a_generate(self, prompt: str) -> str:
        """Async generation with proper rate limiting and retry logic"""

        async def _make_request():
            model = self.load_model()
            response = await model.ainvoke(prompt)
            return response.content

        try:
            async with self.semaphore:
                await self._apply_rate_limiting()
                return await self._retry_request_async(_make_request)
        except Exception as e:
            logger.error(f"Async generation failed: {str(e)}")
            raise

    def get_model_name(self):
        """Return model name"""
        return "Enhanced Azure OpenAI Model"


class DeepEvalMethod(BaseEvaluationMethod):
    method_name = "deepeval"

    def __init__(self, db_session: AsyncSession):
        """Initialize"""
        super().__init__(db_session)
        self.deepeval_available = DEEPEVAL_AVAILABLE

        if not self.deepeval_available:
            raise ImportError("DeepEval library is not available. Please install it with: pip install deepeval")

        logger.info(f"Initializing DeepEval evaluation method. DeepEval available: {self.deepeval_available}")

    async def run_evaluation(self,
                             evaluation: Evaluation,
                             jwt_token: Optional[str] = None
                             ) -> List[EvaluationResultCreate]:
        logger.info(f"Starting DeepEval evaluation {evaluation.id}")

        try:
            # Get related entities
            from backend.app.db.repositories.agent_repository import AgentRepository

            agent_repo = AgentRepository(self.db_session)
            agent = await agent_repo.get_with_decrypted_credentials(evaluation.agent_id)
            dataset = await self.get_dataset(evaluation.dataset_id)
            prompt = await self.get_prompt(evaluation.prompt_id)

            if not agent or not dataset or not prompt:
                raise ValueError(f"Missing required entities for evaluation {evaluation.id}")

            # Load dataset and create test cases
            dataset_items = await self.load_dataset(dataset)
            test_cases = self._create_comprehensive_test_cases(dataset_items)

            # Generate agent responses
            test_cases_with_outputs = await self._generate_agent_responses(
                test_cases, agent, prompt, jwt_token
            )

            # Initialize Azure OpenAI model - removed logger parameter
            azure_openai = EnhancedAzureOpenAI(agent)

            # Create metrics following
            metrics_dict = self._create_comprehensive_metrics(azure_openai, evaluation)

            # Validate test cases
            validated_metrics_dict = self._validate_test_cases_for_metrics(test_cases_with_outputs, metrics_dict)

            if not validated_metrics_dict:
                raise ValueError("No metrics can be used with the provided test cases")

            # Extract metrics for evaluation
            metrics = [m['metric'] for m in validated_metrics_dict.values()]

            logger.info(f"Running evaluation with {len(metrics)} metrics on {len(test_cases_with_outputs)} test cases")

            # Run evaluation
            results = evaluate(test_cases=test_cases_with_outputs, metrics=metrics)

            # Convert results to platform format
            platform_results = await self._convert_results_to_platform_format(
                evaluation, results, test_cases_with_outputs, validated_metrics_dict
            )

            logger.info(f"Completed DeepEval evaluation {evaluation.id} with {len(platform_results)} results")
            return platform_results

        except Exception as e:
            await self._update_evaluation_status(
                evaluation.id,
                EvaluationStatus.FAILED,
                {"end_time": datetime.datetime.now()}
            )
            logger.exception(f"Failed DeepEval evaluation {evaluation.id}: {str(e)}")
            raise

    @staticmethod
    def _create_comprehensive_test_cases(dataset_items: List[Dict[str, Any]]) -> List[LLMTestCase]:
        """Create test cases following"""
        test_cases = []

        for i, item in enumerate(dataset_items):
            try:
                # Extract fields using flexible mapping
                query = item.get("query", item.get("question", item.get("input", "")))
                context = item.get("context", item.get("contexts", []))
                expected_output = item.get("ground_truth", item.get("expected_answer", item.get("answer", "")))

                # Normalize context
                if isinstance(context, str):
                    context = [context] if context else []
                elif not isinstance(context, list):
                    context = []

                # Create test case with all fields
                test_case = LLMTestCase(
                    input=query,
                    actual_output='',  # Will be filled by agent response
                    expected_output=expected_output,
                    context=context,
                    retrieval_context=item.get("retrieval_context", context)
                )

                test_cases.append(test_case)

            except Exception as e:
                logger.error(f"Error creating test case {i}: {e}")
                # Create minimal test case for error handling
                error_test_case = LLMTestCase(
                    input=f"Error processing item {i}",
                    actual_output="Error occurred during processing",
                    expected_output="Error",
                    context=["Error in processing"],
                    retrieval_context=["Error in processing"]
                )
                test_cases.append(error_test_case)

        logger.info(f"Created {len(test_cases)} comprehensive test cases")
        return test_cases

    async def _generate_agent_responses(
            self,
            test_cases: List[LLMTestCase],
            agent,
            prompt,
            jwt_token: Optional[str] = None
    ) -> List[LLMTestCase]:
        """Generate agent responses"""
        from backend.app.services.agent_clients.factory import AgentClientFactory
        from backend.app.db.models.orm import IntegrationType

        # Create agent client
        if agent.integration_type == IntegrationType.MCP and jwt_token:
            logger.info(f"Using JWT token for MCP agent in evaluation")
            agent_client = await AgentClientFactory.create_client(agent, jwt_token)
        else:
            agent_client = await AgentClientFactory.create_client(agent)

        # Process test cases to generate responses
        for i, test_case in enumerate(test_cases):
            try:
                # Format prompt with test case data
                formatted_prompt = self._format_prompt(prompt.content, {
                    "query": test_case.input,
                    "context": "\n".join(test_case.context) if test_case.context else "",
                    "ground_truth": test_case.expected_output or ""
                })

                # Generate response using agent client
                response = await agent_client.process_query(
                    query=test_case.input,
                    context="\n".join(test_case.context) if test_case.context else None,
                    system_message=formatted_prompt
                )

                # Extract answer
                if response.get("success", True):
                    test_case.actual_output = response.get("answer", "")
                else:
                    test_case.actual_output = f"Error: {response.get('error', 'Unknown error')}"

                if (i + 1) % 5 == 0:
                    logger.info(f"Generated {i + 1}/{len(test_cases)} responses")

            except Exception as e:
                logger.error(f"Error generating response for test case {i}: {e}")
                test_case.actual_output = f"Error: {str(e)}"

        return test_cases

    @staticmethod
    def _create_comprehensive_metrics(azure_openai, evaluation: Evaluation) -> Dict[str, Any]:
        """Create metrics following"""
        metrics = {}

        try:
            # Get threshold from config
            threshold = evaluation.config.get('threshold', 0.7) if evaluation.config else 0.7
            include_reason = evaluation.config.get('include_reason', True) if evaluation.config else True

            # Core RAG Metrics
            metrics['answer_relevancy'] = {
                'metric': AnswerRelevancyMetric(
                    threshold=threshold,
                    model=azure_openai,
                    include_reason=include_reason
                ),
                'description': 'Measures how relevant the answer is to the given question',
                'use_case': 'Ensures responses directly address user queries'
            }

            metrics['contextual_recall'] = {
                'metric': ContextualRecallMetric(
                    threshold=threshold,
                    model=azure_openai,
                    include_reason=include_reason
                ),
                'description': 'Measures how much of the expected output can be attributed to the retrieval context',
                'use_case': 'Evaluates if retrieved context contains necessary information'
            }

            metrics['contextual_precision'] = {
                'metric': ContextualPrecisionMetric(
                    threshold=threshold,
                    model=azure_openai,
                    include_reason=include_reason
                ),
                'description': 'Measures whether nodes in retrieval context are relevant to the given input',
                'use_case': 'Evaluates quality of retrieved context'
            }

            metrics['contextual_relevancy'] = {
                'metric': ContextualRelevancyMetric(
                    threshold=threshold,
                    model=azure_openai,
                    include_reason=include_reason
                ),
                'description': 'Measures how relevant the retrieval context is to the given input',
                'use_case': 'Ensures retrieved documents are actually relevant'
            }

            metrics['faithfulness'] = {
                'metric': FaithfulnessMetric(
                    threshold=threshold,
                    model=azure_openai,
                    include_reason=include_reason
                ),
                'description': 'Measures how factually accurate the actual output is to the retrieval context',
                'use_case': 'Prevents hallucinations and ensures factual accuracy'
            }

            # Safety and Ethics Metrics
            metrics['bias'] = {
                'metric': BiasMetric(
                    threshold=0.5,  # Lower threshold for bias detection
                    model=azure_openai,
                    include_reason=include_reason
                ),
                'description': 'Detects bias in model outputs across various dimensions',
                'use_case': 'Ensures fair and unbiased responses'
            }

            metrics['toxicity'] = {
                'metric': ToxicityMetric(
                    threshold=0.5,  # Lower threshold for toxicity detection
                    model=azure_openai,
                    include_reason=include_reason
                ),
                'description': 'Detects toxic, harmful, or inappropriate content',
                'use_case': 'Ensures safe and appropriate responses'
            }

            # Content Quality Metrics
            metrics['hallucination'] = {
                'metric': HallucinationMetric(
                    threshold=0.5,
                    model=azure_openai,
                    include_reason=include_reason
                ),
                'description': 'Detects hallucinated information not present in the context',
                'use_case': 'Prevents generation of false information'
            }

            logger.info(f"Successfully created {len(metrics)} evaluation metrics")

        except Exception as e:
            logger.error(f"Failed to create metrics: {str(e)}")
            raise

        return metrics

    def _validate_test_cases_for_metrics(self, test_cases: List[LLMTestCase], metrics_dict: Dict) -> Dict[str, Any]:
        """Validate test cases against metric requirements."""
        validated_metrics = {}
        issues = []

        for metric_name, metric_info in metrics_dict.items():
            metric = metric_info['metric']
            metric_type = type(metric).__name__

            # Check requirements for each metric type
            can_use_metric = True
            missing_fields = []

            for test_case in test_cases:
                if metric_type == 'HallucinationMetric':
                    if not test_case.context:
                        missing_fields.append(f"Test case '{test_case.input[:30]}...' missing 'context' field")
                        can_use_metric = False

                elif metric_type in ['ContextualRecallMetric', 'ContextualPrecisionMetric',
                                     'ContextualRelevancyMetric']:
                    if not test_case.retrieval_context:
                        missing_fields.append(
                            f"Test case '{test_case.input[:30]}...' missing 'retrieval_context' field")
                        can_use_metric = False

                elif metric_type == 'FaithfulnessMetric':
                    if not test_case.retrieval_context:
                        missing_fields.append(
                            f"Test case '{test_case.input[:30]}...' missing 'retrieval_context' field")
                        can_use_metric = False

            if can_use_metric:
                validated_metrics[metric_name] = metric_info
                logger.info(f"[OK] {metric_name} validated successfully")
            else:
                issues.extend(missing_fields)
                logger.warning(f"[SKIP] Skipping {metric_name} due to missing required fields")

        if issues:
            logger.warning(f"Validation issues found: {len(issues)} problems detected")
            for issue in issues[:5]:  # Show first 5 issues
                logger.warning(f"  - {issue}")

        logger.info(f"Using {len(validated_metrics)}/{len(metrics_dict)} metrics after validation")
        return validated_metrics

    async def _convert_results_to_platform_format(
            self,
            evaluation: Evaluation,
            deepeval_results: Any,
            test_cases: List[LLMTestCase],
            validated_metrics_dict: Dict
    ) -> List[EvaluationResultCreate]:
        """Convert DeepEval results to platform format."""
        platform_results = []

        # Process each test result
        if hasattr(deepeval_results, 'test_results'):
            for i, test_result in enumerate(deepeval_results.test_results):
                try:
                    # Extract metric scores
                    metric_scores = []
                    total_score = 0
                    passed_count = 0

                    # Get metric results from test case
                    for metric_data in test_result.metrics_data:
                        score_value = getattr(metric_data, 'score', 0)
                        success = getattr(metric_data, 'success', False)
                        reason = getattr(metric_data, 'reason', '')

                        metric_scores.append(MetricScoreCreate(
                            name=metric_data.name,
                            value=score_value,
                            weight=1.0,
                            meta_info={
                                'success': success,
                                'reason': reason,
                                'threshold': getattr(metric_data, 'threshold', 0.7)
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
                            'input': test_cases[i].input if i < len(test_cases) else "",
                            'context': test_cases[i].context if i < len(test_cases) else [],
                            'expected_output': test_cases[i].expected_output if i < len(test_cases) else ""
                        },
                        output_data={
                            'actual_output': test_cases[i].actual_output if i < len(test_cases) else ""
                        },
                        metric_scores=metric_scores,
                        passed=passed,
                        pass_threshold=evaluation.pass_threshold or 0.7
                    )

                    platform_results.append(result)

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
                    platform_results.append(error_result)

        return platform_results

    async def calculate_metrics(
            self, input_data: Dict[str, Any], output_data: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate metrics for individual items"""
        logger.info("Individual metric calculation called - DeepEval works best with batch processing")

        # For individual calculation, create a single test case
        test_case = LLMTestCase(
            input=input_data.get('query', ''),
            actual_output=output_data.get('answer', ''),
            expected_output=input_data.get('ground_truth', ''),
            context=input_data.get('context', []),
            retrieval_context=input_data.get('context', [])
        )

        try:
            # Create a mock Azure OpenAI for individual calculations - removed logger parameter
            mock_azure = EnhancedAzureOpenAI(None)

            # Initialize metrics
            metrics = self._create_comprehensive_metrics(mock_azure, type('MockEval', (), {'config': config})())

            # Run evaluation on single test case
            results = evaluate(test_cases=[test_case], metrics=[m['metric'] for m in metrics.values()])

            # Extract scores
            scores = {}
            if hasattr(results, 'test_results') and results.test_results:
                for metric_data in results.test_results[0].metrics_data:
                    scores[metric_data.name] = getattr(metric_data, 'score', 0.0)

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
