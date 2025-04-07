# File: backend/app/evaluation/methods/ragas.py
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.db.models.orm.models import Evaluation, EvaluationStatus
from backend.app.db.schema.evaluation_schema import EvaluationResultCreate, MetricScoreCreate
from backend.app.evaluation.methods.base import BaseEvaluationMethod

# Configure logging
logger = logging.getLogger(__name__)


class RagasEvaluationMethod(BaseEvaluationMethod):
    """Evaluation method using RAGAS library for RAG evaluation."""

    method_name = "ragas"

    def __init__(self, db_session: AsyncSession):
        """Initialize the evaluation method."""
        super().__init__(db_session)
        self._check_ragas_available()

    def _check_ragas_available(self) -> bool:
        """
        Check if the RAGAS library is available and log the availability status.

        Returns:
            bool: True if RAGAS is available, False otherwise
        """
        try:
            import ragas
            self.ragas_available = True
            logger.info(f"RAGAS library found (version: {ragas.__version__})")
            return True
        except ImportError:
            self.ragas_available = False
            logger.warning("RAGAS library not found. Using fallback implementation.")
            return False

    async def run_evaluation(self, evaluation: Evaluation) -> List[EvaluationResultCreate]:
        """
        Run evaluation using RAGAS.

        Args:
            evaluation: Evaluation model

        Returns:
            List[EvaluationResultCreate]: List of evaluation results
        """
        # Update the evaluation to running status if needed
        if evaluation.status == EvaluationStatus.PENDING:
            await self._update_evaluation_status(
                evaluation.id,
                EvaluationStatus.RUNNING,
                {"start_time": time.time()}
            )

        logger.info(f"Starting RAGAS evaluation {evaluation.id}")

        # Process using batch processing approach
        try:
            results = await self.batch_process(evaluation)

            # Update evaluation to completed status
            await self._update_evaluation_status(
                evaluation.id,
                EvaluationStatus.COMPLETED,
                {"end_time": time.time()}
            )

            logger.info(f"Completed RAGAS evaluation {evaluation.id} with {len(results)} results")
            return results

        except Exception as e:
            # Update evaluation to failed status
            await self._update_evaluation_status(
                evaluation.id,
                EvaluationStatus.FAILED,
                {"end_time": time.time()}
            )

            logger.exception(f"Failed RAGAS evaluation {evaluation.id}: {str(e)}")
            raise

    async def _update_evaluation_status(
            self, evaluation_id: UUID, status: EvaluationStatus, additional_data: Dict[str, Any] = None
    ) -> None:
        """
        Update evaluation status in the database.

        Args:
            evaluation_id: Evaluation ID
            status: New status
            additional_data: Additional data to update
        """
        from backend.app.db.repositories.base import BaseRepository
        from backend.app.db.models.orm.models import Evaluation

        evaluation_repo = BaseRepository(Evaluation, self.db_session)

        update_data = {"status": status}
        if additional_data:
            update_data.update(additional_data)

        await evaluation_repo.update(evaluation_id, update_data)

    async def calculate_metrics(
            self, input_data: Dict[str, Any], output_data: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate RAGAS metrics for a single evaluation item.

        Args:
            input_data: Input data for the evaluation
            output_data: Output data from the LLM
            config: Evaluation configuration

        Returns:
            Dict[str, float]: Dictionary mapping metric names to values
        """
        # Extract inputs and outputs
        query = input_data.get("query", "")
        context = input_data.get("context", "")
        ground_truth = input_data.get("ground_truth", "")
        answer = output_data.get("answer", "")

        if not query or not context or not answer:
            logger.warning("Missing required data for RAGAS evaluation")
            return {}

        # Get enabled metrics from config or use defaults
        enabled_metrics = config.get("metrics", ["faithfulness", "response_relevancy", "context_precision"])

        # Try to use actual RAGAS library if available
        if self.ragas_available:
            try:
                metrics = await self._calculate_real_ragas_metrics(
                    query, context, answer, ground_truth, enabled_metrics
                )

                # If we got valid metrics, return them
                if metrics:
                    return metrics

            except Exception as e:
                logger.warning(f"Error using RAGAS library: {e}. Falling back to simple implementation.")
                # Fall through to fallback implementation

        # Fallback implementation
        return await self._calculate_fallback_metrics(
            query, context, answer, ground_truth, enabled_metrics
        )

    async def _calculate_real_ragas_metrics(
            self, query: str, context: str, answer: str, ground_truth: Optional[str],
            enabled_metrics: List[str]
    ) -> Dict[str, float]:
        """
        Calculate metrics using the actual RAGAS library with updated metrics.

        Args:
            query: User query
            context: Source context
            answer: LLM answer
            ground_truth: Optional ground truth
            enabled_metrics: List of metrics to calculate

        Returns:
            Dict[str, float]: Dictionary mapping metric names to values
        """
        try:
            # Import RAGAS components with updated imports
            from ragas import SingleTurnSample
            from ragas.metrics import (
                Faithfulness,
                ResponseRelevancy,
                LLMContextPrecisionWithoutReference,
                LLMContextRecall,
                ContextEntityRecall,
                NoiseSensitivity
            )
            from langchain_openai import AzureChatOpenAI
            from ragas.llms import LangchainLLMWrapper

            # Initialize Azure OpenAI client for evaluation
            llm = AzureChatOpenAI(
                openai_api_key=settings.AZURE_OPENAI_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
                api_version=settings.AZURE_OPENAI_VERSION,
                temperature=0.0
            )

            # Wrap in the RAGAS LLM interface
            ragas_llm = LangchainLLMWrapper(llm)

            # Prepare contexts - convert to list if it's a string
            contexts = [context] if isinstance(context, str) else context

            # Create a SingleTurnSample (updated format)
            sample = SingleTurnSample(
                user_input=query,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth if ground_truth else None
            )

            # Initialize and run metrics based on the enabled list
            metrics_results = {}
            metric_mapping = {
                "faithfulness": Faithfulness(llm=ragas_llm),
                "response_relevancy": ResponseRelevancy(llm=ragas_llm),
                "context_precision": LLMContextPrecisionWithoutReference(llm=ragas_llm),
                "context_recall": LLMContextRecall(llm=ragas_llm) if ground_truth else None,
                "context_entity_recall": ContextEntityRecall(llm=ragas_llm) if ground_truth else None,
                "noise_sensitivity": NoiseSensitivity(llm=ragas_llm) if ground_truth else None
            }

            # Calculate each enabled metric using async methods
            for metric_name in enabled_metrics:
                metric = metric_mapping.get(metric_name)
                if metric:
                    try:
                        # Use the async scoring method
                        score = await metric.single_turn_ascore(sample)
                        metrics_results[metric_name] = float(score)
                        logger.info(f"Calculated {metric_name} score: {score:.4f}")
                    except Exception as e:
                        logger.error(f"Error calculating {metric_name}: {e}")

            return metrics_results

        except Exception as e:
            logger.exception(f"Error in RAGAS metrics calculation: {e}")
            return {}

    async def _calculate_fallback_metrics(
            self, query: str, context: str, answer: str, ground_truth: Optional[str],
            enabled_metrics: List[str]
    ) -> Dict[str, float]:
        """
        Calculate fallback metrics when RAGAS is not available.

        Args:
            query: User query
            context: Source context
            answer: LLM answer
            ground_truth: Optional ground truth
            enabled_metrics: List of metrics to calculate

        Returns:
            Dict[str, float]: Dictionary mapping metric names to values
        """
        # Initialize metrics dictionary
        metrics = {}

        if "faithfulness" in enabled_metrics:
            metrics["faithfulness"] = self._calculate_faithfulness(answer, context)

        if "response_relevancy" in enabled_metrics:
            metrics["response_relevancy"] = self._calculate_answer_relevancy(answer, query)

        if "context_precision" in enabled_metrics:
            metrics["context_precision"] = self._calculate_context_relevancy(context, query)

        # Context recall metric needs ground truth
        if ground_truth and "context_recall" in enabled_metrics:
            metrics["context_recall"] = self._calculate_correctness(answer, ground_truth)

        # Context entity recall (simplified fallback)
        if ground_truth and "context_entity_recall" in enabled_metrics:
            metrics["context_entity_recall"] = self._calculate_entity_recall(context, ground_truth)

        # Noise sensitivity (simplified fallback)
        if ground_truth and "noise_sensitivity" in enabled_metrics:
            # For noise sensitivity, lower is better (0 is ideal)
            # So we invert our correctness score and cap at 1
            metrics["noise_sensitivity"] = min(1.0, 1.0 - self._calculate_correctness(answer, ground_truth))

        return metrics

    def _calculate_faithfulness(self, answer: str, context: str) -> float:
        """Calculate faithfulness score (fallback implementation)."""
        if not answer or not context:
            return 0.0

        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        if not answer_words:
            return 0.0

        overlap = answer_words.intersection(context_words)
        return len(overlap) / len(answer_words)

    def _calculate_answer_relevancy(self, answer: str, query: str) -> float:
        """Calculate answer relevancy score (fallback implementation)."""
        if not answer or not query:
            return 0.0

        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        if not query_words:
            return 0.0

        # Consider key question terms as more important
        question_terms = {"what", "how", "why", "when", "where", "who", "which"}
        query_question_words = {word for word in query_words if word in question_terms}

        # If the query contains question words, check if they're addressed in the answer
        if query_question_words:
            # Calculate a weighted score - question words are more important
            regular_overlap = query_words.intersection(answer_words)
            question_overlap = query_question_words.intersection(answer_words)

            if not question_overlap and query_question_words:
                # Penalize not addressing question words
                score = len(regular_overlap) / (len(query_words) * 2)
            else:
                # Bonus for addressing question words
                score = (len(regular_overlap) + len(question_overlap)) / (len(query_words) + len(query_question_words))

            return min(score, 1.0)

        # Simple overlap for non-question queries
        overlap = query_words.intersection(answer_words)
        return len(overlap) / len(query_words)

    def _calculate_context_relevancy(self, context: str, query: str) -> float:
        """Calculate context relevancy score (fallback implementation)."""
        if not context or not query:
            return 0.0

        query_words = set(query.lower().split())
        context_words = set(context.lower().split())

        if not query_words:
            return 0.0

        overlap = query_words.intersection(context_words)
        return len(overlap) / len(query_words)

    def _calculate_correctness(self, answer: str, ground_truth: str) -> float:
        """Calculate correctness score (fallback implementation)."""
        if not answer or not ground_truth:
            return 0.0

        answer_tokens = answer.lower().split()
        ground_truth_tokens = ground_truth.lower().split()

        if not ground_truth_tokens:
            return 0.0

        # Calculate F1-like score
        common_tokens = sum(1 for token in answer_tokens if token in ground_truth_tokens)

        if not common_tokens:
            return 0.0

        precision = common_tokens / len(answer_tokens) if answer_tokens else 0
        recall = common_tokens / len(ground_truth_tokens) if ground_truth_tokens else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def _calculate_entity_recall(self, context: str, ground_truth: str) -> float:
        """Simple entity recall based on noun phrases (fallback implementation)."""
        if not context or not ground_truth:
            return 0.0

        # Naive entity extraction (words starting with capital letters)
        def extract_entities(text):
            words = text.split()
            entities = set()
            for word in words:
                if word and word[0].isupper() and len(word) > 1:
                    entities.add(word)
            return entities

        context_entities = extract_entities(context)
        ground_truth_entities = extract_entities(ground_truth)

        if not ground_truth_entities:
            return 1.0  # No entities to recall

        # Calculate recall
        common_entities = context_entities.intersection(ground_truth_entities)
        return len(common_entities) / len(ground_truth_entities)

    def _format_prompt(self, prompt_template: str, item: Dict[str, Any]) -> str:
        """Format prompt template with dataset item."""
        formatted_prompt = prompt_template

        # Replace placeholders in the template
        for key, value in item.items():
            placeholder = f"{{{key}}}"
            if placeholder in formatted_prompt:
                formatted_prompt = formatted_prompt.replace(placeholder, str(value))

        return formatted_prompt

    def _get_metric_description(self, metric_name: str) -> str:
        """Get description for a metric."""
        descriptions = {
            "faithfulness": "Measures how well the answer sticks to the information in the context without hallucinating.",
            "response_relevancy": "Measures how relevant the answer is to the query asked.",
            "context_precision": "Measures how precisely the retrieved context matches what's needed to answer the query.",
            "context_recall": "Measures how well the retrieved context covers all the information needed to answer the query.",
            "context_entity_recall": "Measures how well the retrieved context captures the entities mentioned in the reference answer.",
            "noise_sensitivity": "Measures the model's tendency to be misled by irrelevant information in the context (lower is better)."
        }

        return descriptions.get(metric_name, f"Measures the {metric_name} of the response.")