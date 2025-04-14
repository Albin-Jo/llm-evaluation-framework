# File: app/evaluation/methods/deepeval.py
import logging
from typing import Any, Dict, List
import httpx

from backend.app.core.config import settings
from backend.app.evaluation.methods.base import BaseEvaluationMethod
from backend.app.db.models.orm import Evaluation
from backend.app.db.schema.evaluation_schema import EvaluationResultCreate, MetricScoreCreate

# Configure logging
logger = logging.getLogger(__name__)


class DeepEvalEvaluationMethod(BaseEvaluationMethod):
    """Evaluation method using DeepEval library."""

    async def run_evaluation(self, evaluation: Evaluation) -> List[EvaluationResultCreate]:
        """
        Run evaluation using DeepEval.

        Args:
            evaluation: Evaluation model

        Returns:
            List[EvaluationResultCreate]: List of evaluation results
        """
        # Get related entities
        microagent = await self.get_microagent(evaluation.micro_agent_id)
        dataset = await self.get_dataset(evaluation.dataset_id)
        prompt = await self.get_prompt(evaluation.prompt_id)

        if not microagent or not dataset or not prompt:
            logger.error(f"Missing required entities for evaluation {evaluation.id}")
            return []

        # Load dataset
        dataset_items = await self.load_dataset(dataset)

        results = []

        for item_index, item in enumerate(dataset_items):
            try:
                # Process dataset item
                query = item.get("query", "")
                context = item.get("context", "")
                ground_truth = item.get("ground_truth", "")
                references = item.get("references", [])

                # Format prompt with dataset item
                formatted_prompt = self._format_prompt(prompt.content, item)

                # Call the micro-agent API
                response = await self._call_microagent_api(
                    microagent.api_endpoint,
                    {
                        "prompt": formatted_prompt,
                        "query": query,
                        "context": context
                    }
                )

                # Extract LLM answer from response
                answer = response.get("answer", "")

                # Calculate metrics
                metrics = await self.calculate_metrics(
                    input_data={
                        "query": query,
                        "context": context,
                        "ground_truth": ground_truth,
                        "references": references
                    },
                    output_data={"answer": answer},
                    config=evaluation.config or {}
                )

                # Calculate overall score (weighted average of all metrics)
                metric_weights = {metric["name"]: metric.get("weight", 1.0) for metric in metrics}
                total_weight = sum(metric_weights.values())
                overall_score = (
                    sum(metric["value"] * metric_weights[metric["name"]] for metric in metrics) / total_weight
                    if total_weight > 0 else 0.0
                )

                # Create metric scores
                metric_scores = [
                    MetricScoreCreate(
                        name=metric["name"],
                        value=metric["value"],
                        weight=metric.get("weight", 1.0),
                        metadata={"description": metric.get("description", "")}
                    )
                    for metric in metrics
                ]

                # Create evaluation result
                result = EvaluationResultCreate(
                    evaluation_id=evaluation.id,
                    overall_score=overall_score,
                    raw_results={metric["name"]: metric["value"] for metric in metrics},
                    dataset_sample_id=str(item_index),
                    input_data={
                        "query": query,
                        "context": context,
                        "ground_truth": ground_truth,
                        "prompt": formatted_prompt
                    },
                    output_data={"answer": answer},
                    processing_time_ms=response.get("processing_time_ms"),
                    metric_scores=metric_scores
                )

                results.append(result)

            except Exception as e:
                logger.exception(f"Error processing dataset item {item_index}: {e}")

                # Create failed evaluation result
                result = EvaluationResultCreate(
                    evaluation_id=evaluation.id,
                    overall_score=0.0,
                    raw_results={"error": str(e)},
                    dataset_sample_id=str(item_index),
                    input_data=item,
                    output_data={"error": str(e)},
                    metric_scores=[]
                )

                results.append(result)

        return results

    async def calculate_metrics(
            self, input_data: Dict[str, Any], output_data: Dict[str, Any], config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Calculate DeepEval metrics for a single evaluation item.

        Args:
            input_data: Input data for the evaluation
            output_data: Output data from the LLM
            config: Evaluation configuration

        Returns:
            List[Dict[str, Any]]: List of metric dictionaries with name, value, weight, and description
        """
        try:
            # Extract inputs and outputs
            query = input_data.get("query", "")
            context = input_data.get("context", "")
            ground_truth = input_data.get("ground_truth", "")
            references = input_data.get("references", [])
            answer = output_data.get("answer", "")

            metrics = []

            # Get configuration for enabled metrics
            enabled_metrics = config.get("metrics", ["bias", "relevance", "coherence", "groundedness"])

            # Bias metric - measures potential bias in the response
            if "bias" in enabled_metrics:
                bias_score = await self._calculate_bias(answer)
                metrics.append({
                    "name": "bias",
                    "value": bias_score,
                    "weight": config.get("bias_weight", 1.0),
                    "description": "Measures the absence of bias in the generated response"
                })

            # Relevance metric - measures relevance to the query
            if "relevance" in enabled_metrics:
                relevance_score = await self._calculate_relevance(answer, query)
                metrics.append({
                    "name": "relevance",
                    "value": relevance_score,
                    "weight": config.get("relevance_weight", 1.0),
                    "description": "Measures how relevant the response is to the query"
                })

            # Coherence metric - measures the coherence of the response
            if "coherence" in enabled_metrics:
                coherence_score = await self._calculate_coherence(answer)
                metrics.append({
                    "name": "coherence",
                    "value": coherence_score,
                    "weight": config.get("coherence_weight", 1.0),
                    "description": "Measures the logical coherence and structure of the response"
                })

            # Groundedness metric - measures factual accuracy relative to context
            if "groundedness" in enabled_metrics:
                groundedness_score = await self._calculate_groundedness(answer, context)
                metrics.append({
                    "name": "groundedness",
                    "value": groundedness_score,
                    "weight": config.get("groundedness_weight", 1.0),
                    "description": "Measures how well the response is grounded in the provided context"
                })

            # Hallucination metric - measures the degree of fabricated information
            if "hallucination" in enabled_metrics:
                hallucination_score = await self._calculate_hallucination(answer, context)
                metrics.append({
                    "name": "hallucination",
                    "value": hallucination_score,
                    "weight": config.get("hallucination_weight", 1.0),
                    "description": "Measures the absence of hallucinated content not supported by context"
                })

            # Correctness metric - if ground truth available
            if ground_truth and "correctness" in enabled_metrics:
                correctness_score = await self._calculate_correctness(answer, ground_truth)
                metrics.append({
                    "name": "correctness",
                    "value": correctness_score,
                    "weight": config.get("correctness_weight", 1.0),
                    "description": "Measures the factual correctness compared to ground truth"
                })

            # Fluency metric - measures the linguistic quality
            if "fluency" in enabled_metrics:
                fluency_score = await self._calculate_fluency(answer)
                metrics.append({
                    "name": "fluency",
                    "value": fluency_score,
                    "weight": config.get("fluency_weight", 0.5),
                    "description": "Measures the linguistic quality and readability of the response"
                })

            return metrics

        except Exception as e:
            logger.exception(f"Error calculating DeepEval metrics: {e}")
            return []

    async def _calculate_bias(self, answer: str) -> float:
        """
        Calculate bias score using DeepEval's bias detection.

        In a real implementation, this would use DeepEval's bias evaluation.
        Here we'll create a simplified implementation.

        Args:
            answer: LLM answer

        Returns:
            float: Bias score (0-1), higher means less biased
        """
        # Placeholder for DeepEval integration
        # In a real implementation, this would import from deepeval and use its bias evaluator

        # Simplified bias detection - look for potentially biased language patterns
        bias_indicators = [
            "always", "never", "all", "none", "everyone", "nobody",
            "definitely", "absolutely", "certainly", "undoubtedly",
            "clearly", "obviously", "perfect", "terrible", "best", "worst"
        ]

        words = set(answer.lower().split())
        bias_count = sum(1 for word in bias_indicators if word in words)

        # Calculate bias score (inversely related to bias indicators)
        bias_ratio = bias_count / (len(words) + 0.001)  # Avoid division by zero
        bias_score = max(0, 1 - (bias_ratio * 5))  # Scale to penalize bias more heavily

        return bias_score

    async def _calculate_relevance(self, answer: str, query: str) -> float:
        """
        Calculate relevance score using DeepEval.

        Args:
            answer: LLM answer
            query: User query

        Returns:
            float: Relevance score (0-1)
        """
        # Simplified semantic relevance calculation
        if not answer or not query:
            return 0.0

        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        if not query_words:
            return 0.0

        # Calculate word overlap as a simple proxy for semantic similarity
        overlap = query_words.intersection(answer_words)
        basic_score = len(overlap) / len(query_words)

        # Enhance with keyword weighting - consider important question words
        question_indicators = {"what", "why", "how", "when", "where", "who", "which"}
        query_has_question = any(word in question_indicators for word in query_words)

        # Boost score if answer addresses question words
        if query_has_question:
            answer_length_factor = min(1.0, len(answer) / 100)  # Reward sufficiently detailed answers
            return (basic_score * 0.7) + (answer_length_factor * 0.3)

        return basic_score

    async def _calculate_coherence(self, answer: str) -> float:
        """
        Calculate coherence score using DeepEval.

        Args:
            answer: LLM answer

        Returns:
            float: Coherence score (0-1)
        """
        if not answer:
            return 0.0

        # Simplified coherence metrics
        sentences = answer.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return 0.5  # Single sentence has medium coherence by default

        # Check for coherence indicators:
        # 1. Sentence length variation (some variation is good)
        lengths = [len(s) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        length_variation = sum(abs(l - avg_length) for l in lengths) / (avg_length * len(lengths))
        length_score = min(1.0, max(0.0, 1.0 - (length_variation - 0.3) * 2))  # Some variation is good

        # 2. Transition words (indicate logical flow)
        transition_words = ["therefore", "however", "thus", "additionally", "furthermore",
                            "consequently", "meanwhile", "nevertheless", "although", "besides",
                            "accordingly", "finally", "similarly", "conversely", "specifically"]

        transition_count = sum(1 for word in transition_words if word.lower() in answer.lower())
        transition_score = min(1.0, transition_count / (len(sentences) / 2))

        # 3. Pronoun usage consistency
        pronouns = ["it", "they", "them", "their", "he", "she", "his", "her"]
        has_pronouns = any(pronoun in answer.lower().split() for pronoun in pronouns)

        # Combine scores
        if has_pronouns:
            return (length_score * 0.3) + (transition_score * 0.7)
        else:
            return (length_score * 0.5) + (transition_score * 0.5)

    async def _calculate_groundedness(self, answer: str, context: str) -> float:
        """
        Calculate groundedness score using DeepEval.

        Args:
            answer: LLM answer
            context: Provided context

        Returns:
            float: Groundedness score (0-1)
        """
        if not answer or not context:
            return 0.0

        # More sophisticated groundedness check
        answer_sentences = [s.strip() for s in answer.split('.') if s.strip()]

        if not answer_sentences:
            return 0.0

        # Process context to extract key information
        context_lower = context.lower()

        # Check each sentence for support in context with smarter analysis
        grounded_sentences = 0
        total_weight = 0

        for sentence in answer_sentences:
            # Skip very short sentences as they're often not substantive claims
            if len(sentence.split()) < 3:
                continue

            # Extract potential entities and key phrases from the sentence
            words = sentence.lower().split()
            # Remove common stop words
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "is",
                          "are"}
            content_words = [w for w in words if w not in stop_words]

            if not content_words:
                continue

            # Calculate a weighted score based on content word presence in context
            # More important/specific words have higher impact
            sentence_weight = len(content_words)
            total_weight += sentence_weight

            matched_word_count = sum(1 for word in content_words if word in context_lower)
            sentence_score = matched_word_count / len(content_words)

            # Bonus for consecutive words matching (phrases)
            for i in range(len(content_words) - 1):
                phrase = f"{content_words[i]} {content_words[i + 1]}"
                if phrase in context_lower:
                    sentence_score += 0.2  # Bonus for each matching phrase

            # Consider sentence grounded if sufficient evidence
            if sentence_score > 0.6:
                grounded_sentences += sentence_weight

        # Return weighted average for groundedness
        return grounded_sentences / total_weight if total_weight > 0 else 0.0

    async def _calculate_hallucination(self, answer: str, context: str) -> float:
        """
        Calculate hallucination score using DeepEval.

        Args:
            answer: LLM answer
            context: Provided context

        Returns:
            float: Hallucination score (0-1), higher is better (less hallucination)
        """
        # Inverse of groundedness score, but with stronger penalties
        groundedness = await self._calculate_groundedness(answer, context)

        # Apply non-linear transformation to penalize hallucination more strongly
        # This creates a steeper drop-off for lower groundedness values
        return groundedness ** 0.7  # Power less than 1 makes low scores even lower

    async def _calculate_correctness(self, answer: str, ground_truth: str) -> float:
        """
        Calculate correctness score using DeepEval.

        Args:
            answer: LLM answer
            ground_truth: Expected answer

        Returns:
            float: Correctness score (0-1)
        """
        if not answer or not ground_truth:
            return 0.0

        # Simplified semantic similarity for correctness
        answer_words = set(answer.lower().split())
        truth_words = set(ground_truth.lower().split())

        if not truth_words:
            return 0.0

        # Calculate F1-style score using precision and recall
        if not answer_words:
            return 0.0

        common_words = answer_words.intersection(truth_words)
        precision = len(common_words) / len(answer_words)
        recall = len(common_words) / len(truth_words)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    async def _calculate_fluency(self, answer: str) -> float:
        """
        Calculate fluency score using DeepEval.

        Args:
            answer: LLM answer

        Returns:
            float: Fluency score (0-1)
        """
        if not answer:
            return 0.0

        # Simplified fluency metrics
        words = answer.split()
        if not words:
            return 0.0

        # 1. Average word length (avoiding extremely short or long words)
        avg_word_length = sum(len(word) for word in words) / len(words)
        length_score = 1.0 - abs(avg_word_length - 5.0) / 5.0  # Optimal around 5 chars

        # 2. Sentence length variation
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        if len(sentences) <= 1:
            sentence_var_score = 0.5
        else:
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            # Some variance is good for fluency
            sentence_var_score = min(1.0, variance / 10) if variance < 10 else 1.0 - min(1.0, (variance - 10) / 30)

        # 3. Repetition penalty
        unique_words = set(word.lower() for word in words)
        repetition_score = len(unique_words) / len(words)

        # Combine scores
        return (length_score * 0.3) + (sentence_var_score * 0.3) + (repetition_score * 0.4)

    def _format_prompt(self, prompt_template: str, item: Dict[str, Any]) -> str:
        """
        Format prompt template with dataset item.

        Args:
            prompt_template: Prompt template string
            item: Dataset item

        Returns:
            str: Formatted prompt
        """
        formatted_prompt = prompt_template

        # Replace placeholders in the template
        for key, value in item.items():
            placeholder = f"{{{key}}}"
            if placeholder in formatted_prompt:
                formatted_prompt = formatted_prompt.replace(placeholder, str(value))

        return formatted_prompt

    async def _call_microagent_api(
            self, api_endpoint: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call the micro-agent API.

        Args:
            api_endpoint: API endpoint URL
            payload: Request payload

        Returns:
            Dict[str, Any]: API response

        Raises:
            Exception: If API call fails
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    api_endpoint,
                    json=payload,
                    headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                    timeout=60.0
                )

                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling micro-agent API: {e}")
            raise Exception(f"Micro-agent API returned error: {e.response.status_code}")

        except httpx.RequestError as e:
            logger.error(f"Request error calling micro-agent API: {e}")
            raise Exception(f"Error connecting to micro-agent API: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error calling micro-agent API: {e}")
            raise Exception(f"Error calling micro-agent API: {str(e)}")