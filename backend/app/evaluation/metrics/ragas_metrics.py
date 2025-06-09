import asyncio
import logging
import time
from functools import lru_cache
from typing import Dict, List, Optional, Union, Any, Set, Tuple

import ragas
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextPrecision,
    Faithfulness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity,
    ContextRecall,
    AnswerCorrectness,
    AnswerSimilarity,
    AnswerRelevancy,
    FactualCorrectness
)

# Configure logging to include line numbers
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)

RAGAS_AVAILABLE = True
logger.info(f"RAGAS library found (version: {ragas.__version__})")

# Cache for LLM and embedding instances to avoid recreating them
_cache = {}


# Custom Exception Hierarchies
class RAGASException(Exception):
    """Base exception for RAGAS-related errors."""
    pass


class RAGASInitializationException(RAGASException):
    """Raised when RAGAS components fail to initialize."""
    pass


class RAGASTimeoutException(RAGASException):
    """Raised when RAGAS calculation times out."""
    pass


class RAGASAPIException(RAGASException):
    """Raised when API calls fail."""
    pass


class RAGASValidationException(RAGASException):
    """Raised when input validation fails."""
    pass


class FallbackCalculationException(RAGASException):
    """Raised when fallback calculations fail."""
    pass


class FallbackCalculator:
    """Consolidated fallback calculations with proper logging."""

    @staticmethod
    @lru_cache(maxsize=1000)
    def _tokenize_and_extract_features(text: str) -> Tuple[Set[str], Set[str]]:
        """
        Extract word tokens and simple entities from text.

        Returns:
            Tuple of (word_tokens, entities)
        """
        try:
            if not text or not isinstance(text, str):
                logger.warning(f"Invalid text input for tokenization")
                return set(), set()

            words = set(text.lower().split())
            # Simple entity extraction (capitalized words)
            entities = {word for word in text.split() if word and len(word) > 1 and word[0].isupper()}

            logger.debug(
                f"Tokenized text: {len(words)} words, {len(entities)} entities")
            return words, entities

        except Exception as e:
            logger.error(f"Error in tokenization: {e}")
            raise FallbackCalculationException(f"Tokenization failed: {e}")

    @staticmethod
    def _calculate_token_overlap_score(text1: str, text2: str, score_type: str = "jaccard") -> float:
        """
        Calculate token overlap score between two texts.

        Args:
            text1: First text
            text2: Second text
            score_type: Type of score ('jaccard', 'precision', 'recall', 'f1')

        Returns:
            float: Overlap score (0-1)
        """
        try:
            logger.debug(f"Calculating {score_type} overlap score")

            if not text1 or not text2:
                logger.warning(f"Empty input for overlap calculation")
                return 0.0

            words1, _ = FallbackCalculator._tokenize_and_extract_features(text1)
            words2, _ = FallbackCalculator._tokenize_and_extract_features(text2)

            if not words1 or not words2:
                logger.warning(f"No tokens found in texts for overlap")
                return 0.0

            intersection = words1.intersection(words2)

            if score_type == "jaccard":
                union = words1.union(words2)
                score = len(intersection) / len(union) if union else 0.0
            elif score_type == "precision":
                score = len(intersection) / len(words1)
            elif score_type == "recall":
                score = len(intersection) / len(words2)
            elif score_type == "f1":
                if not intersection:
                    score = 0.0
                else:
                    precision = len(intersection) / len(words1)
                    recall = len(intersection) / len(words2)
                    score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                logger.error(f"Unknown score type: {score_type}")
                raise FallbackCalculationException(f"Unknown score type: {score_type}")

            logger.debug(f"{score_type} score calculated: {score:.3f}")
            return score

        except Exception as e:
            logger.error(f"Error calculating token overlap: {e}")
            raise FallbackCalculationException(f"Token overlap calculation failed: {e}")

    @staticmethod
    def _calculate_entity_overlap(text1: str, text2: str, score_type: str = "recall") -> float:
        """
        Calculate entity overlap between two texts.

        Args:
            text1: First text
            text2: Second text
            score_type: Type of score ('precision', 'recall', 'f1')

        Returns:
            float: Entity overlap score (0-1)
        """
        try:
            logger.debug(f"Calculating entity {score_type}")

            if not text1 or not text2:
                logger.warning(f"Empty input for entity overlap")
                return 0.0

            _, entities1 = FallbackCalculator._tokenize_and_extract_features(text1)
            _, entities2 = FallbackCalculator._tokenize_and_extract_features(text2)

            if score_type == "recall" and not entities2:
                logger.info(
                    f"No entities in reference text, returning 1.0")
                return 1.0
            elif score_type == "precision" and not entities1:
                logger.info(
                    f"No entities in candidate text, returning 1.0")
                return 1.0

            if not entities1 or not entities2:
                logger.warning(
                    f"No entities found for overlap calculation")
                return 0.0

            intersection = entities1.intersection(entities2)

            if score_type == "precision":
                score = len(intersection) / len(entities1)
            elif score_type == "recall":
                score = len(intersection) / len(entities2)
            elif score_type == "f1":
                if not intersection:
                    score = 0.0
                else:
                    precision = len(intersection) / len(entities1)
                    recall = len(intersection) / len(entities2)
                    score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                logger.error(f"Unknown entity score type: {score_type}")
                raise FallbackCalculationException(f"Unknown entity score type: {score_type}")

            logger.debug(f"Entity {score_type} calculated: {score:.3f}")
            return score

        except Exception as e:
            logger.error(f"Error calculating entity overlap: {e}")
            raise FallbackCalculationException(f"Entity overlap calculation failed: {e}")

    @staticmethod
    def _normalize_context(context: Union[str, List[str]]) -> str:
        """
        Normalize context to string format.

        Args:
            context: Context as string or list of strings

        Returns:
            str: Normalized context string
        """
        try:
            if isinstance(context, list):
                normalized = " ".join(str(item) for item in context if item)
                logger.debug(
                    f"Normalized list context to string: {len(normalized)} chars")
                return normalized
            elif isinstance(context, str):
                logger.debug(
                    f"Context already string: {len(context)} chars")
                return context
            else:
                logger.warning(
                    f"Unexpected context type: {type(context)}")
                return str(context) if context else ""

        except Exception as e:
            logger.error(f"Error normalizing context: {e}")
            raise FallbackCalculationException(f"Context normalization failed: {e}")


# Initialize fallback calculator
fallback_calc = FallbackCalculator()


async def get_ragas_llm():
    """Get or create a cached RAGAS LLM wrapper."""
    if not RAGAS_AVAILABLE:
        logger.warning(f"RAGAS not available")
        return None

    if "ragas_llm" not in _cache:
        try:
            logger.info(f"Initializing RAGAS LLM wrapper")
            from langchain_openai import AzureChatOpenAI
            from backend.app.core.config import settings

            # Initialize Azure OpenAI client for evaluation
            llm = AzureChatOpenAI(
                openai_api_key=settings.AZURE_OPENAI_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
                api_version=settings.AZURE_OPENAI_VERSION,
                temperature=0.0,
                # Add rate limiting settings
                max_retries=5,
                timeout=60.0
            )

            # Wrap in the RAGAS LLM interface
            _cache["ragas_llm"] = LangchainLLMWrapper(llm)
            logger.info(f"Successfully initialized RAGAS LLM wrapper")
        except Exception as e:
            logger.error(f"Error initializing RAGAS LLM: {e}")
            raise RAGASInitializationException(f"Failed to initialize RAGAS LLM: {e}")

    return _cache["ragas_llm"]


async def get_ragas_embeddings():
    """Get or create a cached embeddings model for RAGAS."""
    if not RAGAS_AVAILABLE:
        logger.warning(f"RAGAS not available")
        return None

    if "ragas_embeddings" not in _cache:
        try:
            logger.info(f"Initializing RAGAS embeddings")
            from langchain_openai import AzureOpenAIEmbeddings
            from backend.app.core.config import settings

            # Initialize Azure OpenAI embeddings with rate limit handling
            embeddings = AzureOpenAIEmbeddings(
                openai_api_key=settings.AZURE_OPENAI_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                azure_deployment=settings.AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
                api_version=settings.AZURE_OPENAI_EMBEDDINGS_VERSION,
                max_retries=5,
                timeout=60.0
            )

            _cache["ragas_embeddings"] = embeddings
            logger.info(
                f"Successfully initialized embeddings model for RAGAS")
        except Exception as e:
            logger.error(f"Error initializing embeddings model: {e}")
            raise RAGASInitializationException(f"Failed to initialize embeddings: {e}")

    return _cache["ragas_embeddings"]


def _validate_sample_fields(sample: Dict[str, Any], required_fields: List[str], metric_name: str) -> None:
    """
    Validate that sample contains required fields.

    Args:
        sample: Sample data dictionary
        required_fields: List of required field names
        metric_name: Name of the metric for error reporting

    Raises:
        RAGASValidationException: If required fields are missing
    """
    missing_fields = [field for field in required_fields if field not in sample or not sample[field]]
    if missing_fields:
        error_msg = f"Missing required fields for {metric_name}: {missing_fields}"
        logger.error(f"{error_msg}")
        raise RAGASValidationException(error_msg)

    logger.debug(f"Sample validation passed for {metric_name}")


async def _calculate_with_ragas(
        metric_class, sample: Dict[str, Any], requires_reference: bool = False,
        requires_embeddings: bool = False
) -> Optional[float]:
    """Generic function to calculate metrics using RAGAS with enhanced error handling."""
    if not RAGAS_AVAILABLE:
        logger.warning(
            f"RAGAS not available for {metric_class.__name__}")
        return None

    # Check if required fields are present
    required_fields = ["query", "answer", "context"]
    if requires_reference:
        required_fields.append("ground_truth")

    try:
        _validate_sample_fields(sample, required_fields, metric_class.__name__)
    except RAGASValidationException:
        return None

    try:
        logger.info(
            f"Starting RAGAS calculation for {metric_class.__name__}")

        # Get RAGAS LLM
        try:
            ragas_llm = await get_ragas_llm()
            if not ragas_llm:
                raise RAGASInitializationException("Could not initialize RAGAS LLM")
        except Exception as e:
            logger.error(
                f"LLM initialization failed for {metric_class.__name__}: {e}")
            raise RAGASAPIException(f"LLM initialization failed: {e}")

        # Get embeddings if required
        embeddings = None
        if requires_embeddings:
            try:
                embeddings = await get_ragas_embeddings()
                if not embeddings:
                    raise RAGASInitializationException("Could not initialize embeddings")
            except Exception as e:
                logger.error(
                    f"Embeddings initialization failed for {metric_class.__name__}: {e}")
                raise RAGASAPIException(f"Embeddings initialization failed: {e}")

        # Prepare context (ensure it's a list)
        contexts = sample["context"]
        if isinstance(contexts, str):
            contexts = [contexts]

        # Special handling for metrics that have dependencies or require special initialization
        if metric_class == AnswerCorrectness:
            # AnswerCorrectness requires AnswerSimilarity instance
            if not embeddings:
                embeddings = await get_ragas_embeddings()
                if not embeddings:
                    raise RAGASInitializationException(
                        "Could not initialize embeddings for AnswerSimilarity dependency")

            # First initialize AnswerSimilarity (which only needs embeddings)
            answer_similarity = AnswerSimilarity(embeddings=embeddings)

            # Then initialize AnswerCorrectness with all required dependencies
            metric_instance = metric_class(
                llm=ragas_llm,
                embeddings=embeddings,
                answer_similarity=answer_similarity
            )
            logger.info(
                f"Created {metric_class.__name__} with LLM, embeddings, and AnswerSimilarity dependency")
        elif metric_class == AnswerSimilarity:
            # AnswerSimilarity only uses embeddings, not LLM
            metric_instance = metric_class(embeddings=embeddings)
            logger.info(
                f"Created {metric_class.__name__} with embeddings only")
        elif requires_embeddings:
            # Other metrics that need both LLM and embeddings
            metric_instance = metric_class(llm=ragas_llm, embeddings=embeddings)
            logger.info(
                f"Created {metric_class.__name__} with LLM and embeddings")
        else:
            # Metrics that only need LLM
            metric_instance = metric_class(llm=ragas_llm)
            logger.info(f"Created {metric_class.__name__} with LLM only")

        # Create SingleTurnSample with the correct field mapping for RAGAS
        ragas_sample = SingleTurnSample(
            user_input=sample["query"],
            response=sample["answer"],
            retrieved_contexts=contexts,
            reference=sample.get("ground_truth", "")  # Ensure reference is not None
        )

        # Calculate score with proper timeout handling
        logger.info(f"Calculating {metric_class.__name__} using RAGAS")
        start_time = time.time()

        try:
            score = await asyncio.wait_for(
                metric_instance.single_turn_ascore(ragas_sample),
                timeout=120.0  # 2-minute timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout calculating {metric_class.__name__}")
            raise RAGASTimeoutException(f"Timeout calculating {metric_class.__name__}")

        calculation_time = time.time() - start_time
        logger.info(
            f"{metric_class.__name__} score: {score} (took {calculation_time:.2f}s)")
        return float(score)

    except (RAGASTimeoutException, RAGASAPIException, RAGASInitializationException):
        # Re-raise specific RAGAS exceptions
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error calculating RAGAS metric {metric_class.__name__}: {e}",
            exc_info=True)
        raise RAGASException(f"Unexpected error in {metric_class.__name__}: {e}")


# Batch Processing Implementation
async def calculate_metrics_batch(
        samples: List[Dict[str, Any]],
        metric_names: List[str],
        max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    """
    Calculate multiple metrics for multiple samples efficiently.

    Args:
        samples: List of sample dictionaries
        metric_names: List of metric names to calculate
        max_concurrent: Maximum number of concurrent calculations

    Returns:
        List of dictionaries with results for each sample
    """
    logger.info(
        f"Starting batch calculation for {len(samples)} samples and {len(metric_names)} metrics")

    # Create semaphore to limit concurrent operations
    semaphore = asyncio.Semaphore(max_concurrent)

    async def calculate_single_metric(sample_idx: int, sample: Dict[str, Any], metric_name: str) -> Dict[str, Any]:
        """Calculate a single metric for a single sample."""
        async with semaphore:
            try:
                logger.debug(
                    f"Calculating {metric_name} for sample {sample_idx}")

                # Get the calculation function
                if metric_name not in METRIC_REQUIREMENTS:
                    logger.error(f"Unknown metric: {metric_name}")
                    return {
                        "sample_idx": sample_idx,
                        "metric_name": metric_name,
                        "score": None,
                        "error": f"Unknown metric: {metric_name}",
                        "calculation_time": 0.0
                    }

                calc_func = METRIC_REQUIREMENTS[metric_name]["calculation_func"]
                required_fields = METRIC_REQUIREMENTS[metric_name]["required_fields"]

                # Check if sample has required fields
                missing_fields = [field for field in required_fields if field not in sample or not sample[field]]
                if missing_fields:
                    logger.warning(
                        f"Sample {sample_idx} missing fields {missing_fields} for {metric_name}")
                    return {
                        "sample_idx": sample_idx,
                        "metric_name": metric_name,
                        "score": None,
                        "error": f"Missing required fields: {missing_fields}",
                        "calculation_time": 0.0
                    }

                # Calculate metric
                start_time = time.time()

                # Call the appropriate calculation function with the right parameters
                if metric_name == "faithfulness":
                    score = await calc_func(sample["answer"], sample["context"])
                elif metric_name == "response_relevancy":
                    score = await calc_func(sample["answer"], sample["query"], sample.get("context"))
                elif metric_name == "context_precision":
                    score = await calc_func(sample["context"], sample["query"])
                elif metric_name == "context_recall":
                    score = await calc_func(sample["context"], sample["query"], sample["ground_truth"])
                elif metric_name == "context_entity_recall":
                    score = await calc_func(sample["context"], sample["ground_truth"])
                elif metric_name == "noise_sensitivity":
                    score = await calc_func(sample["query"], sample["answer"], sample["context"],
                                            sample["ground_truth"])
                elif metric_name == "answer_correctness":
                    score = await calc_func(sample["answer"], sample["ground_truth"])
                elif metric_name == "answer_similarity":
                    score = await calc_func(sample["answer"], sample["ground_truth"])
                elif metric_name == "answer_relevancy":
                    score = await calc_func(sample["query"], sample["answer"], sample.get("context"))
                elif metric_name == "factual_correctness":
                    score = await calc_func(sample["answer"], sample["context"], sample.get("ground_truth"))
                else:
                    raise ValueError(f"Unsupported metric: {metric_name}")

                calculation_time = time.time() - start_time

                logger.debug(
                    f"Completed {metric_name} for sample {sample_idx}: {score}")

                return {
                    "sample_idx": sample_idx,
                    "metric_name": metric_name,
                    "score": score,
                    "error": None,
                    "calculation_time": calculation_time
                }

            except Exception as e:
                calculation_time = time.time() - start_time
                logger.error(
                    f"Error calculating {metric_name} for sample {sample_idx}: {e}")
                return {
                    "sample_idx": sample_idx,
                    "metric_name": metric_name,
                    "score": None,
                    "error": str(e),
                    "calculation_time": calculation_time
                }

    # Create all tasks
    tasks = []
    for sample_idx, sample in enumerate(samples):
        for metric_name in metric_names:
            task = calculate_single_metric(sample_idx, sample, metric_name)
            tasks.append(task)

    logger.info(f"Created {len(tasks)} calculation tasks")

    # Execute all tasks
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time

    logger.info(f"Batch calculation completed in {total_time:.2f}s")

    # Process results into structured format
    structured_results = []
    for sample_idx in range(len(samples)):
        sample_results = {
            "sample_idx": sample_idx,
            "metrics": {},
            "errors": {},
            "calculation_times": {}
        }

        # Find results for this sample
        for result in results:
            if isinstance(result, Exception):
                logger.error(
                    f"Task exception in batch processing: {result}")
                continue

            if result["sample_idx"] == sample_idx:
                metric_name = result["metric_name"]
                sample_results["metrics"][metric_name] = result["score"]
                sample_results["errors"][metric_name] = result["error"]
                sample_results["calculation_times"][metric_name] = result["calculation_time"]

        structured_results.append(sample_results)

    logger.info(f"Processed {len(structured_results)} sample results")
    return structured_results


async def calculate_faithfulness(answer: str, context: Union[str, List[str]]) -> float:
    """
    Calculate faithfulness score using RAGAS.

    Args:
        answer: LLM answer
        context: Input context (string or list of strings)

    Returns:
        float: Faithfulness score (0-1)
    """
    logger.debug(f"Calculating faithfulness")

    # Try using RAGAS
    if RAGAS_AVAILABLE:
        try:
            result = await _calculate_with_ragas(
                Faithfulness,
                {"query": "", "answer": answer, "context": context}
            )
            if result is not None:
                return result
        except Exception as e:
            logger.warning(
                f"RAGAS faithfulness failed, using fallback: {e}")

    # Fallback implementation using consolidated logic
    try:
        logger.info(f"Using fallback calculation for faithfulness")

        if not answer or not context:
            logger.warning(f"Empty inputs for faithfulness fallback")
            return 0.0

        normalized_context = fallback_calc._normalize_context(context)
        score = fallback_calc._calculate_token_overlap_score(answer, normalized_context, "precision")

        logger.info(f"Faithfulness fallback score: {score}")
        return score

    except Exception as e:
        logger.error(f"Faithfulness fallback calculation failed: {e}")
        return 0.0


async def calculate_response_relevancy(answer: str, query: str,
                                       context: Optional[Union[str, List[str]]] = None) -> float:
    """
    Calculate response relevancy score using RAGAS.

    Args:
        answer: LLM answer
        query: User query
        context: Optional context for RAGAS

    Returns:
        float: Response relevancy score (0-1)
    """
    logger.debug(f"Calculating response relevancy")

    # Try using RAGAS
    if RAGAS_AVAILABLE:
        try:
            sample = {"query": query, "answer": answer}
            if context is not None:
                sample["context"] = context
            else:
                sample["context"] = [""]  # Empty context as placeholder

            # ResponseRelevancy requires embeddings
            result = await _calculate_with_ragas(
                ResponseRelevancy,
                sample,
                requires_embeddings=True
            )
            if result is not None:
                return result
        except Exception as e:
            logger.warning(
                f"RAGAS response relevancy failed, using fallback: {e}")

    # Fallback implementation using consolidated logic
    try:
        logger.info(f"Using fallback calculation for response relevancy")

        if not answer or not query:
            logger.warning(
                f"Empty inputs for response relevancy fallback")
            return 0.0

        # Enhanced relevancy calculation considering question words
        query_words, _ = fallback_calc._tokenize_and_extract_features(query)
        answer_words, _ = fallback_calc._tokenize_and_extract_features(answer)

        if not query_words:
            logger.warning(f"No query words found")
            return 0.0

        # Consider key question terms as more important
        question_terms = {"what", "how", "why", "when", "where", "who", "which"}
        query_question_words = query_words.intersection(question_terms)

        # Calculate weighted score
        regular_overlap = query_words.intersection(answer_words)

        if query_question_words:
            question_overlap = query_question_words.intersection(answer_words)
            if not question_overlap:
                # Penalize not addressing question words
                score = len(regular_overlap) / (len(query_words) * 2)
            else:
                # Bonus for addressing question words
                score = (len(regular_overlap) + len(question_overlap)) / (len(query_words) + len(query_question_words))
            score = min(score, 1.0)
        else:
            # Simple overlap for non-question queries
            score = len(regular_overlap) / len(query_words)

        logger.info(f"Response relevancy fallback score: {score}")
        return score

    except Exception as e:
        logger.error(
            f"Response relevancy fallback calculation failed: {e}")
        return 0.0


async def calculate_context_precision(context: Union[str, List[str]], query: str) -> float:
    """
    Calculate context precision score using RAGAS.

    Args:
        context: Input context (string or list of strings)
        query: User query

    Returns:
        float: Context precision score (0-1)
    """
    logger.debug(f"Calculating context precision")

    # Try using RAGAS
    if RAGAS_AVAILABLE:
        try:
            # ContextPrecision requires a response and empty reference
            placeholder_answer = "This is a placeholder answer for context precision evaluation."
            result = await _calculate_with_ragas(
                ContextPrecision,
                {
                    "query": query,
                    "answer": placeholder_answer,
                    "context": context,
                    "ground_truth": ""  # Empty reference - crucial!
                }
            )
            if result is not None:
                return result
        except Exception as e:
            logger.warning(
                f"RAGAS context precision failed, using fallback: {e}")

    # Fallback implementation using consolidated logic
    try:
        logger.info(f"Using fallback calculation for context precision")

        if not context or not query:
            logger.warning(
                f"Empty inputs for context precision fallback")
            return 0.0

        normalized_context = fallback_calc._normalize_context(context)
        score = fallback_calc._calculate_token_overlap_score(query, normalized_context, "recall")

        logger.info(f"Context precision fallback score: {score}")
        return score

    except Exception as e:
        logger.error(
            f"Context precision fallback calculation failed: {e}")
        return 0.0


async def calculate_context_recall(context: Union[str, List[str]], query: str, ground_truth: str) -> float:
    """
    Calculate context recall score using RAGAS.

    Args:
        context: Input context (string or list of strings)
        query: User query
        ground_truth: Expected answer

    Returns:
        float: Context recall score (0-1)
    """
    logger.debug(f"Calculating context recall")

    # Try using RAGAS
    if RAGAS_AVAILABLE:
        try:
            # ContextRecall requires a reference/ground_truth
            placeholder_answer = "This is a placeholder answer for context recall evaluation."
            result = await _calculate_with_ragas(
                ContextRecall,
                {
                    "query": query,
                    "answer": placeholder_answer,
                    "context": context,
                    "ground_truth": ground_truth
                },
                requires_reference=True
            )
            if result is not None:
                return result
        except Exception as e:
            logger.warning(
                f"RAGAS context recall failed, using fallback: {e}")

    # Fallback implementation using consolidated logic
    try:
        logger.info(f"Using fallback calculation for context recall")

        if not context or not ground_truth:
            logger.warning(f"Empty inputs for context recall fallback")
            return 0.0

        normalized_context = fallback_calc._normalize_context(context)
        score = fallback_calc._calculate_token_overlap_score(ground_truth, normalized_context, "recall")

        logger.info(f"Context recall fallback score: {score}")
        return score

    except Exception as e:
        logger.error(f"Context recall fallback calculation failed: {e}")
        return 0.0


async def calculate_context_entity_recall(context: Union[str, List[str]], ground_truth: str) -> float:
    """
    Calculate context entity recall score using RAGAS.

    Args:
        context: Input context (string or list of strings)
        ground_truth: Expected answer

    Returns:
        float: Context entity recall score (0-1)
    """
    logger.debug(f"Calculating context entity recall")

    # Try using RAGAS
    if RAGAS_AVAILABLE:
        try:
            # ContextEntityRecall just needs context and reference
            placeholder_query = "This is a placeholder query for context entity recall."
            placeholder_answer = "This is a placeholder answer for context entity recall evaluation."
            result = await _calculate_with_ragas(
                ContextEntityRecall,
                {
                    "query": placeholder_query,
                    "answer": placeholder_answer,
                    "context": context,
                    "ground_truth": ground_truth
                },
                requires_reference=True
            )
            if result is not None:
                return result
        except Exception as e:
            logger.warning(
                f"RAGAS context entity recall failed, using fallback: {e}")

    # Fallback implementation using consolidated logic
    try:
        logger.info(
            f"Using fallback calculation for context entity recall")

        if not context or not ground_truth:
            logger.warning(
                f"Empty inputs for context entity recall fallback")
            return 0.0

        normalized_context = fallback_calc._normalize_context(context)
        score = fallback_calc._calculate_entity_overlap(normalized_context, ground_truth, "recall")

        logger.info(f"Context entity recall fallback score: {score}")
        return score

    except Exception as e:
        logger.error(
            f"Context entity recall fallback calculation failed: {e}")
        return 0.0


async def calculate_noise_sensitivity(query: str, answer: str, context: Union[str, List[str]],
                                      ground_truth: str) -> float:
    """
    Calculate noise sensitivity score using RAGAS.
    Lower scores are better for this metric (0 is best).

    Args:
        query: User query
        answer: LLM answer
        context: Input context (string or list of strings)
        ground_truth: Expected answer

    Returns:
        float: Noise sensitivity score (0-1)
    """
    logger.debug(f"Calculating noise sensitivity")

    # Try using RAGAS
    if RAGAS_AVAILABLE:
        try:
            result = await _calculate_with_ragas(
                NoiseSensitivity,
                {
                    "query": query,
                    "answer": answer,
                    "context": context,
                    "ground_truth": ground_truth
                },
                requires_reference=True
            )
            if result is not None:
                return result
        except Exception as e:
            logger.warning(
                f"RAGAS noise sensitivity failed, using fallback: {e}")

    # Fallback implementation using consolidated logic
    try:
        logger.info(f"Using fallback calculation for noise sensitivity")

        # For noise sensitivity, lower is better, so we invert the correctness score
        if not answer or not ground_truth:
            logger.warning(
                f"Empty inputs for noise sensitivity fallback")
            return 1.0  # Worst score if missing inputs

        # Calculate F1-like score for correctness
        f1_score = fallback_calc._calculate_token_overlap_score(answer, ground_truth, "f1")

        # Invert to get noise sensitivity (1 - correctness)
        score = min(1.0, 1.0 - f1_score)

        logger.info(f"Noise sensitivity fallback score: {score}")
        return score

    except Exception as e:
        logger.error(
            f"Noise sensitivity fallback calculation failed: {e}")
        return 1.0  # Worst score on error


async def calculate_answer_correctness(answer: str, ground_truth: str) -> float:
    """
    Calculate answer correctness score using RAGAS.

    Args:
        answer: LLM answer
        ground_truth: Expected answer

    Returns:
        float: Correctness score (0-1)
    """
    logger.debug(f"Calculating answer correctness")

    # Try using RAGAS
    if RAGAS_AVAILABLE:
        try:
            placeholder_query = "This is a placeholder query for answer correctness."
            placeholder_context = ["This is a placeholder context for answer correctness evaluation."]

            result = await _calculate_with_ragas(
                AnswerCorrectness,
                {
                    "query": placeholder_query,
                    "answer": answer,
                    "context": placeholder_context,
                    "ground_truth": ground_truth
                },
                requires_reference=True,
                requires_embeddings=True
            )
            if result is not None:
                return result
        except Exception as e:
            logger.warning(
                f"RAGAS answer correctness failed, using fallback: {e}")

    # Fallback implementation using consolidated logic
    try:
        logger.info(f"Using fallback calculation for answer correctness")

        if not answer or not ground_truth:
            logger.warning(
                f"Empty inputs for answer correctness fallback")
            return 0.0

        score = fallback_calc._calculate_token_overlap_score(answer, ground_truth, "f1")

        logger.info(f"Answer correctness fallback score: {score}")
        return score

    except Exception as e:
        logger.error(
            f"Answer correctness fallback calculation failed: {e}")
        return 0.0


async def calculate_answer_similarity(answer: str, ground_truth: str) -> float:
    """
    Calculate answer similarity score using RAGAS.

    Args:
        answer: LLM answer
        ground_truth: Expected answer

    Returns:
        float: Similarity score (0-1)
    """
    logger.debug(f"Calculating answer similarity")

    # Try using RAGAS
    if RAGAS_AVAILABLE:
        try:
            placeholder_query = "This is a placeholder query for answer similarity."
            placeholder_context = ["This is a placeholder context for answer similarity evaluation."]

            result = await _calculate_with_ragas(
                AnswerSimilarity,
                {
                    "query": placeholder_query,
                    "answer": answer,
                    "context": placeholder_context,
                    "ground_truth": ground_truth
                },
                requires_reference=True,
                requires_embeddings=True
            )
            if result is not None:
                return result
        except Exception as e:
            logger.warning(
                f"RAGAS answer similarity failed, using fallback: {e}")

    # Fallback implementation using consolidated logic
    try:
        logger.info(f"Using fallback calculation for answer similarity")

        if not answer or not ground_truth:
            logger.warning(
                f"Empty inputs for answer similarity fallback")
            return 0.0

        # Use Jaccard similarity as it measures similarity well
        score = fallback_calc._calculate_token_overlap_score(answer, ground_truth, "jaccard")

        logger.info(f"Answer similarity fallback score: {score}")
        return score

    except Exception as e:
        logger.error(
            f"Answer similarity fallback calculation failed: {e}")
        return 0.0


async def calculate_answer_relevancy(query: str, answer: str, context: Optional[Union[str, List[str]]] = None) -> float:
    """
    Calculate answer relevancy score using RAGAS.

    Args:
        query: User query
        answer: LLM answer
        context: Optional context

    Returns:
        float: Answer relevancy score (0-1)
    """
    logger.debug(f"Calculating answer relevancy")

    # Try using RAGAS
    if RAGAS_AVAILABLE:
        try:
            sample = {
                "query": query,
                "answer": answer,
                "context": context if context else [""]
            }

            result = await _calculate_with_ragas(
                AnswerRelevancy,
                sample,
                requires_embeddings=True
            )
            if result is not None:
                return result
        except Exception as e:
            logger.warning(
                f"RAGAS answer relevancy failed, using fallback: {e}")

    # Fallback to response relevancy as they measure similar things
    try:
        logger.info(
            f"Using response relevancy fallback for answer relevancy")
        return await calculate_response_relevancy(answer, query, context)
    except Exception as e:
        logger.error(
            f"Answer relevancy fallback calculation failed: {e}")
        return 0.0


async def calculate_factual_correctness(answer: str, context: Union[str, List[str]],
                                        ground_truth: Optional[str] = None) -> float:
    """
    Calculate factual correctness score using RAGAS.

    Args:
        answer: LLM answer
        context: Input context (string or list of strings)
        ground_truth: Optional ground truth answer (if not provided, context will be used as reference)

    Returns:
        float: Factual correctness score (0-1)
    """
    logger.debug(f"Calculating factual correctness")

    # Try using RAGAS
    if RAGAS_AVAILABLE:
        try:
            placeholder_query = "This is a placeholder query for factual correctness."

            # If ground_truth is not provided, use context as reference
            reference = ground_truth if ground_truth else (
                context if isinstance(context, str) else " ".join(context)
            )

            result = await _calculate_with_ragas(
                FactualCorrectness,
                {
                    "query": placeholder_query,
                    "answer": answer,
                    "context": context,
                    "ground_truth": reference
                },
                requires_reference=True
            )
            if result is not None:
                return result
        except Exception as e:
            logger.warning(
                f"RAGAS factual correctness failed, using fallback: {e}")

    # Fallback to faithfulness as they measure similar things
    try:
        logger.info(
            f"Using faithfulness fallback for factual correctness")
        return await calculate_faithfulness(answer, context)
    except Exception as e:
        logger.error(
            f"Factual correctness fallback calculation failed: {e}")
        return 0.0


# Mapping of metrics to their required dataset fields and calculation functions
METRIC_REQUIREMENTS = {
    "faithfulness": {
        "required_fields": ["answer", "context"],
        "calculation_func": calculate_faithfulness,
        "description": "Measures how well the answer sticks to the information in the context without hallucinating."
    },
    "response_relevancy": {
        "required_fields": ["answer", "query"],
        "calculation_func": calculate_response_relevancy,
        "description": "Measures how relevant the answer is to the query asked."
    },
    "context_precision": {
        "required_fields": ["context", "query"],
        "calculation_func": calculate_context_precision,
        "description": "Measures how precisely the retrieved context matches what's needed to answer the query."
    },
    "context_recall": {
        "required_fields": ["context", "query", "ground_truth"],
        "calculation_func": calculate_context_recall,
        "description": "Measures how well the retrieved context covers all the information needed to answer the query."
    },
    "context_entity_recall": {
        "required_fields": ["context", "ground_truth"],
        "calculation_func": calculate_context_entity_recall,
        "description": "Measures how well the retrieved context captures the entities mentioned in the reference answer."
    },
    "noise_sensitivity": {
        "required_fields": ["query", "answer", "context", "ground_truth"],
        "calculation_func": calculate_noise_sensitivity,
        "description": "Measures the model's tendency to be misled by irrelevant information (lower is better)."
    },
    "answer_correctness": {
        "required_fields": ["answer", "ground_truth"],
        "calculation_func": calculate_answer_correctness,
        "description": "Measures how accurately the answer matches the ground truth."
    },
    "answer_similarity": {
        "required_fields": ["answer", "ground_truth"],
        "calculation_func": calculate_answer_similarity,
        "description": "Measures the semantic similarity between the answer and the ground truth."
    },
    "answer_relevancy": {
        "required_fields": ["query", "answer"],
        "calculation_func": calculate_answer_relevancy,
        "description": "Measures how relevant the answer is to the query."
    },
    "factual_correctness": {
        "required_fields": ["answer", "context", "ground_truth"],
        "calculation_func": calculate_factual_correctness,
        "description": "Measures how factually accurate the answer is based on the provided context."
    }
}

# Mapping of which metrics can be applied to which dataset types
DATASET_TYPE_METRICS = {
    "user_query": [
        "faithfulness",
        "response_relevancy",
        "context_precision",
        "answer_relevancy",
        "factual_correctness"
    ],
    "context": [
        "context_precision",
        "context_recall",
        "context_entity_recall",
        "factual_correctness"
    ],
    "question_answer": [
        "faithfulness",
        "response_relevancy",
        "context_recall",
        "noise_sensitivity",
        "answer_correctness",
        "answer_similarity",
        "answer_relevancy",
        "factual_correctness"
    ],
    "conversation": [
        "response_relevancy",
        "faithfulness",
        "answer_relevancy"
    ],
    "custom": [
        "faithfulness",
        "response_relevancy",
        "context_precision",
        "context_recall",
        "context_entity_recall",
        "noise_sensitivity",
        "answer_correctness",
        "answer_similarity",
        "answer_relevancy",
        "factual_correctness"
    ]
}
