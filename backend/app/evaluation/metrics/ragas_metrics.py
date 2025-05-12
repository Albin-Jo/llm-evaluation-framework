# backend/app/evaluation/metrics/ragas_metrics.py
import logging
from typing import Dict, List, Optional, Union, Any

import ragas
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextEntityRecall,
    AnswerCorrectness,
    AnswerRelevancy,
    AnswerSimilarity,
    FactualCorrectness,
    ContextPrecision,
    ContextRecall,
    NoiseSensitivity,
    AspectCritic,
    TopicAdherenceScore
)

logger = logging.getLogger(__name__)
RAGAS_AVAILABLE = True
logger.info(f"RAGAS library found (version: {ragas.__version__})")

# Cache for LLM and embedding instances to avoid recreating them
_cache = {}


async def get_ragas_llm():
    """Get or create a cached RAGAS LLM wrapper."""
    if not RAGAS_AVAILABLE:
        return None

    if "ragas_llm" not in _cache:
        try:
            from langchain_openai import AzureChatOpenAI
            from backend.app.core.config import settings

            # Initialize Azure OpenAI client for evaluation
            llm = AzureChatOpenAI(
                openai_api_key=settings.AZURE_OPENAI_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
                api_version=settings.AZURE_OPENAI_VERSION,
                temperature=0.0
            )

            # Wrap in the RAGAS LLM interface
            _cache["ragas_llm"] = LangchainLLMWrapper(llm)
            logger.info("Successfully initialized RAGAS LLM wrapper")
        except Exception as e:
            logger.error(f"Error initializing RAGAS LLM: {e}")
            return None

    return _cache["ragas_llm"]


async def get_ragas_embeddings():
    """Get or create a cached embeddings model for RAGAS."""
    if not RAGAS_AVAILABLE:
        return None

    if "ragas_embeddings" not in _cache:
        try:
            from langchain_openai import AzureOpenAIEmbeddings
            from backend.app.core.config import settings

            # Initialize Azure OpenAI embeddings
            embeddings = AzureOpenAIEmbeddings(
                openai_api_key=settings.AZURE_OPENAI_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                azure_deployment=settings.AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
                api_version=settings.AZURE_OPENAI_EMBEDDINGS_VERSION,
            )

            _cache["ragas_embeddings"] = embeddings

            logger.info(f"embeddings: {embeddings}")
            logger.info("Successfully initialized embeddings model for RAGAS")
        except Exception as e:
            logger.error(f"Error initializing embeddings model: {e}")
            return None

    return _cache["ragas_embeddings"]


async def _calculate_with_ragas(
        metric_class, sample: Dict[str, Any], requires_reference: bool = False,
        requires_embeddings: bool = False
) -> Optional[float]:
    """Generic function to calculate metrics using RAGAS."""
    if not RAGAS_AVAILABLE:
        return None

    # Check if required fields are present
    if "query" not in sample or "answer" not in sample or "context" not in sample:
        logger.warning(f"Missing required fields for {metric_class.__name__}. Needed: query, answer, context")
        return None

    if requires_reference and "ground_truth" not in sample:
        logger.warning(f"Missing ground_truth field required for {metric_class.__name__}")
        return None

    try:
        # Get RAGAS LLM
        ragas_llm = await get_ragas_llm()
        if not ragas_llm:
            logger.error(f"Could not initialize RAGAS LLM for {metric_class.__name__}")
            return None

        # Get embeddings if required
        embeddings = None
        if requires_embeddings:
            embeddings = await get_ragas_embeddings()
            if not embeddings:
                logger.error(f"Embeddings required for {metric_class.__name__} but could not be initialized")
                return None

        # Create metric instance with appropriate parameters
        if requires_embeddings:
            metric_instance = metric_class(llm=ragas_llm, embeddings=embeddings)
            logger.debug(f"Created {metric_class.__name__} with LLM and embeddings")
        else:
            metric_instance = metric_class(llm=ragas_llm)
            logger.debug(f"Created {metric_class.__name__} with LLM only")

        # Prepare context (ensure it's a list)
        contexts = sample["context"]
        if isinstance(contexts, str):
            contexts = [contexts]

        # Create SingleTurnSample
        ragas_sample = SingleTurnSample(
            user_input=sample["query"],
            response=sample["answer"],
            retrieved_contexts=contexts,
            reference=sample.get("ground_truth")
        )

        # Calculate score
        logger.info(f"Calculating {metric_class.__name__} using RAGAS")
        score = await metric_instance.single_turn_ascore(ragas_sample)
        logger.info(f"{metric_class.__name__} score: {score}")
        return float(score)

    except Exception as e:
        logger.error(f"Error calculating RAGAS metric {metric_class.__name__}: {e}", exc_info=True)
        return None


# Existing metric implementations
async def calculate_faithfulness(answer: str, context: Union[str, List[str]]) -> float:
    """
    Calculate faithfulness score using RAGAS if available.

    Args:
        answer: LLM answer
        context: Input context (string or list of strings)

    Returns:
        float: Faithfulness score (0-1)
    """
    # Try using RAGAS
    if RAGAS_AVAILABLE:
        result = await _calculate_with_ragas(
            Faithfulness,
            {"query": "", "answer": answer, "context": context}
        )
        if result is not None:
            return result

    # Fallback implementation
    return _calculate_faithfulness_fallback(answer, context)


def _calculate_faithfulness_fallback(answer: str, context: Union[str, List[str]]) -> float:
    """Simple fallback implementation for faithfulness when RAGAS is not available."""
    if not answer or not context:
        return 0.0

    # Ensure context is a string for fallback
    if isinstance(context, list):
        context = " ".join(context)

    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    if not answer_words:
        return 0.0

    overlap = answer_words.intersection(context_words)
    return len(overlap) / len(answer_words)


async def calculate_response_relevancy(answer: str, query: str,
                                       context: Optional[Union[str, List[str]]] = None) -> float:
    """
    Calculate response relevancy score using RAGAS if available.

    Args:
        answer: LLM answer
        query: User query
        context: Optional context for RAGAS

    Returns:
        float: Response relevancy score (0-1)
    """
    # Try using RAGAS
    if RAGAS_AVAILABLE:
        sample = {"query": query, "answer": answer}
        if context is not None:
            sample["context"] = context
        else:
            sample["context"] = [""]  # Empty context as placeholder

        # Note: ResponseRelevancy requires embeddings
        result = await _calculate_with_ragas(
            ResponseRelevancy,
            sample,
            requires_embeddings=True
        )
        if result is not None:
            return result

    # Fallback implementation
    return _calculate_response_relevancy_fallback(answer, query)


def _calculate_response_relevancy_fallback(answer: str, query: str) -> float:
    """Simple fallback implementation for response relevancy when RAGAS is not available."""
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


async def calculate_context_precision(context: Union[str, List[str]], query: str) -> float:
    """
    Calculate context precision score using RAGAS if available.

    Args:
        context: Input context (string or list of strings)
        query: User query

    Returns:
        float: Context precision score (0-1)
    """
    # Try using RAGAS
    if RAGAS_AVAILABLE:
        # ContextPrecision doesn't need an answer but the API requires it
        placeholder_answer = "This is a placeholder answer for context precision evaluation."
        result = await _calculate_with_ragas(
            ContextPrecision,
            {"query": query, "answer": placeholder_answer, "context": context}
        )
        if result is not None:
            return result

    # Fallback implementation
    return _calculate_context_precision_fallback(context, query)


def _calculate_context_precision_fallback(context: Union[str, List[str]], query: str) -> float:
    """Simple fallback implementation for context precision when RAGAS is not available."""
    if not context or not query:
        return 0.0

    # Ensure context is a string for fallback
    if isinstance(context, list):
        context = " ".join(context)

    query_words = set(query.lower().split())
    context_words = set(context.lower().split())

    if not query_words:
        return 0.0

    overlap = query_words.intersection(context_words)
    return len(overlap) / len(query_words)


async def calculate_context_recall(context: Union[str, List[str]], query: str, ground_truth: str) -> float:
    """
    Calculate context recall score using RAGAS if available.

    Args:
        context: Input context (string or list of strings)
        query: User query
        ground_truth: Expected answer

    Returns:
        float: Context recall score (0-1)
    """
    # Try using RAGAS
    if RAGAS_AVAILABLE:
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

    # Fallback implementation
    return _calculate_context_recall_fallback(context, ground_truth)


def _calculate_context_recall_fallback(context: Union[str, List[str]], ground_truth: str) -> float:
    """Simple fallback implementation for context recall when RAGAS is not available."""
    if not context or not ground_truth:
        return 0.0

    # Ensure context is a string for fallback
    if isinstance(context, list):
        context = " ".join(context)

    ground_truth_words = set(ground_truth.lower().split())
    context_words = set(context.lower().split())

    if not ground_truth_words:
        return 0.0

    # Calculate token overlap
    common_words = ground_truth_words.intersection(context_words)
    return len(common_words) / len(ground_truth_words)


async def calculate_context_entity_recall(context: Union[str, List[str]], ground_truth: str) -> float:
    """
    Calculate context entity recall score using RAGAS if available.

    Args:
        context: Input context (string or list of strings)
        ground_truth: Expected answer

    Returns:
        float: Context entity recall score (0-1)
    """
    # Try using RAGAS
    if RAGAS_AVAILABLE:
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

    # Fallback implementation
    return _calculate_entity_recall_fallback(context, ground_truth)


def _calculate_entity_recall_fallback(context: Union[str, List[str]], ground_truth: str) -> float:
    """Simple entity extraction fallback when RAGAS is not available."""
    if not context or not ground_truth:
        return 0.0

    # Ensure context is a string for fallback
    if isinstance(context, list):
        context = " ".join(context)

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


async def calculate_noise_sensitivity(query: str, answer: str, context: Union[str, List[str]],
                                      ground_truth: str) -> float:
    """
    Calculate noise sensitivity score using RAGAS if available.
    Lower scores are better for this metric (0 is best).

    Args:
        query: User query
        answer: LLM answer
        context: Input context (string or list of strings)
        ground_truth: Expected answer

    Returns:
        float: Noise sensitivity score (0-1)
    """
    # Try using RAGAS
    if RAGAS_AVAILABLE:
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

    # Fallback implementation
    return _calculate_noise_sensitivity_fallback(answer, ground_truth)


def _calculate_noise_sensitivity_fallback(answer: str, ground_truth: str) -> float:
    """Simple fallback implementation for noise sensitivity when RAGAS is not available."""
    # For noise sensitivity, lower is better, so we invert the correctness score
    # and cap at 1.0

    # First calculate correctness between answer and ground truth
    if not answer or not ground_truth:
        return 1.0  # Worst score if missing inputs

    answer_tokens = answer.lower().split()
    ground_truth_tokens = ground_truth.lower().split()

    if not ground_truth_tokens:
        return 1.0  # Worst score if no ground truth

    # Calculate F1-like score for correctness
    common_tokens = sum(1 for token in answer_tokens if token in ground_truth_tokens)

    if not common_tokens:
        return 1.0  # Completely incorrect = high noise sensitivity

    precision = common_tokens / len(answer_tokens) if answer_tokens else 0
    recall = common_tokens / len(ground_truth_tokens) if ground_truth_tokens else 0

    if precision + recall == 0:
        return 1.0  # Avoid division by zero

    f1 = 2 * (precision * recall) / (precision + recall)

    # Invert to get noise sensitivity (1 - correctness) and ensure in range [0,1]
    return min(1.0, 1.0 - f1)


# NEW METRIC IMPLEMENTATIONS

async def calculate_answer_correctness(answer: str, ground_truth: str) -> float:
    """
    Calculate answer correctness score using RAGAS if available.

    Args:
        answer: LLM answer
        ground_truth: Expected answer

    Returns:
        float: Correctness score (0-1)
    """
    # Try using RAGAS
    if RAGAS_AVAILABLE:
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
            requires_reference=True
        )
        if result is not None:
            return result

    # Fallback implementation
    return _calculate_answer_correctness_fallback(answer, ground_truth)


def _calculate_answer_correctness_fallback(answer: str, ground_truth: str) -> float:
    """Simple fallback implementation for answer correctness when RAGAS is not available."""
    if not answer or not ground_truth:
        return 0.0

    answer_tokens = answer.lower().split()
    ground_truth_tokens = ground_truth.lower().split()

    if not answer_tokens or not ground_truth_tokens:
        return 0.0

    # Calculate F1-like score
    common_tokens = sum(1 for token in answer_tokens if token in ground_truth_tokens)

    if not common_tokens:
        return 0.0

    precision = common_tokens / len(answer_tokens)
    recall = common_tokens / len(ground_truth_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


async def calculate_answer_relevancy(query: str, answer: str, context: Optional[Union[str, List[str]]] = None) -> float:
    """
    Calculate answer relevancy score using RAGAS if available.

    Args:
        query: User query
        answer: LLM answer
        context: Optional context

    Returns:
        float: Answer relevancy score (0-1)
    """
    # Try using RAGAS
    if RAGAS_AVAILABLE:
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

    # Fallback to response relevancy as they measure similar things
    return await calculate_response_relevancy(answer, query, context)


async def calculate_answer_similarity(answer: str, ground_truth: str) -> float:
    """
    Calculate answer similarity score using RAGAS if available.

    Args:
        answer: LLM answer
        ground_truth: Expected answer

    Returns:
        float: Similarity score (0-1)
    """
    # Try using RAGAS
    if RAGAS_AVAILABLE:
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

    # Fallback implementation
    return _calculate_answer_similarity_fallback(answer, ground_truth)


def _calculate_answer_similarity_fallback(answer: str, ground_truth: str) -> float:
    """Simple fallback implementation for answer similarity when RAGAS is not available."""
    if not answer or not ground_truth:
        return 0.0

    # Convert to sets of words for a simple Jaccard similarity
    answer_words = set(answer.lower().split())
    ground_truth_words = set(ground_truth.lower().split())

    if not answer_words or not ground_truth_words:
        return 0.0

    # Jaccard similarity: intersection size / union size
    intersection = answer_words.intersection(ground_truth_words)
    union = answer_words.union(ground_truth_words)

    return len(intersection) / len(union)


async def calculate_factual_correctness(answer: str, context: Union[str, List[str]]) -> float:
    """
    Calculate factual correctness score using RAGAS if available.

    Args:
        answer: LLM answer
        context: Input context (string or list of strings)

    Returns:
        float: Factual correctness score (0-1)
    """
    # Try using RAGAS
    if RAGAS_AVAILABLE:
        placeholder_query = "This is a placeholder query for factual correctness."

        result = await _calculate_with_ragas(
            FactualCorrectness,
            {
                "query": placeholder_query,
                "answer": answer,
                "context": context
            }
        )
        if result is not None:
            return result

    # Fallback to faithfulness as they measure similar things
    return await calculate_faithfulness(answer, context)


async def calculate_topic_adherence(query: str, answer: str) -> float:
    """
    Calculate topic adherence score using RAGAS if available.

    Args:
        query: User query
        answer: LLM answer

    Returns:
        float: Topic adherence score (0-1)
    """
    # Try using RAGAS
    if RAGAS_AVAILABLE:
        placeholder_context = ["This is a placeholder context for topic adherence evaluation."]

        result = await _calculate_with_ragas(
            TopicAdherenceScore,
            {
                "query": query,
                "answer": answer,
                "context": placeholder_context
            },
            requires_embeddings=True
        )
        if result is not None:
            return result

    # Fallback to response relevancy as they measure similar things
    return await calculate_response_relevancy(answer, query)


async def calculate_aspect_critic(query: str, answer: str, aspects: List[str]) -> Dict[str, float]:
    """
    Calculate aspect critic scores using RAGAS if available.

    Args:
        query: User query
        answer: LLM answer
        aspects: List of aspects to evaluate

    Returns:
        Dict[str, float]: Dictionary of aspect scores
    """
    # This is a special metric that returns multiple scores, one per aspect
    # It's not directly compatible with our current metric framework
    # So we'll just return a placeholder for now
    return {aspect: 0.5 for aspect in aspects}


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
    "answer_relevancy": {
        "required_fields": ["query", "answer"],
        "calculation_func": calculate_answer_relevancy,
        "description": "Measures how relevant the answer is to the query."
    },
    "answer_similarity": {
        "required_fields": ["answer", "ground_truth"],
        "calculation_func": calculate_answer_similarity,
        "description": "Measures the semantic similarity between the answer and the ground truth."
    },
    "factual_correctness": {
        "required_fields": ["answer", "context"],
        "calculation_func": calculate_factual_correctness,
        "description": "Measures how factually accurate the answer is based on the provided context."
    },
    "TopicAdherenceScore": {
        "required_fields": ["query", "answer"],
        "calculation_func": calculate_topic_adherence,
        "description": "Measures how well the answer stays on topic with the query."
    }
}

# Mapping of which metrics can be applied to which dataset types
DATASET_TYPE_METRICS = {
    "user_query": [
        "faithfulness",
        "response_relevancy",
        "context_precision",
        "answer_relevancy",
        "topic_adherence",
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
        "factual_correctness",
        "topic_adherence"
    ],
    "conversation": [
        "response_relevancy",
        "faithfulness",
        "answer_relevancy",
        "topic_adherence"
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
        "factual_correctness",
        "topic_adherence"
    ]
}