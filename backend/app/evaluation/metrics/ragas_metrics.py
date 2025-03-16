# File: app/evaluation/metrics/ragas_metrics.py
"""
Metrics for RAGAS evaluation.

This module provides implementations of RAGAS metrics.
"""

async def calculate_faithfulness(answer: str, context: str) -> float:
    """
    Calculate faithfulness score.

    Args:
        answer: LLM answer
        context: Input context

    Returns:
        float: Faithfulness score (0-1)
    """
    # Check if answer is based on the context
    if not answer or not context:
        return 0.0

    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    if not answer_words:
        return 0.0

    overlap = answer_words.intersection(context_words)
    return len(overlap) / len(answer_words)


async def calculate_answer_relevancy(answer: str, query: str) -> float:
    """
    Calculate answer relevancy score.

    Args:
        answer: LLM answer
        query: User query

    Returns:
        float: Answer relevancy score (0-1)
    """
    # Check if answer is relevant to the query
    if not answer or not query:
        return 0.0

    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())

    if not query_words:
        return 0.0

    overlap = query_words.intersection(answer_words)
    return len(overlap) / len(query_words)


async def calculate_context_relevancy(context: str, query: str) -> float:
    """
    Calculate context relevancy score.

    Args:
        context: Input context
        query: User query

    Returns:
        float: Context relevancy score (0-1)
    """
    # Check if context is relevant to the query
    if not context or not query:
        return 0.0

    query_words = set(query.lower().split())
    context_words = set(context.lower().split())

    if not query_words:
        return 0.0

    overlap = query_words.intersection(context_words)
    return len(overlap) / len(query_words)


async def calculate_correctness(answer: str, ground_truth: str) -> float:
    """
    Calculate correctness score.

    Args:
        answer: LLM answer
        ground_truth: Expected answer

    Returns:
        float: Correctness score (0-1)
    """
    # Check if answer matches ground truth
    if not answer or not ground_truth:
        return 0.0

    answer_tokens = answer.lower().split()
    ground_truth_tokens = ground_truth.lower().split()

    if not ground_truth_tokens:
        return 0.0

    # Calculate token overlap
    common_tokens = sum(1 for token in answer_tokens if token in ground_truth_tokens)
    return common_tokens / len(ground_truth_tokens)