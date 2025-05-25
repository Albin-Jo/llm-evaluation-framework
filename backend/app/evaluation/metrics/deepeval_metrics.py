import logging
from typing import Dict, List, Any

# Check if DeepEval is available
try:
    import deepeval
    from deepeval.metrics import (
        AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric,
        ContextualRecallMetric, ContextualRelevancyMetric, HallucinationMetric,
        ToxicityMetric, BiasMetric, GEval
    )

    DEEPEVAL_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("DeepEval library is available")
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("DeepEval library is not available. Install with: pip install deepeval")

from backend.app.db.models.orm import DatasetType

# Dataset type to supported DeepEval metrics mapping
DEEPEVAL_DATASET_TYPE_METRICS: Dict[DatasetType, List[str]] = {
    DatasetType.USER_QUERY: [
        "answer_relevancy",
        "hallucination",
        "toxicity",
        "bias",
        "g_eval_coherence",
        "g_eval_helpfulness"
    ],
    DatasetType.QUESTION_ANSWER: [
        "answer_relevancy",
        "faithfulness",
        "contextual_precision",
        "contextual_recall",
        "contextual_relevancy",
        "hallucination",
        "toxicity",
        "bias",
        "g_eval_correctness",
        "g_eval_completeness"
    ],
    DatasetType.CONTEXT: [
        "faithfulness",
        "contextual_precision",
        "contextual_recall",
        "contextual_relevancy",
        "answer_relevancy",
        "hallucination"
    ],
    DatasetType.CONVERSATION: [
        "answer_relevancy",
        "hallucination",
        "toxicity",
        "bias",
        "g_eval_coherence",
        "g_eval_helpfulness"
    ],
    DatasetType.CUSTOM: [
        "answer_relevancy",
        "faithfulness",
        "contextual_precision",
        "contextual_recall",
        "contextual_relevancy",
        "hallucination",
        "toxicity",
        "bias",
        "g_eval_coherence",
        "g_eval_correctness",
        "g_eval_completeness",
        "g_eval_helpfulness"
    ]
}

# Metric requirements and descriptions
DEEPEVAL_METRIC_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "answer_relevancy": {
        "description": "Measures how relevant the answer is to the given question",
        "required_fields": ["input", "actual_output"],
        "optional_fields": ["context"],
        "threshold_range": (0.0, 1.0),
        "higher_is_better": True,
        "category": "relevance"
    },
    "faithfulness": {
        "description": "Measures how grounded the answer is in the provided context",
        "required_fields": ["actual_output", "context"],
        "optional_fields": ["input"],
        "threshold_range": (0.0, 1.0),
        "higher_is_better": True,
        "category": "groundedness"
    },
    "contextual_precision": {
        "description": "Measures the precision of retrieved context relative to the question",
        "required_fields": ["input", "context", "expected_output"],
        "optional_fields": ["actual_output"],
        "threshold_range": (0.0, 1.0),
        "higher_is_better": True,
        "category": "retrieval"
    },
    "contextual_recall": {
        "description": "Measures how well the retrieved context covers the information needed",
        "required_fields": ["input", "context", "expected_output"],
        "optional_fields": ["actual_output"],
        "threshold_range": (0.0, 1.0),
        "higher_is_better": True,
        "category": "retrieval"
    },
    "contextual_relevancy": {
        "description": "Measures the relevance of context to the given question",
        "required_fields": ["input", "context"],
        "optional_fields": ["actual_output", "expected_output"],
        "threshold_range": (0.0, 1.0),
        "higher_is_better": True,
        "category": "retrieval"
    },
    "hallucination": {
        "description": "Detects fabricated or unsupported information in the response",
        "required_fields": ["actual_output", "context"],
        "optional_fields": ["input"],
        "threshold_range": (0.0, 1.0),
        "higher_is_better": False,  # Lower hallucination is better
        "category": "safety"
    },
    "toxicity": {
        "description": "Detects harmful, offensive, or inappropriate content",
        "required_fields": ["actual_output"],
        "optional_fields": ["input", "context"],
        "threshold_range": (0.0, 1.0),
        "higher_is_better": False,  # Lower toxicity is better
        "category": "safety"
    },
    "bias": {
        "description": "Detects demographic, cultural, or ideological biases",
        "required_fields": ["actual_output"],
        "optional_fields": ["input", "context"],
        "threshold_range": (0.0, 1.0),
        "higher_is_better": False,  # Lower bias is better
        "category": "safety"
    },
    "g_eval_coherence": {
        "description": "G-Eval metric for response coherence and logical flow",
        "required_fields": ["input", "actual_output"],
        "optional_fields": ["context", "expected_output"],
        "threshold_range": (0.0, 1.0),
        "higher_is_better": True,
        "category": "quality",
        "g_eval_config": {
            "name": "Coherence",
            "criteria": "Determine whether the actual output is coherent, well-structured, and flows logically from one point to the next.",
            "evaluation_steps": [
                "Read the input and actual output carefully",
                "Assess the logical flow and structure of the response",
                "Check for contradictions, unclear statements, or abrupt transitions",
                "Evaluate the overall coherence and readability",
                "Provide a score from 1-10 based on coherence quality"
            ],
            "evaluation_params": ["coherence", "logical_flow", "structure", "clarity"]
        }
    },
    "g_eval_correctness": {
        "description": "G-Eval metric for factual correctness and accuracy",
        "required_fields": ["input", "actual_output"],
        "optional_fields": ["context", "expected_output"],
        "threshold_range": (0.0, 1.0),
        "higher_is_better": True,
        "category": "quality",
        "g_eval_config": {
            "name": "Correctness",
            "criteria": "Determine whether the actual output is factually correct and accurate.",
            "evaluation_steps": [
                "Read the input, context (if provided), and actual output",
                "Verify factual claims against the provided context or general knowledge",
                "Identify any inaccuracies, errors, or misleading information",
                "Assess the overall correctness of the response",
                "Provide a score from 1-10 based on accuracy"
            ],
            "evaluation_params": ["accuracy", "factual_correctness", "truthfulness"]
        }
    },
    "g_eval_completeness": {
        "description": "G-Eval metric for response completeness and thoroughness",
        "required_fields": ["input", "actual_output"],
        "optional_fields": ["context", "expected_output"],
        "threshold_range": (0.0, 1.0),
        "higher_is_better": True,
        "category": "quality",
        "g_eval_config": {
            "name": "Completeness",
            "criteria": "Determine whether the actual output comprehensively addresses all aspects of the input question or request.",
            "evaluation_steps": [
                "Identify all aspects and sub-questions in the input",
                "Check if the actual output addresses each aspect adequately",
                "Assess whether important information is missing",
                "Evaluate the thoroughness and depth of the response",
                "Provide a score from 1-10 based on completeness"
            ],
            "evaluation_params": ["completeness", "thoroughness", "comprehensiveness"]
        }
    },
    "g_eval_helpfulness": {
        "description": "G-Eval metric for response helpfulness and usefulness",
        "required_fields": ["input", "actual_output"],
        "optional_fields": ["context", "expected_output"],
        "threshold_range": (0.0, 1.0),
        "higher_is_better": True,
        "category": "quality",
        "g_eval_config": {
            "name": "Helpfulness",
            "criteria": "Determine whether the actual output is helpful, useful, and actionable for the user.",
            "evaluation_steps": [
                "Understand the user's intent and needs from the input",
                "Assess whether the response provides useful information or guidance",
                "Check if the response is actionable and practical",
                "Evaluate the overall helpfulness and value to the user",
                "Provide a score from 1-10 based on helpfulness"
            ],
            "evaluation_params": ["helpfulness", "usefulness", "actionability", "value"]
        }
    }
}

# Metric categories for organizing in UI
DEEPEVAL_METRIC_CATEGORIES: Dict[str, List[str]] = {
    "relevance": ["answer_relevancy"],
    "groundedness": ["faithfulness"],
    "retrieval": ["contextual_precision", "contextual_recall", "contextual_relevancy"],
    "safety": ["hallucination", "toxicity", "bias"],
    "quality": ["g_eval_coherence", "g_eval_correctness", "g_eval_completeness", "g_eval_helpfulness"]
}

# Default metric configurations
DEFAULT_DEEPEVAL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "answer_relevancy": {
        "threshold": 0.7,
        "model": "gpt-4",
        "include_reason": True
    },
    "faithfulness": {
        "threshold": 0.8,
        "model": "gpt-4",
        "include_reason": True
    },
    "contextual_precision": {
        "threshold": 0.7,
        "model": "gpt-4",
        "include_reason": True
    },
    "contextual_recall": {
        "threshold": 0.7,
        "model": "gpt-4",
        "include_reason": True
    },
    "contextual_relevancy": {
        "threshold": 0.7,
        "model": "gpt-4",
        "include_reason": True
    },
    "hallucination": {
        "threshold": 0.3,  # Lower is better
        "model": "gpt-4"
    },
    "toxicity": {
        "threshold": 0.2,  # Lower is better
        "model": "gpt-4"
    },
    "bias": {
        "threshold": 0.2,  # Lower is better
        "model": "gpt-4"
    },
    "g_eval_coherence": {
        "threshold": 0.7,
        "model": "gpt-4"
    },
    "g_eval_correctness": {
        "threshold": 0.8,
        "model": "gpt-4"
    },
    "g_eval_completeness": {
        "threshold": 0.7,
        "model": "gpt-4"
    },
    "g_eval_helpfulness": {
        "threshold": 0.7,
        "model": "gpt-4"
    }
}


def get_supported_metrics_for_dataset_type(dataset_type: DatasetType) -> List[str]:
    """Get supported DeepEval metrics for a dataset type."""
    return DEEPEVAL_DATASET_TYPE_METRICS.get(dataset_type, [])


def get_metric_requirements(metric_name: str) -> Dict[str, Any]:
    """Get requirements for a specific metric."""
    return DEEPEVAL_METRIC_REQUIREMENTS.get(metric_name, {})


def get_metric_categories() -> Dict[str, List[str]]:
    """Get metric categories for UI organization."""
    return DEEPEVAL_METRIC_CATEGORIES


def get_default_config(metric_name: str) -> Dict[str, Any]:
    """Get default configuration for a metric."""
    return DEFAULT_DEEPEVAL_CONFIGS.get(metric_name, {})


def validate_metric_compatibility(metric_name: str, dataset_type: DatasetType) -> bool:
    """Check if a metric is compatible with a dataset type."""
    supported_metrics = get_supported_metrics_for_dataset_type(dataset_type)
    return metric_name in supported_metrics


def get_recommended_metrics(dataset_type: DatasetType) -> List[str]:
    """Get recommended metrics for a dataset type."""
    recommendations = {
        DatasetType.USER_QUERY: ["answer_relevancy", "toxicity", "g_eval_helpfulness"],
        DatasetType.QUESTION_ANSWER: ["answer_relevancy", "faithfulness", "g_eval_correctness"],
        DatasetType.CONTEXT: ["faithfulness", "contextual_precision", "contextual_recall"],
        DatasetType.CONVERSATION: ["answer_relevancy", "g_eval_coherence", "toxicity"],
        DatasetType.CUSTOM: ["answer_relevancy", "faithfulness", "g_eval_helpfulness"]
    }
    return recommendations.get(dataset_type, ["answer_relevancy"])


def create_metric_config_schema() -> Dict[str, Any]:
    """Create JSON schema for metric configuration."""
    return {
        "type": "object",
        "properties": {
            "metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": list(DEEPEVAL_METRIC_REQUIREMENTS.keys())
                        },
                        "threshold": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "model": {
                            "type": "string",
                            "default": "gpt-4"
                        },
                        "include_reason": {
                            "type": "boolean",
                            "default": True
                        }
                    },
                    "required": ["name"]
                }
            }
        },
        "required": ["metrics"]
    }
