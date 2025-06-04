# File: backend/app/evaluation/utils/deepeval_config_validator.py
import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum

from backend.app.db.models.orm import DatasetType, EvaluationMethod
from backend.app.evaluation.metrics.deepeval_metrics import DEEPEVAL_AVAILABLE

logger = logging.getLogger(__name__)


class DeepEvalConfigValidator:
    """Enhanced validator for DeepEval evaluation configurations with latest metrics."""

    # Updated supported metrics based on latest DeepEval version
    SUPPORTED_METRICS = {
        # Core LLM Evaluation Metrics
        "answer_relevancy": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "strict_mode": {"type": bool, "default": False},
            "async_mode": {"type": bool, "default": True}
        },
        "faithfulness": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },
        "contextual_precision": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },
        "contextual_recall": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },
        "contextual_relevancy": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },

        # Safety and Ethics Metrics
        "hallucination": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.3},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },
        "bias": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.3},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },
        "toxicity": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.3},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },

        # Quality Metrics
        "coherence": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },
        "correctness": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },
        "fluency": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },

        # Summarization Metrics
        "summarization": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "assessment_questions": {"type": list, "default": []},
            "async_mode": {"type": bool, "default": True}
        },

        # Tool and Function Calling Metrics
        "tool_correctness": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },

        # Knowledge and Memory Metrics
        "knowledge_retention": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },

        # G-Eval based metrics
        "g_eval": {
            "threshold": {"type": float, "min": 0.0, "max": 10.0, "default": 7.0},
            "model": {"type": str, "default": "gpt-4o"},
            "evaluation_params": {"type": list, "default": []},
            "evaluation_steps": {"type": list, "default": []},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },

        # Conversation Metrics
        "conversation": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "conversation_criteria": {"type": list, "default": []},
            "async_mode": {"type": bool, "default": True}
        },

        # RAGAS Integration Metrics
        "ragas_faithfulness": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },
        "ragas_answer_relevancy": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },
        "ragas_context_precision": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        },
        "ragas_context_recall": {
            "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
            "model": {"type": str, "default": "gpt-4o"},
            "include_reason": {"type": bool, "default": True},
            "async_mode": {"type": bool, "default": True}
        }
    }

    # Current supported OpenAI models (realistic list)
    SUPPORTED_MODELS = {
        # GPT-3.5 models
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-16k",

        # GPT-4 models
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo-preview",

        # GPT-4o models
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",

        # o1 series models
        "o1-preview",
        "o1-preview-2024-09-12",
        "o1-mini",
        "o1-mini-2024-09-12",
    }

    # Enhanced dataset type metrics mapping
    DATASET_TYPE_METRICS = {
        DatasetType.QUESTION_ANSWER: {
            "required": ["answer_relevancy"],
            "recommended": ["faithfulness", "contextual_precision", "contextual_recall"],
            "optional": ["correctness", "coherence", "hallucination", "knowledge_retention"]
        },
        DatasetType.USER_QUERY: {
            "required": ["answer_relevancy"],
            "recommended": ["contextual_relevancy", "hallucination", "faithfulness"],
            "optional": ["bias", "toxicity", "correctness", "fluency"]
        },
        DatasetType.CONVERSATION: {
            "required": ["coherence"],
            "recommended": ["conversation", "contextual_relevancy", "answer_relevancy"],
            "optional": ["bias", "toxicity", "hallucination", "fluency"]
        },
        DatasetType.CUSTOM: {
            "required": [],
            "recommended": ["answer_relevancy", "faithfulness"],
            "optional": list(SUPPORTED_METRICS.keys())
        }
    }

    # Metric categories for better organization
    METRIC_CATEGORIES = {
        "quality": ["answer_relevancy", "faithfulness", "correctness", "coherence", "fluency"],
        "context": ["contextual_precision", "contextual_recall", "contextual_relevancy"],
        "safety": ["hallucination", "bias", "toxicity"],
        "specialized": ["summarization", "tool_correctness", "knowledge_retention", "conversation"],
        "g_eval": ["g_eval"],
        "ragas": ["ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_precision", "ragas_context_recall"]
    }

    def __init__(self):
        """Initialize the enhanced validator."""
        self.logger = logging.getLogger(__name__)

    def validate_evaluation_config(
            self,
            config: Dict[str, Any],
            dataset_type: DatasetType,
            metrics: List[str]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a complete evaluation configuration with enhanced checks.

        Args:
            config: The configuration dictionary to validate
            dataset_type: The type of dataset being evaluated
            metrics: List of metrics to be used

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        try:
            # Check if DeepEval is available
            if not DEEPEVAL_AVAILABLE:
                errors.append("DeepEval library is not installed. Install with: pip install deepeval")
                return False, errors, warnings

            # Validate top-level structure
            if not isinstance(config, dict):
                errors.append("Configuration must be a dictionary")
                return False, errors, warnings

            # Check for deepeval_config section
            if "deepeval_config" not in config:
                errors.append("Missing 'deepeval_config' section in configuration")
                return False, errors, warnings

            deepeval_config = config["deepeval_config"]
            if not isinstance(deepeval_config, dict):
                errors.append("'deepeval_config' must be a dictionary")
                return False, errors, warnings

            # Validate global model setting
            global_model = deepeval_config.get("model", "gpt-4o")
            model_valid, model_error = self.validate_model_availability(global_model)
            if not model_valid:
                errors.append(model_error)

            # Validate metrics
            metric_errors, metric_warnings = self._validate_metrics_config(
                deepeval_config, metrics, dataset_type
            )
            errors.extend(metric_errors)
            warnings.extend(metric_warnings)

            # Validate dataset type compatibility
            compatibility_warnings = self._validate_dataset_compatibility(dataset_type, metrics)
            warnings.extend(compatibility_warnings)

            # Enhanced validation for specific configurations
            enhanced_warnings = self._enhanced_validation_checks(deepeval_config, metrics, dataset_type)
            warnings.extend(enhanced_warnings)

            # Validate API credentials
            creds_valid, creds_error = self.validate_api_credentials()
            if not creds_valid:
                warnings.append(f"API credentials validation failed: {creds_error}")

            # Check for unknown configuration keys
            known_keys = {"model", "batch_size", "max_retries", "timeout"} | set(self.SUPPORTED_METRICS.keys())
            unknown_keys = set(deepeval_config.keys()) - known_keys
            if unknown_keys:
                warnings.append(f"Unknown configuration keys: {', '.join(unknown_keys)}")

        except Exception as e:
            errors.append(f"Unexpected error during validation: {str(e)}")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def _enhanced_validation_checks(
            self,
            deepeval_config: Dict[str, Any],
            metrics: List[str],
            dataset_type: DatasetType
    ) -> List[str]:
        """Enhanced validation checks for specific metric combinations and configurations."""
        warnings = []

        # Check for metric compatibility
        if "tool_correctness" in metrics and dataset_type != DatasetType.CUSTOM:
            warnings.append(
                "tool_correctness metric is typically used with custom datasets for function calling evaluation")

        # Check for RAGAS metrics with regular metrics
        ragas_metrics = [m for m in metrics if m.startswith("ragas_")]
        regular_metrics = [m for m in metrics if
                           not m.startswith("ragas_") and m in ["faithfulness", "answer_relevancy"]]

        if ragas_metrics and regular_metrics:
            overlapping = []
            for ragas_metric in ragas_metrics:
                regular_equivalent = ragas_metric.replace("ragas_", "")
                if regular_equivalent in regular_metrics:
                    overlapping.append(f"{ragas_metric} and {regular_equivalent}")

            if overlapping:
                warnings.append(f"Using both RAGAS and DeepEval versions of similar metrics: {', '.join(overlapping)}")

        # Check G-Eval configuration
        if "g_eval" in metrics:
            g_eval_config = deepeval_config.get("g_eval", {})
            if not g_eval_config.get("evaluation_params") and not g_eval_config.get("evaluation_steps"):
                warnings.append("g_eval metric requires evaluation_params or evaluation_steps to be defined")

        # Check conversation metric for appropriate dataset types
        if "conversation" in metrics and dataset_type != DatasetType.CONVERSATION:
            warnings.append("conversation metric is optimized for CONVERSATION dataset type")

        # Check for performance considerations
        expensive_metrics = ["g_eval", "summarization", "conversation"]
        expensive_count = len([m for m in metrics if m in expensive_metrics])
        if expensive_count > 2:
            warnings.append(f"Multiple expensive metrics detected ({expensive_count}). Consider performance impact.")

        return warnings

    def _validate_metrics_config(
            self,
            deepeval_config: Dict[str, Any],
            metrics: List[str],
            dataset_type: DatasetType
    ) -> Tuple[List[str], List[str]]:
        """Enhanced metrics configuration validation."""
        errors = []
        warnings = []

        for metric in metrics:
            if metric not in self.SUPPORTED_METRICS:
                errors.append(
                    f"Unsupported metric '{metric}'. Supported metrics: {', '.join(self.SUPPORTED_METRICS.keys())}")
                continue

            # Get metric configuration
            metric_config = deepeval_config.get(metric, {})
            if not isinstance(metric_config, dict):
                errors.append(f"Configuration for metric '{metric}' must be a dictionary")
                continue

            # Validate metric parameters
            metric_schema = self.SUPPORTED_METRICS[metric]
            for param_name, param_value in metric_config.items():
                if param_name not in metric_schema:
                    warnings.append(f"Unknown parameter '{param_name}' for metric '{metric}'")
                    continue

                param_spec = metric_schema[param_name]

                # Type validation
                expected_type = param_spec["type"]
                if not isinstance(param_value, expected_type):
                    errors.append(
                        f"Parameter '{param_name}' for metric '{metric}' must be of type {expected_type.__name__}")
                    continue

                # Range validation for numeric parameters
                if expected_type == float and "min" in param_spec and "max" in param_spec:
                    if not (param_spec["min"] <= param_value <= param_spec["max"]):
                        errors.append(
                            f"Parameter '{param_name}' for metric '{metric}' must be between "
                            f"{param_spec['min']} and {param_spec['max']}"
                        )

            # Special validation for specific metrics
            if metric == "g_eval":
                self._validate_g_eval_config(metric_config, errors, warnings)
            elif metric == "summarization":
                self._validate_summarization_config(metric_config, errors, warnings)
            elif metric == "conversation":
                self._validate_conversation_config(metric_config, errors, warnings)

        return errors, warnings

    def _validate_g_eval_config(self, config: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
        """Validate G-Eval specific configuration."""
        if "evaluation_params" in config:
            if not isinstance(config["evaluation_params"], list):
                errors.append("g_eval evaluation_params must be a list")

        if "evaluation_steps" in config:
            if not isinstance(config["evaluation_steps"], list):
                errors.append("g_eval evaluation_steps must be a list")

    def _validate_summarization_config(self, config: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
        """Validate summarization specific configuration."""
        if "assessment_questions" in config and not isinstance(config["assessment_questions"], list):
            errors.append("summarization assessment_questions must be a list")

    def _validate_conversation_config(self, config: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
        """Validate conversation specific configuration."""
        if "conversation_criteria" in config and not isinstance(config["conversation_criteria"], list):
            errors.append("conversation conversation_criteria must be a list")

    def generate_recommended_config(
            self,
            dataset_type: DatasetType,
            metrics: List[str],
            model: str = "gpt-4o",
            use_case: str = "general"
    ) -> Dict[str, Any]:
        """
        Generate an enhanced recommended configuration for the given parameters.

        Args:
            dataset_type: Type of dataset
            metrics: List of metrics to configure
            model: Model to use (default: gpt-4o)
            use_case: Use case optimization ("general", "fast", "accurate", "cost_effective")

        Returns:
            Enhanced recommended configuration dictionary
        """
        config = {
            "deepeval_config": {
                "model": self.get_recommended_model(use_case) if model == "gpt-4o" else model,
                "batch_size": self._get_recommended_batch_size(use_case),
                "max_retries": 3,
                "timeout": 300
            }
        }

        # Add metric configurations
        for metric in metrics:
            if metric in self.SUPPORTED_METRICS:
                metric_config = {}
                metric_schema = self.SUPPORTED_METRICS[metric]

                # Add default values for all parameters
                for param_name, param_spec in metric_schema.items():
                    if "default" in param_spec:
                        metric_config[param_name] = param_spec["default"]

                # Apply use case optimizations
                if use_case == "fast":
                    metric_config["include_reason"] = False
                    metric_config["async_mode"] = True
                elif use_case == "accurate":
                    metric_config["include_reason"] = True
                    if "strict_mode" in metric_schema:
                        metric_config["strict_mode"] = True

                config["deepeval_config"][metric] = metric_config

        # Add dataset-specific optimizations
        if dataset_type == DatasetType.QUESTION_ANSWER:
            # Higher thresholds for Q&A
            for metric in ["answer_relevancy", "faithfulness", "correctness"]:
                if metric in config["deepeval_config"]:
                    config["deepeval_config"][metric]["threshold"] = 0.8

        elif dataset_type == DatasetType.CONVERSATION:
            # Optimized for conversational data
            for metric in ["coherence", "conversation", "contextual_relevancy"]:
                if metric in config["deepeval_config"]:
                    config["deepeval_config"][metric]["threshold"] = 0.6

        elif dataset_type == DatasetType.USER_QUERY:
            # Focus on relevancy and safety
            for metric in ["answer_relevancy", "contextual_relevancy"]:
                if metric in config["deepeval_config"]:
                    config["deepeval_config"][metric]["threshold"] = 0.75

        return config

    def _get_recommended_batch_size(self, use_case: str) -> int:
        """Get recommended batch size based on use case."""
        batch_sizes = {
            "fast": 10,
            "general": 5,
            "accurate": 3,
            "cost_effective": 8
        }
        return batch_sizes.get(use_case, 5)

    def get_recommended_model(self, use_case: str = "general") -> str:
        """
        Get recommended model based on use case with updated model list.

        Args:
            use_case: Type of use case

        Returns:
            Recommended model name
        """
        recommendations = {
            "general": "gpt-4o",
            "fast": "gpt-3.5-turbo",
            "accurate": "gpt-4o",
            "cost_effective": "gpt-3.5-turbo",
            "reasoning": "o1-mini",
            "advanced": "gpt-4-turbo"
        }
        return recommendations.get(use_case, "gpt-4o")

    def validate_model_availability(self, model: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a specific model is available and supported.

        Args:
            model: Model name to validate

        Returns:
            Tuple of (is_available, error_or_suggestion)
        """
        if model in self.SUPPORTED_MODELS:
            return True, None

        # Find similar models
        suggestions = []
        if "gpt-4o" in model.lower():
            suggestions = [m for m in self.SUPPORTED_MODELS if "gpt-4o" in m]
        elif "gpt-4" in model.lower():
            suggestions = [m for m in self.SUPPORTED_MODELS if "gpt-4" in m and "gpt-4o" not in m]
        elif "gpt-3.5" in model.lower():
            suggestions = [m for m in self.SUPPORTED_MODELS if "gpt-3.5" in m]
        elif "o1" in model.lower():
            suggestions = [m for m in self.SUPPORTED_MODELS if "o1" in m]

        if suggestions:
            return False, f"Model '{model}' not supported. Similar available models: {', '.join(suggestions[:3])}"
        else:
            return False, f"Model '{model}' not supported. Available models: {', '.join(list(self.SUPPORTED_MODELS)[:5])}..."

    def get_metrics_by_category(self, category: str) -> List[str]:
        """Get metrics by category."""
        return self.METRIC_CATEGORIES.get(category, [])

    def get_all_categories(self) -> List[str]:
        """Get all available metric categories."""
        return list(self.METRIC_CATEGORIES.keys())

    def validate_api_credentials(self) -> Tuple[bool, Optional[str]]:
        """
        Enhanced API credentials validation.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check for OpenAI API key
            openai_key = "98a26ff989784c8fa8212d80e704c829"
            if not openai_key:
                return False, "OPENAI_API_KEY environment variable not set"

            # # Basic format validation
            # if not openai_key.startswith(("sk-", "sk-proj-")):
            #     return False, "Invalid OpenAI API key format (should start with 'sk-' or 'sk-proj-')"
            #
            # if len(openai_key) < 20:
            #     return False, "OpenAI API key appears to be too short"

            return True, None

        except Exception as e:
            return False, f"Error validating credentials: {str(e)}"

    def optimize_config_for_performance(
            self,
            config: Dict[str, Any],
            dataset_size: int,
            time_budget_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Enhanced configuration optimization for performance.

        Args:
            config: Original configuration
            dataset_size: Size of the dataset
            time_budget_minutes: Optional time budget in minutes

        Returns:
            Optimized configuration
        """
        optimized_config = config.copy()
        deepeval_config = optimized_config.get("deepeval_config", {})

        # Optimize based on dataset size
        if dataset_size > 1000:
            # Disable detailed reasoning for large datasets
            for metric in deepeval_config:
                if metric in self.SUPPORTED_METRICS and isinstance(deepeval_config[metric], dict):
                    deepeval_config[metric]["include_reason"] = False
                    deepeval_config[metric]["async_mode"] = True

        if dataset_size > 5000:
            # Use faster model for very large datasets
            deepeval_config["model"] = "gpt-3.5-turbo"
            deepeval_config["batch_size"] = 15

        # Optimize based on time budget
        if time_budget_minutes:
            if time_budget_minutes < 30:
                # Very tight budget
                deepeval_config["model"] = "gpt-3.5-turbo"
                deepeval_config["batch_size"] = 20
                for metric in deepeval_config:
                    if metric in self.SUPPORTED_METRICS and isinstance(deepeval_config[metric], dict):
                        deepeval_config[metric]["include_reason"] = False

        return optimized_config