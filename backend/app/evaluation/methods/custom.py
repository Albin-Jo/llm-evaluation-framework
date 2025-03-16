# File: app/evaluation/methods/custom.py
import logging
from typing import Any, Dict, List
from uuid import UUID

from app.evaluation.methods.base import BaseEvaluationMethod
from app.models.orm.models import Evaluation
from app.schema.evaluation_schema import EvaluationResultCreate, MetricScoreCreate

# Configure logging
logger = logging.getLogger(__name__)


class CustomEvaluationMethod(BaseEvaluationMethod):
    """Custom evaluation method allowing user-defined metrics and calculations."""

    async def run_evaluation(self, evaluation: Evaluation) -> List[EvaluationResultCreate]:
        """
        Run a custom evaluation based on user-defined logic.

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

        # Get custom metric functions from configuration
        custom_metrics = evaluation.config.get("custom_metrics", [])
        if not custom_metrics:
            logger.error(f"No custom metrics defined for evaluation {evaluation.id}")
            return []

        results = []

        for item_index, item in enumerate(dataset_items):
            try:
                # Process dataset item
                formatted_prompt = self._format_prompt(prompt.content, item)

                # Call the micro-agent API
                response = await self._call_microagent_api(
                    microagent.api_endpoint,
                    {
                        "prompt": formatted_prompt,
                        "query": item.get("query", ""),
                        "context": item.get("context", "")
                    }
                )

                # Extract LLM answer from response
                answer = response.get("answer", "")

                # Calculate metrics
                metrics = await self.calculate_metrics(
                    input_data=item,
                    output_data={"answer": answer},
                    config=evaluation.config or {}
                )

                # Calculate overall score based on weighted average
                total_weight = sum(metric.get("weight", 1.0) for metric in metrics.values())
                overall_score = sum(
                    metric["value"] * metric.get("weight", 1.0) for metric in metrics.values()
                ) / total_weight if total_weight > 0 else 0.0

                # Create metric scores
                metric_scores = [
                    MetricScoreCreate(
                        name=name,
                        value=metric["value"],
                        weight=metric.get("weight", 1.0),
                        metadata={"description": metric.get("description", "")}
                    )
                    for name, metric in metrics.items()
                ]

                # Create evaluation result
                result = EvaluationResultCreate(
                    evaluation_id=evaluation.id,
                    overall_score=overall_score,
                    raw_results={name: metric["value"] for name, metric in metrics.items()},
                    dataset_sample_id=str(item_index),
                    input_data={**item, "prompt": formatted_prompt},
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
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate custom metrics based on configuration.

        Args:
            input_data: Input data for the evaluation
            output_data: Output data from the LLM
            config: Evaluation configuration with custom metric definitions

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping metric names to value dictionaries
        """
        try:
            custom_metrics = config.get("custom_metrics", [])
            metrics = {}

            for metric_config in custom_metrics:
                metric_name = metric_config.get("name")
                metric_type = metric_config.get("type")

                if not metric_name or not metric_type:
                    continue

                # Calculate metric based on type
                if metric_type == "keyword_match":
                    keywords = metric_config.get("keywords", [])
                    answer = output_data.get("answer", "")
                    score = self._calculate_keyword_match(answer, keywords)

                    metrics[metric_name] = {
                        "value": score,
                        "weight": metric_config.get("weight", 1.0),
                        "description": metric_config.get("description", "Keyword match score")
                    }

                elif metric_type == "length_check":
                    min_length = metric_config.get("min_length", 0)
                    max_length = metric_config.get("max_length", 1000)
                    answer = output_data.get("answer", "")
                    score = self._calculate_length_check(answer, min_length, max_length)

                    metrics[metric_name] = {
                        "value": score,
                        "weight": metric_config.get("weight", 1.0),
                        "description": metric_config.get("description", "Length check score")
                    }

                elif metric_type == "json_validation":
                    schema = metric_config.get("schema", {})
                    answer = output_data.get("answer", "")
                    score = self._calculate_json_validation(answer, schema)

                    metrics[metric_name] = {
                        "value": score,
                        "weight": metric_config.get("weight", 1.0),
                        "description": metric_config.get("description", "JSON validation score")
                    }

                # Add more metric types as needed

            return metrics

        except Exception as e:
            logger.exception(f"Error calculating custom metrics: {e}")
            return {}

    def _calculate_keyword_match(self, text: str, keywords: List[str]) -> float:
        """
        Calculate keyword match score.

        Args:
            text: Text to check
            keywords: List of keywords to look for

        Returns:
            float: Match score (0-1)
        """
        if not text or not keywords:
            return 0.0

        text_lower = text.lower()
        matched_keywords = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return matched_keywords / len(keywords) if keywords else 0.0

    def _calculate_length_check(self, text: str, min_length: int, max_length: int) -> float:
        """
        Calculate length check score.

        Args:
            text: Text to check
            min_length: Minimum desired length
            max_length: Maximum desired length

        Returns:
            float: Length score (0-1)
        """
        if not text:
            return 0.0 if min_length > 0 else 1.0

        text_length = len(text.split())

        if text_length < min_length:
            return text_length / min_length
        elif text_length > max_length:
            return max(0.0, 1.0 - ((text_length - max_length) / max_length))
        else:
            return 1.0

    def _calculate_json_validation(self, text: str, schema: Dict[str, Any]) -> float:
        """
        Calculate JSON validation score.

        Args:
            text: Text to check (should be JSON)
            schema: JSON schema to validate against

        Returns:
            float: Validation score (0-1)
        """
        import json
        from jsonschema import validate, ValidationError

        if not text:
            return 0.0

        try:
            # Parse JSON
            data = json.loads(text)

            # Validate against schema
            validate(instance=data, schema=schema)
            return 1.0

        except json.JSONDecodeError:
            # Not valid JSON
            return 0.0

        except ValidationError:
            # Valid JSON but doesn't match schema
            return 0.5

        except Exception:
            return 0.0

    async def _call_microagent_api(self, api_endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the micro-agent API.

        Args:
            api_endpoint: API endpoint URL
            payload: Request payload

        Returns:
            Dict[str, Any]: API response
        """
        import httpx
        from app.core.config.settings import settings

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


        except Exception as e:
            logger.error(f"Error calling micro-agent API: {e}")
            raise Exception(f"Error calling micro-agent API: {str(e)}")

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