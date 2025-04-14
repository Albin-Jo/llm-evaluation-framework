# File: app/evaluation/methods/manual.py
import logging
from typing import Any, Dict, List

from backend.app.evaluation.methods.base import BaseEvaluationMethod
from backend.app.db.models.orm import Evaluation
from backend.app.db.schema.evaluation_schema import EvaluationResultCreate

# Configure logging
logger = logging.getLogger(__name__)


class ManualEvaluationMethod(BaseEvaluationMethod):
    """Manual evaluation method for human review of LLM outputs."""

    async def run_evaluation(self, evaluation: Evaluation) -> List[EvaluationResultCreate]:
        """
        Prepare for manual evaluation by generating LLM outputs for each dataset item.

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

                # Create evaluation result with placeholder metrics
                # These will be updated by human reviewers later
                result = EvaluationResultCreate(
                    evaluation_id=evaluation.id,
                    overall_score=None,  # To be filled by reviewer
                    raw_results={},      # To be filled by reviewer
                    dataset_sample_id=str(item_index),
                    input_data={**item, "prompt": formatted_prompt},
                    output_data={"answer": answer},
                    processing_time_ms=response.get("processing_time_ms"),
                    metric_scores=[]     # To be filled by reviewer
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
    ) -> Dict[str, float]:
        """
        Placeholder for manual metric calculation.
        In practice, this would be filled by human reviewers.

        Args:
            input_data: Input data for the evaluation
            output_data: Output data from the LLM
            config: Evaluation configuration

        Returns:
            Dict[str, float]: Dictionary mapping metric names to values
        """
        # This method doesn't do automatic calculation for manual evaluations
        # It will be populated by human reviewers through the UI
        return {}

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
        from backend.app.core.config import settings

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