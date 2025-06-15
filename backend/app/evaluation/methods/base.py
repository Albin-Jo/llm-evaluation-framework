import asyncio
import csv
import io
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import Dataset, Evaluation, Agent, Prompt, EvaluationMethod
from backend.app.db.schema.evaluation_schema import EvaluationResultCreate, MetricScoreCreate
from backend.app.evaluation.utils.dataset_utils import (
    process_user_query_dataset, process_context_dataset,
    process_qa_dataset, process_conversation_dataset,
    process_custom_dataset
)
from backend.app.evaluation.utils.progress import update_evaluation_progress
from backend.app.services.agent_clients.base import AgentClient
from backend.app.services.storage import get_storage_service

# Add logger configuration
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)


def _is_lower_better_metric(metric_name: str, evaluation_method: EvaluationMethod) -> bool:
    """Check if a metric is 'lower is better' based on evaluation method."""
    if evaluation_method == EvaluationMethod.DEEPEVAL:
        from backend.app.evaluation.metrics.deepeval_metrics import get_metric_requirements
        requirements = get_metric_requirements(metric_name)
        return not requirements.get("higher_is_better", True)
    elif evaluation_method == EvaluationMethod.RAGAS:
        # Define RAGAS metrics that are lower-is-better
        lower_better_ragas = {"noise_sensitivity"}
        return metric_name in lower_better_ragas
    return False


def _get_metric_threshold(metric_name: str, evaluation_method: EvaluationMethod,
                          config: Dict[str, Any]) -> float:
    """Get threshold for a metric based on evaluation method and config."""
    # Check if threshold is specified in config
    if config and config.get("thresholds") and metric_name in config["thresholds"]:
        return config["thresholds"][metric_name]

    # Use evaluation-wide threshold if available
    if config and config.get("threshold"):
        return config["threshold"]

    # Default thresholds by evaluation method
    if evaluation_method == EvaluationMethod.DEEPEVAL:
        from backend.app.evaluation.metrics.deepeval_metrics import get_default_config
        default_config = get_default_config(metric_name)
        return default_config.get("threshold", 0.7)
    elif evaluation_method == EvaluationMethod.RAGAS:
        # RAGAS default thresholds
        ragas_thresholds = {
            "faithfulness": 0.7,
            "response_relevancy": 0.7,
            "context_precision": 0.7,
            "context_recall": 0.7,
            "context_entity_recall": 0.7,
            "noise_sensitivity": 0.3,  # Lower is better
            "answer_correctness": 0.7,
            "answer_similarity": 0.7,
            "answer_relevancy": 0.7,
            "factual_correctness": 0.7
        }
        return ragas_thresholds.get(metric_name, 0.7)

    return 0.7  # Default threshold


def _generate_ragas_reason(metric_name: str, value: float, threshold: float,
                           success: bool, is_lower_better: bool) -> str:
    """Generate explanation reason for RAGAS metrics."""
    if success:
        if is_lower_better:
            return f"The score is {value:.2f} which is below the threshold of {threshold:.2f}, indicating good performance."
        else:
            return f"The score is {value:.2f} which meets or exceeds the threshold of {threshold:.2f}, indicating good performance."
    else:
        if is_lower_better:
            return f"The score is {value:.2f} which exceeds the threshold of {threshold:.2f}, indicating room for improvement."
        else:
            return f"The score is {value:.2f} which is below the threshold of {threshold:.2f}, indicating room for improvement."


class BaseEvaluationMethod(ABC):
    """Base class for all evaluation methods."""

    # Add method name class variable
    method_name = "base"  # Override in subclasses

    def __init__(self, db_session: AsyncSession):
        """
        Initialize the evaluation method.

        Args:
            db_session: Database session
        """
        self.db_session = db_session

    async def get_agent(self, agent_id: UUID) -> Optional[Agent]:
        """
        Get MicroAgent by ID.

        Args:
            agent_id: MicroAgent ID

        Returns:
            Optional[MicroAgent]: MicroAgent if found, None otherwise
        """
        from backend.app.db.repositories.base import BaseRepository
        repo = BaseRepository(Agent, self.db_session)
        return await repo.get(agent_id)

    async def get_dataset(self, dataset_id: UUID) -> Optional[Dataset]:
        """
        Get Dataset by ID.

        Args:
            dataset_id: Dataset ID

        Returns:
            Optional[Dataset]: Dataset if found, None otherwise
        """
        from backend.app.db.repositories.base import BaseRepository
        repo = BaseRepository(Dataset, self.db_session)
        return await repo.get(dataset_id)

    async def get_prompt(self, prompt_id: UUID) -> Optional[Prompt]:
        """
        Get Prompt by ID.

        Args:
            prompt_id: Prompt ID

        Returns:
            Optional[Prompt]: Prompt if found, None otherwise
        """
        from backend.app.db.repositories.base import BaseRepository
        repo = BaseRepository(Prompt, self.db_session)
        return await repo.get(prompt_id)

    async def load_dataset(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """
        Load dataset from storage with support for different dataset types.

        Args:
            dataset: Dataset model

        Returns:
            List[Dict[str, Any]]: Dataset items
        """
        # Get storage service
        storage_service = get_storage_service()

        try:
            # Log dataset info
            logger.info(
                f"Loading dataset {dataset.id} (name: {dataset.name}) from {dataset.file_path} (type: {dataset.type})")

            # Load dataset file
            data = await storage_service.read_file(dataset.file_path)

            # Log file size
            logger.info(f"Dataset file size: {len(data) if data else 0} bytes")

            if not data:
                logger.error(f"Empty dataset file: {dataset.file_path}")
                # Return a dummy dataset item for testing
                return [{
                    "query": "Why is the dataset empty?",
                    "context": "This is a placeholder context because the dataset file was empty.",
                    "ground_truth": "The dataset file was empty or could not be read."
                }]

            # Determine file type based on the file path extension
            file_path = dataset.file_path.lower()
            is_json = file_path.endswith('.json')
            is_csv = file_path.endswith('.csv')

            # Log the file format for debugging
            logger.info(f"File format detected: {'JSON' if is_json else 'CSV' if is_csv else 'Unknown'}")

            # Process based on file format and dataset type
            if is_json:
                try:
                    # Parse JSON data
                    parsed_data = json.loads(data)

                    # Process based on dataset type
                    if dataset.type == "user_query":
                        logger.info("Processing USER_QUERY dataset")
                        return process_user_query_dataset(parsed_data)

                    elif dataset.type == "context":
                        logger.info("Processing CONTEXT dataset")
                        return process_context_dataset(parsed_data)

                    elif dataset.type == "question_answer":
                        logger.info("Processing QUESTION_ANSWER dataset")
                        return process_qa_dataset(parsed_data)

                    elif dataset.type == "conversation":
                        logger.info("Processing CONVERSATION dataset")
                        return process_conversation_dataset(parsed_data)

                    elif dataset.type == "custom":
                        logger.info("Processing CUSTOM dataset")
                        return process_custom_dataset(parsed_data)

                    else:
                        logger.warning(f"Unknown dataset type: {dataset.type}, treating as generic JSON")
                        # Default JSON processing
                        if isinstance(parsed_data, list):
                            return parsed_data
                        else:
                            return [parsed_data]

                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON dataset: {e}")
                    # Log the beginning of the file for debugging
                    logger.error(f"File content (first 200 chars): {data[:200]}")
                    raise ValueError(f"Invalid JSON in dataset file: {e}")

            elif is_csv:
                try:
                    csv_file = io.StringIO(data)
                    reader = csv.DictReader(csv_file)
                    csv_data = [dict(row) for row in reader]

                    # Process based on dataset type
                    if dataset.type == "user_query":
                        logger.info("Processing USER_QUERY dataset from CSV")
                        return process_user_query_dataset(csv_data)

                    elif dataset.type == "context":
                        logger.info("Processing CONTEXT dataset from CSV")
                        return process_context_dataset(csv_data)

                    elif dataset.type == "question_answer":
                        logger.info("Processing QUESTION_ANSWER dataset from CSV")
                        return process_qa_dataset(csv_data)

                    elif dataset.type == "conversation":
                        logger.info("Processing CONVERSATION dataset from CSV")
                        return process_conversation_dataset(csv_data)

                    elif dataset.type == "custom":
                        logger.info("Processing CUSTOM dataset from CSV")
                        return process_custom_dataset(csv_data)

                    else:
                        logger.warning(f"Unknown dataset type: {dataset.type}, treating as generic CSV")
                        return csv_data

                except csv.Error as e:
                    logger.error(f"Error parsing CSV dataset: {e}")
                    raise ValueError(f"Invalid CSV in dataset file: {e}")
            else:
                # Handle raw text or other formats
                logger.warning(f"Unrecognized file format for dataset: {dataset.file_path}")
                return [{
                    "query": "What is in this file?",
                    "context": data[:2000] + "..." if len(data) > 2000 else data
                }]

        except FileNotFoundError:
            logger.error(f"Dataset file not found: {dataset.file_path}")
            raise ValueError(f"Dataset file not found: {dataset.file_path}")

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise ValueError(f"Error loading dataset: {str(e)}")

    async def process_batch(self, evaluation: Evaluation, batch: List[Dict[str, Any]]) -> List[EvaluationResultCreate]:
        """
        Process a batch of dataset items.

        Args:
            evaluation: Evaluation model
            batch: List of dataset items

        Returns:
            List[EvaluationResultCreate]: List of evaluation results
        """
        tasks = []
        for item_index, item in enumerate(batch):
            task = asyncio.create_task(self.process_item(evaluation, item, item_index))
            tasks.append(task)

        # Wait for all tasks to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        results = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing item {i}: {result}")
                # Create error result
                error_result = EvaluationResultCreate(
                    evaluation_id=evaluation.id,
                    overall_score=0.0,
                    raw_results={"error": str(result)},
                    dataset_sample_id=str(i),
                    input_data=batch[i],
                    output_data={"error": str(result)},
                    metric_scores=[]
                )
                results.append(error_result)
            else:
                results.append(result)

        return results

    async def process_item(self, evaluation: Evaluation, item: Dict[str, Any],
                           item_index: int) -> EvaluationResultCreate:
        """
        Process a single dataset item.

        Args:
            evaluation: Evaluation model
            item: Dataset item
            item_index: Index of the item in the dataset

        Returns:
            EvaluationResultCreate: Evaluation result

        Raises:
            Exception: If processing fails
        """
        raise NotImplementedError("Subclasses must implement process_item method")

    @staticmethod
    async def log_progress(evaluation_id: UUID, total: int, processed: int) -> None:
        """
        Log evaluation progress.

        Args:
            evaluation_id: Evaluation ID
            total: Total number of items
            processed: Number of processed items
        """
        if total > 0:
            progress = (processed / total) * 100
            logger.info(f"Evaluation {evaluation_id}: {processed}/{total} items processed ({progress:.2f}%)")

    @abstractmethod
    async def run_evaluation(self, evaluation: Evaluation, jwt_token: Optional[str] = None) -> List[
        EvaluationResultCreate]:
        """
        Run the evaluation.

        Args:
            evaluation: Evaluation model

        Returns:
            List[EvaluationResultCreate]: List of evaluation results
        """
        pass

    @abstractmethod
    async def calculate_metrics(
            self, input_data: Dict[str, Any], output_data: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate metrics for a single evaluation item.

        Args:
            input_data: Input data for the evaluation
            output_data: Output data from the LLM
            config: Evaluation configuration

        Returns:
            Dict[str, float]: Dictionary mapping metric names to values
        """
        pass

    def _format_prompt(self, prompt_template: str, item: Dict[str, Any]) -> str:
        """
        Format a prompt template with values from the dataset item.

        Args:
            prompt_template: Prompt template string with {placeholders}
            item: Dataset item with values to substitute

        Returns:
            str: Formatted prompt
        """
        try:
            # Start with the original template
            formatted_prompt = prompt_template

            # Log basic info for debugging
            logger.info(f"Formatting prompt template (length: {len(prompt_template)})")

            # Check if the item is empty or None
            if not item:
                logger.warning("Empty dataset item provided for prompt formatting")
                return prompt_template

            # Identify all placeholders in the template
            import re
            placeholders = re.findall(r'\{([^{}]*)}', prompt_template)
            logger.info(f"Found placeholders: {placeholders}")

            # Replace placeholders in the template with item values
            for key, value in item.items():
                placeholder = f"{{{key}}}"
                if placeholder in formatted_prompt:
                    # Convert value to string if it's not already
                    str_value = str(value) if value is not None else ""
                    formatted_prompt = formatted_prompt.replace(placeholder, str_value)
                    logger.info(f"Replaced placeholder {placeholder} with value (length: {len(str_value)})")

            # Check for any remaining placeholders and log.json them
            remaining = re.findall(r'\{([^{}]*)}', formatted_prompt)
            if remaining:
                logger.warning(f"Unreplaced placeholders in prompt template: {remaining}")

            return formatted_prompt

        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            # Return original template as fallback
            return prompt_template

    async def batch_process(
            self,
            evaluation: Evaluation,
            batch_size: int = 10,
            jwt_token: Optional[str] = None
    ) -> List[EvaluationResultCreate]:
        """
        Process dataset items in batches with improved performance.

        Args:
            evaluation: Evaluation model
            batch_size: Number of items to process in each batch
            jwt_token: Optional JWT token to use for authentication with MCP agents

        Returns:
            List[EvaluationResultCreate]: List of evaluation results
        """
        # Get related entities - fetch all at once
        from backend.app.db.repositories.agent_repository import AgentRepository
        from backend.app.db.models.orm import IntegrationType

        agent_repo = AgentRepository(self.db_session)

        # Get agent with decrypted credentials
        agent = await agent_repo.get_with_decrypted_credentials(evaluation.agent_id)
        dataset = await self.get_dataset(evaluation.dataset_id)
        prompt = await self.get_prompt(evaluation.prompt_id)

        if not agent or not dataset or not prompt:
            logger.error(f"Missing required entities for evaluation {evaluation.id}")
            return []

        # Load dataset
        dataset_items = await self.load_dataset(dataset)
        all_results = []

        # Create agent client factory
        from backend.app.services.agent_clients.factory import AgentClientFactory

        try:
            # Create agent client with JWT token if agent is MCP type
            logger.info(f"Creating client for agent type: {agent.integration_type}")
            if agent.integration_type == IntegrationType.MCP and jwt_token:
                logger.info(f"Using JWT token for MCP agent in evaluation {evaluation.id}")
                agent_client = await AgentClientFactory.create_client(agent, jwt_token)
            else:
                agent_client = await AgentClientFactory.create_client(agent)

            # Process in batches
            for batch_start in range(0, len(dataset_items), batch_size):
                batch_end = min(batch_start + batch_size, len(dataset_items))
                batch_items = dataset_items[batch_start:batch_end]

                # Create tasks for concurrent processing
                tasks = []
                for batch_idx, item in enumerate(batch_items):
                    item_index = batch_start + batch_idx

                    # Format prompt once per item
                    formatted_prompt = self._format_prompt(prompt.content, item)

                    # Create task for processing
                    task = asyncio.create_task(
                        self._process_batch_item_with_client(
                            client=agent_client,
                            evaluation=evaluation,
                            item=item,
                            item_index=item_index,
                            formatted_prompt=formatted_prompt
                        )
                    )
                    tasks.append(task)

                # Wait for all batch tasks to complete
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing item {batch_start + i}: {result}")
                        # Create error result
                        error_result = EvaluationResultCreate(
                            evaluation_id=evaluation.id,
                            overall_score=0.0,
                            raw_results={"error": str(result)},
                            dataset_sample_id=str(batch_start + i),
                            input_data=batch_items[i],
                            output_data={"error": str(result)},
                            metric_scores=[]
                        )
                        all_results.append(error_result)
                    else:
                        all_results.append(result)
                await update_evaluation_progress(evaluation.id, batch_end, len(dataset_items))
                # Log progress
                await self.log_progress(
                    evaluation.id,
                    len(dataset_items),
                    batch_end
                )

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # Add error result
            error_result = EvaluationResultCreate(
                evaluation_id=evaluation.id,
                overall_score=0.0,
                raw_results={"error": str(e)},
                dataset_sample_id="batch_error",
                input_data={},
                output_data={"error": str(e)},
                metric_scores=[]
            )
            all_results.append(error_result)

        return all_results

    async def _process_batch_item(
            self,
            client: httpx.AsyncClient,
            evaluation: Evaluation,
            agent: Agent,
            item: Dict[str, Any],
            item_index: int,
            formatted_prompt: str
    ) -> EvaluationResultCreate:
        """
        Process a single dataset item within a batch with improved error handling.

        Args:
            client: HTTP client
            evaluation: Evaluation model
            agent: MicroAgent model
            item: Dataset item
            item_index: Index of the item
            formatted_prompt: Formatted prompt

        Returns:
            EvaluationResultCreate: Evaluation result
        """
        from backend.app.evaluation.utils.dataset_utils import truncate_text, process_content_filter_error
        from backend.app.core.config import settings
        import asyncio
        import random
        import json

        try:
            # Extract data
            query = item.get("query", "")
            context = item.get("context", "")
            ground_truth = item.get("ground_truth", "")

            # Start timing
            start_time = time.time()

            # Define truncation limits
            MAX_SYSTEM_LENGTH = 1500
            MAX_USER_LENGTH = 4500

            # Truncate content if needed
            system_message = truncate_text(formatted_prompt, MAX_SYSTEM_LENGTH)
            truncated_query = truncate_text(query, 500)
            truncated_context = truncate_text(context, MAX_USER_LENGTH - len(truncated_query) - 50)

            # Log truncation if it occurred
            if len(formatted_prompt) > MAX_SYSTEM_LENGTH:
                logger.warning(
                    f"System prompt truncated from {len(formatted_prompt)} to {MAX_SYSTEM_LENGTH} characters")
            if len(query) > 500:
                logger.warning(f"Query truncated from {len(query)} to 500 characters")
            if len(context) > (MAX_USER_LENGTH - len(truncated_query) - 50):
                logger.warning(f"Context truncated from {len(context)} to {len(truncated_context)} characters")

            # Prepare user message
            user_message = f"Question: {truncated_query}"
            if truncated_context:
                user_message += f"\n\nContext: {truncated_context}"

            # Add evaluation prefix to help with content filters
            system_message = "This is an evaluation for testing purposes. " + system_message
            user_message = "For evaluation purposes: " + user_message

            payload = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 1000,
                "temperature": 0.0
            }

            logger.debug(f"Payload message lengths - system: {len(system_message)}, user: {len(user_message)}")
            logger.info(f"Calling Azure OpenAI API for item {item_index}")

            # Make the API call with retry for content length errors
            max_retries = 3
            retry_count = 0
            backoff_base = 1.5
            current_system_max = MAX_SYSTEM_LENGTH
            current_user_max = MAX_USER_LENGTH

            while retry_count < max_retries:
                try:
                    response = await client.post(
                        agent.api_endpoint,
                        json=payload,
                        headers={
                            "api-key": settings.AZURE_OPENAI_KEY,
                            "Content-Type": "application/json"
                        },
                        timeout=60.0
                    )

                    # Check for API errors
                    if response.status_code >= 400:
                        error_text = await response.text()
                        logger.warning(f"API call failed with status {response.status_code}: {error_text}")

                        try:
                            error_json = json.loads(error_text)

                            # Handle content filter errors
                            if error_json.get("code") == "content_filter" or "content_filter" in error_text:
                                logger.warning("Content filter error detected")

                                # Process error for detailed reporting
                                error_details = process_content_filter_error(error_json, item_index)

                                # Return an evaluation result with error details
                                return EvaluationResultCreate(
                                    evaluation_id=evaluation.id,
                                    overall_score=0.0,
                                    raw_results={"error": error_details},
                                    dataset_sample_id=str(item_index),
                                    input_data={
                                        "query": query,
                                        "context": context,
                                        "ground_truth": ground_truth,
                                        "prompt": formatted_prompt
                                    },
                                    output_data={
                                        "error": "Content filter error",
                                        "error_details": error_details,
                                        "success": False
                                    },
                                    metric_scores=[]
                                )

                            # For token limit errors, retry with shorter content
                            elif error_json.get("code") == "context_length_exceeded" or "token" in error_text.lower():
                                # Reduce content length for next attempt
                                current_system_max = int(current_system_max * 0.7)
                                current_user_max = int(current_user_max * 0.7)

                                # Create shorter payload
                                system_message = truncate_text(formatted_prompt, current_system_max)
                                truncated_query = truncate_text(query, min(500, current_user_max // 10))
                                truncated_context = truncate_text(
                                    context,
                                    current_user_max - len(truncated_query) - 50
                                )

                                user_message = f"Question: {truncated_query}"
                                if truncated_context:
                                    user_message += f"\n\nContext: {truncated_context}"

                                system_message = "For evaluation: " + system_message
                                user_message = "Evaluation query: " + user_message

                                payload = {
                                    "messages": [
                                        {"role": "system", "content": system_message},
                                        {"role": "user", "content": user_message}
                                    ],
                                    "max_tokens": 1000,
                                    "temperature": 0.0
                                }

                                logger.warning(
                                    f"Reduced content length for retry - system:{len(system_message)}, user:{len(user_message)}")
                        except json.JSONDecodeError:
                            # Not a JSON error response, handle generically
                            pass

                        # If we couldn't handle the error specifically, raise for status
                        response.raise_for_status()

                    # If we get here, the API call was successful
                    response_data = response.json()
                    processing_time = int((time.time() - start_time) * 1000)

                    # Extract answer from response
                    try:
                        # Log response structure for debugging
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Response data structure: {json.dumps(response_data)[:500]}...")

                        # Extract from standard Azure OpenAI format
                        answer = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

                        if not answer and "choices" in response_data:
                            # Try alternative extraction methods if needed
                            choices = response_data["choices"]
                            if choices and isinstance(choices[0], dict):
                                # Try different fields that might contain the answer
                                for field in ["text", "content", "answer"]:
                                    if field in choices[0]:
                                        answer = choices[0][field]
                                        break

                        logger.debug(f"Extracted answer: {answer[:100]}..." if len(
                            answer) > 100 else f"Extracted answer: {answer}")

                        # Break out of the retry loop on success
                        break

                    except (KeyError, IndexError) as e:
                        logger.error(f"Error extracting answer from response: {e}")
                        answer = f"Error extracting answer: {str(e)}"
                        # Continue with retry since we couldn't properly parse the response

                except Exception as e:
                    retry_count += 1
                    logger.error(f"Error on API call attempt {retry_count}: {e}")

                    if retry_count < max_retries:
                        wait_time = (backoff_base ** retry_count) + (random.random() * 0.5)
                        logger.info(f"Retrying in {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        # Max retries reached
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise

            # Calculate metrics
            metrics = await self.calculate_metrics(
                input_data={
                    "query": query,
                    "context": context,
                    "ground_truth": ground_truth
                },
                output_data={"answer": answer},
                config=evaluation.config or {}
            )

            # Calculate overall score
            overall_score = sum(metrics.values()) / len(metrics) if metrics else 0.0

            # Create metric scores
            metric_scores = [
                MetricScoreCreate(
                    name=name,
                    value=value,
                    weight=1.0,
                    metadata={"description": self._get_metric_description(name)}
                )
                for name, value in metrics.items()
            ]

            # Create result
            return EvaluationResultCreate(
                evaluation_id=evaluation.id,
                overall_score=overall_score,
                raw_results=metrics,
                dataset_sample_id=str(item_index),
                input_data={
                    "query": query,
                    "context": context,
                    "ground_truth": ground_truth,
                    "prompt": formatted_prompt
                },
                output_data={"answer": answer},
                processing_time_ms=processing_time,
                metric_scores=metric_scores
            )

        except Exception as e:
            logger.exception(f"Error processing dataset item {item_index}: {e}")
            # Create error result
            return EvaluationResultCreate(
                evaluation_id=evaluation.id,
                overall_score=0.0,
                raw_results={"error": str(e)},
                dataset_sample_id=str(item_index),
                input_data=item,
                output_data={"error": str(e)},
                metric_scores=[]
            )

    @staticmethod
    async def _call_agent_api_with_retry(
            api_endpoint: str, payload: Dict[str, Any], max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Call the micro-agent API with retry logic.

        Args:
            api_endpoint: API endpoint URL
            payload: Request payload
            max_retries: Maximum number of retries

        Returns:
            Dict[str, Any]: API response
        """
        import httpx
        from backend.app.core.config import settings
        from asyncio import sleep

        retries = 0
        last_exception = None
        backoff_factor = 0.5

        while retries < max_retries:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}

                    # Add more detailed logging
                    logger.info(f"Calling agent API: {api_endpoint}")
                    response = await client.post(
                        api_endpoint,
                        json=payload,
                        headers=headers
                    )

                    response.raise_for_status()

                    # Log success
                    logger.info(f"Microagent API call successful, status: {response.status_code}")
                    return response.json()

            except httpx.HTTPStatusError as e:
                last_exception = e
                status_code = e.response.status_code

                # Log detailed error
                logger.error(f"HTTP error calling agent API: {status_code}, response: {e.response.text}")

                # More comprehensive retry logic
                if status_code in (429, 500, 502, 503, 504):
                    retries += 1
                    if retries < max_retries:
                        # Exponential backoff with jitter
                        wait_time = backoff_factor * (2 ** retries) + random.uniform(0, 0.1)
                        logger.warning(f"Retrying in {wait_time:.2f} seconds (attempt {retries}/{max_retries})")
                        await sleep(wait_time)
                        continue
                else:
                    # For other status codes, don't retry and provide specific error info
                    error_detail = f"Microagent API returned error: HTTP {status_code}"
                    try:
                        response_data = e.response.json()
                        if "error" in response_data:
                            error_detail += f" - {response_data['error']}"
                    except Exception:
                        pass
                    raise Exception(error_detail)

            except httpx.RequestError as e:
                last_exception = e
                logger.error(f"Network error calling agent API: {e}")
                retries += 1
                if retries < max_retries:
                    wait_time = backoff_factor * (2 ** retries)
                    logger.warning(f"Retrying in {wait_time:.2f} seconds (attempt {retries}/{max_retries})")
                    await sleep(wait_time)
                    continue

            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error calling agent API: {e}")
                raise Exception(f"Error calling agent API: {str(e)}")

        # If we've exhausted retries
        logger.error(f"Exhausted retries calling agent API: {last_exception}")
        raise Exception(f"Failed to call agent API after {max_retries} retries: {str(last_exception)}")

    def _get_metric_description(self, metric_name: str) -> str:
        """
        Get a human-readable description for an evaluation metric.

        Args:
            metric_name: Name of the metric

        Returns:
            str: Description of the metric
        """
        descriptions = {
            "faithfulness": "Measures how well the answer sticks to the information in the context without hallucinating.",
            "response_relevancy": "Measures how relevant the answer is to the query asked.",
            "context_precision": "Measures how precisely the retrieved context matches what's needed to answer the query.",
            "context_recall": "Measures how well the retrieved context covers all the information needed to answer the query.",
            "context_entity_recall": "Measures how well the retrieved context captures the entities mentioned in the reference answer.",
            "noise_sensitivity": "Measures the model's tendency to be misled by irrelevant information in the context (lower is better).",
            "correctness": "Measures how accurately the answer matches the ground truth.",
            "completeness": "Measures how completely the answer addresses all aspects of the question.",
            "coherence": "Measures how well-structured and logically connected the answer is.",
            "conciseness": "Measures how concise and to-the-point the answer is without unnecessary information.",
            "helpfulness": "Measures how helpful and actionable the answer would be to a user."
        }

        # Return description if available, otherwise provide a generic one
        return descriptions.get(
            metric_name,
            f"Measures the {metric_name.replace('_', ' ')} of the response."
        )

    async def _process_batch_item_with_client(
            self,
            client: AgentClient,
            evaluation: Evaluation,
            item: Dict[str, Any],
            item_index: int,
            formatted_prompt: str
    ) -> EvaluationResultCreate:
        """
        Process a single dataset item using agent client abstraction.

        Args:
            client: Agent client
            evaluation: Evaluation model
            item: Dataset item
            item_index: Index of the item
            formatted_prompt: Formatted prompt

        Returns:
            EvaluationResultCreate: Evaluation result
        """
        try:
            # Extract data
            query = item.get("query", "")
            context = item.get("context", "")
            ground_truth = item.get("ground_truth", "")

            # Start timing
            start_time = time.time()

            # Call the agent with the client abstraction
            response = await client.process_query(
                query=query,
                context=context,
                system_message=formatted_prompt,
                config={
                    "progress_token": item_index,
                    "evaluation_id": str(evaluation.id)
                }
            )

            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)

            # Extract results
            success = response.get("success", False)
            answer = response.get("answer", "")
            response_processing_time = response.get("processing_time_ms", 0)
            error = response.get("error")

            # Use the actual processing time from response if available, otherwise use calculated
            final_processing_time = response_processing_time if response_processing_time else processing_time

            # If there was an error, log it
            if not success or error:
                logger.warning(f"Error processing item {item_index}: {error}")

                # Return result with error details
                return EvaluationResultCreate(
                    evaluation_id=evaluation.id,
                    overall_score=0.0,
                    raw_results={"error": error},
                    dataset_sample_id=str(item_index),
                    input_data={
                        "query": query,
                        "context": context,
                        "ground_truth": ground_truth,
                        "prompt": formatted_prompt
                    },
                    output_data={
                        "error": error,
                        "answer": answer if answer else "Error occurred during processing",
                        "success": False
                    },
                    processing_time_ms=final_processing_time,
                    metric_scores=[],
                    passed=False,
                    pass_threshold=evaluation.pass_threshold or 0.7
                )

            eval_config = evaluation.config or {}
            if evaluation.metrics:
                eval_config["selected_metrics"] = evaluation.metrics

            # Calculate metrics
            metrics = await self.calculate_metrics(
                input_data={
                    "query": query,
                    "context": context,
                    "ground_truth": ground_truth
                },
                output_data={"answer": answer},
                config=eval_config
            )

            # Calculate overall score
            # overall_score = sum(metrics.values()) / len(metrics) if metrics else 0.0
            overall_score = 0.0
            normalized_scores = []

            if metrics:
                for name, value in metrics.items():
                    is_lower_better = _is_lower_better_metric(name, evaluation.method)

                    if is_lower_better:
                        if value == 0:
                            normalized_score = 1.0
                        else:
                            normalized_score = 1 / (1 + value)
                    else:
                        normalized_score = value
                    normalized_scores.append(normalized_score)

                overall_score = sum(normalized_scores) / len(normalized_scores)

            # Determine pass/fail status
            pass_threshold = evaluation.pass_threshold or 0.7  # Default to 0.7 if not specified
            passed = overall_score >= pass_threshold

            logger.info(f"Item {item_index} evaluation result - score: {overall_score:.4f}, "
                        f"threshold: {pass_threshold:.4f}, passed: {passed}")

            # Create metric scores with enhanced meta_info for consistency
            metric_scores = []
            for name, value in metrics.items():
                # Get threshold for this metric
                threshold = _get_metric_threshold(name, evaluation.method, evaluation.config)

                # Determine if metric passed
                is_lower_better = _is_lower_better_metric(name, evaluation.method)
                if is_lower_better:
                    success = value <= threshold
                else:
                    success = value >= threshold

                # Create enhanced meta_info
                meta_info = {
                    "description": self._get_metric_description(name),
                    "success": success,
                    "threshold": threshold
                }

                # Add reason if available
                if evaluation.method == EvaluationMethod.RAGAS:
                    reason = _generate_ragas_reason(name, value, threshold, success, is_lower_better)
                    meta_info["reason"] = reason

                metric_scores.append(MetricScoreCreate(
                    name=name,
                    value=value,
                    weight=1.0,
                    meta_info=meta_info
                ))

            # Create result
            return EvaluationResultCreate(
                evaluation_id=evaluation.id,
                overall_score=overall_score,
                raw_results=metrics,
                dataset_sample_id=str(item_index),
                input_data={
                    "query": query,
                    "context": context,
                    "ground_truth": ground_truth,
                    "prompt": formatted_prompt
                },
                output_data={"answer": answer},
                processing_time_ms=final_processing_time,
                metric_scores=metric_scores,
                passed=passed,
                pass_threshold=pass_threshold
            )

        except Exception as e:
            logger.exception(f"Error processing dataset item {item_index}: {e}")
            # Create error result with all required fields
            return EvaluationResultCreate(
                evaluation_id=evaluation.id,
                overall_score=0.0,
                raw_results={"error": str(e)},
                dataset_sample_id=str(item_index),
                input_data=item,
                output_data={"error": str(e)},
                processing_time_ms=0,
                metric_scores=[],
                passed=False,
                pass_threshold=evaluation.pass_threshold or 0.7
            )
