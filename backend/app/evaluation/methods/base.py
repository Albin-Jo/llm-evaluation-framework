# File: app/evaluation/methods/base.py
import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm.models import Dataset, Evaluation, MicroAgent, Prompt
from backend.app.db.schema.evaluation_schema import EvaluationResultCreate, MetricScoreCreate

# Add logger configuration
logger = logging.getLogger(__name__)


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

    async def get_microagent(self, microagent_id: UUID) -> Optional[MicroAgent]:
        """
        Get MicroAgent by ID.

        Args:
            microagent_id: MicroAgent ID

        Returns:
            Optional[MicroAgent]: MicroAgent if found, None otherwise
        """
        from backend.app.db.repositories.base import BaseRepository
        repo = BaseRepository(MicroAgent, self.db_session)
        return await repo.get(microagent_id)

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
        Load dataset from storage.

        Args:
            dataset: Dataset model

        Returns:
            List[Dict[str, Any]]: Dataset items
        """
        from backend.app.services.storage import get_storage_service

        # Get storage service
        storage_service = get_storage_service()

        try:
            # Load dataset file
            data = await storage_service.read_file(dataset.file_path)

            # Parse dataset based on type
            if dataset.type.value.endswith("json"):
                import json
                try:
                    parsed_data = json.loads(data)
                    # Ensure we return a list
                    if isinstance(parsed_data, list):
                        return parsed_data
                    else:
                        return [parsed_data]
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON dataset: {e}")
                    raise ValueError(f"Invalid JSON in dataset file: {e}")

            elif dataset.type.value.endswith("csv"):
                import csv
                import io
                csv_data = []
                try:
                    csv_file = io.StringIO(data)
                    reader = csv.DictReader(csv_file)
                    for row in reader:
                        csv_data.append(dict(row))
                    return csv_data
                except csv.Error as e:
                    logger.error(f"Error parsing CSV dataset: {e}")
                    raise ValueError(f"Invalid CSV in dataset file: {e}")

            else:
                # Return raw data for custom processing
                return [{"data": data}]

        except FileNotFoundError:
            logger.error(f"Dataset file not found: {dataset.file_path}")
            raise ValueError(f"Dataset file not found: {dataset.file_path}")

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise ValueError(f"Error loading dataset: {str(e)}")

    # Add this new method for batch processing
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

    # Add this helper method for tracking progress
    async def log_progress(self, evaluation_id: UUID, total: int, processed: int) -> None:
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
    async def run_evaluation(self, evaluation: Evaluation) -> List[EvaluationResultCreate]:
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

    # File: app/evaluation/methods/base.py
    # Add this method to the BaseEvaluationMethod class

    async def batch_process(
            self,
            evaluation: Evaluation,
            batch_size: int = 10
    ) -> List[EvaluationResultCreate]:
        """
        Process dataset items in batches with improved performance.

        Args:
            evaluation: Evaluation model
            batch_size: Number of items to process in each batch

        Returns:
            List[EvaluationResultCreate]: List of evaluation results
        """
        # Get related entities - fetch all at once
        microagent = await self.get_microagent(evaluation.micro_agent_id)
        dataset = await self.get_dataset(evaluation.dataset_id)
        prompt = await self.get_prompt(evaluation.prompt_id)

        if not microagent or not dataset or not prompt:
            logger.error(f"Missing required entities for evaluation {evaluation.id}")
            return []

        # Load dataset
        dataset_items = await self.load_dataset(dataset)
        all_results = []

        # Create a reusable HTTP client
        from backend.app.core.config import settings
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
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
                        self._process_batch_item(
                            client=client,
                            evaluation=evaluation,
                            microagent=microagent,
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

                # Log progress
                await self.log_progress(
                    evaluation.id,
                    len(dataset_items),
                    batch_end
                )

        return all_results

    async def _process_batch_item(
            self,
            client: httpx.AsyncClient,
            evaluation: Evaluation,
            microagent: MicroAgent,
            item: Dict[str, Any],
            item_index: int,
            formatted_prompt: str
    ) -> EvaluationResultCreate:
        """
        Process a single dataset item within a batch.

        Args:
            client: HTTP client
            evaluation: Evaluation model
            microagent: MicroAgent model
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

            # Call the microagent API using the shared client
            from backend.app.core.config import settings

            start_time = time.time()
            response = await client.post(
                microagent.api_endpoint,
                json={
                    "prompt": formatted_prompt,
                    "query": query,
                    "context": context
                },
                headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
            )
            response.raise_for_status()
            response_data = response.json()
            processing_time = int((time.time() - start_time) * 1000)

            # Extract answer
            answer = response_data.get("answer", "")

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
            raise e

    async def _call_microagent_api_with_retry(
            self, api_endpoint: str, payload: Dict[str, Any], max_retries: int = 3
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
                    logger.debug(f"Calling microagent API: {api_endpoint}")
                    response = await client.post(
                        api_endpoint,
                        json=payload,
                        headers=headers
                    )

                    response.raise_for_status()

                    # Log success
                    logger.debug(f"Microagent API call successful, status: {response.status_code}")
                    return response.json()

            except httpx.HTTPStatusError as e:
                last_exception = e
                status_code = e.response.status_code

                # Log detailed error
                logger.error(f"HTTP error calling microagent API: {status_code}, response: {e.response.text}")

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
                logger.error(f"Network error calling microagent API: {e}")
                retries += 1
                if retries < max_retries:
                    wait_time = backoff_factor * (2 ** retries)
                    logger.warning(f"Retrying in {wait_time:.2f} seconds (attempt {retries}/{max_retries})")
                    await sleep(wait_time)
                    continue

            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error calling microagent API: {e}")
                raise Exception(f"Error calling microagent API: {str(e)}")

        # If we've exhausted retries
        logger.error(f"Exhausted retries calling microagent API: {last_exception}")
        raise Exception(f"Failed to call microagent API after {max_retries} retries: {str(last_exception)}")