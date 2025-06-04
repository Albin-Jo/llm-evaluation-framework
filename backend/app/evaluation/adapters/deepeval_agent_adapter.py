import logging
from typing import List, Optional
import asyncio

from deepeval.test_case import LLMTestCase
from backend.app.db.models.orm import Agent, Prompt
from backend.app.services.agent_clients.base import AgentClient
from backend.app.evaluation.adapters.prompt_adapter import PromptAdapter

logger = logging.getLogger(__name__)


class DeepEvalAgentAdapter:
    """Adapter to generate agent responses for DeepEval test cases."""

    def __init__(self):
        self.prompt_adapter = PromptAdapter()

    async def generate_responses_for_test_cases(
            self,
            test_cases: List[LLMTestCase],
            agent_client: AgentClient,
            prompt: Prompt,
            batch_size: int = 5
    ) -> List[LLMTestCase]:
        """
        Generate actual agent responses for DeepEval test cases.

        Args:
            test_cases: List of test cases to generate responses for
            agent_client: Agent client to use for generation
            prompt: Prompt template to use
            batch_size: Number of concurrent requests

        Returns:
            List of test cases with actual_output filled
        """
        logger.info(f"Generating responses for {len(test_cases)} test cases")

        # Process in batches to avoid overwhelming the agent
        processed_cases = []

        for batch_start in range(0, len(test_cases), batch_size):
            batch_end = min(batch_start + batch_size, len(test_cases))
            batch = test_cases[batch_start:batch_end]

            # Create tasks for concurrent processing
            tasks = []
            for test_case in batch:
                task = asyncio.create_task(
                    self._generate_single_response(test_case, agent_client, prompt)
                )
                tasks.append(task)

            # Wait for batch completion
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error generating response for test case {batch_start + i}: {result}")
                    batch[i].actual_output = f"Error: {str(result)}"
                else:
                    batch[i].actual_output = result

                processed_cases.append(batch[i])

            logger.info(f"Completed batch {batch_start}-{batch_end}")

        return processed_cases

    async def _generate_single_response(
            self,
            test_case: LLMTestCase,
            agent_client: AgentClient,
            prompt: Prompt
    ) -> str:
        """Generate a single response for a test case."""
        try:
            # Use the prompt adapter to apply the template
            response = await self.prompt_adapter.apply_prompt_to_agent_client(
                agent_client=agent_client,
                prompt=prompt,
                test_input=test_case.input,
                context=test_case.context or []
            )
            return response

        except Exception as e:
            logger.error(f"Error generating response for input '{test_case.input[:50]}...': {e}")
            return f"Error generating response: {str(e)}"


