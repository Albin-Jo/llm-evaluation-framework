import logging
from typing import List, Dict, Any

from backend.app.db.models.orm import Dataset, DatasetType

# Configure logging
logger = logging.getLogger(__name__)

# Only import deepeval if available
try:
    from deepeval.test_case import TestCase
    from deepeval.dataset import EvaluationDataset

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    TestCase = None
    EvaluationDataset = None


class DatasetAdapter:
    """Converts your dataset formats to DeepEval TestCases."""

    def __init__(self):
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("DeepEval library is required for DatasetAdapter")

    async def convert_to_deepeval_dataset(
            self,
            dataset: Dataset,
            dataset_content: List[Dict[str, Any]]
    ) -> EvaluationDataset:
        """Convert your dataset to DeepEval format."""

        logger.info(f"Converting dataset {dataset.id} (type: {dataset.type}) to DeepEval format")

        if dataset.type == DatasetType.USER_QUERY:
            test_cases = await self._convert_user_query_dataset(dataset_content)
        elif dataset.type == DatasetType.QUESTION_ANSWER:
            test_cases = await self._convert_qa_dataset(dataset_content)
        elif dataset.type == DatasetType.CONTEXT:
            test_cases = await self._convert_context_dataset(dataset_content)
        elif dataset.type == DatasetType.CONVERSATION:
            test_cases = await self._convert_conversation_dataset(dataset_content)
        elif dataset.type == DatasetType.CUSTOM:
            test_cases = await self._convert_custom_dataset(dataset_content)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset.type}")

        logger.info(f"Converted {len(test_cases)} items to DeepEval TestCases")
        return EvaluationDataset(test_cases=test_cases)

    async def _convert_user_query_dataset(
            self,
            dataset_content: List[Dict[str, Any]]
    ) -> List[TestCase]:
        """Convert USER_QUERY dataset to TestCases."""
        test_cases = []

        for i, item in enumerate(dataset_content):
            try:
                # Extract fields with fallbacks for different naming conventions
                query = (
                        item.get('query') or
                        item.get('question') or
                        item.get('user_query') or
                        item.get('input', '')
                )

                expected_output = (
                        item.get('expected_answer') or
                        item.get('expected_output') or
                        item.get('answer') or
                        item.get('ground_truth')
                )

                context = item.get('context', [])
                if isinstance(context, str):
                    context = [context]
                elif context is None:
                    context = []

                test_case = TestCase(
                    input=query,
                    actual_output=None,  # Will be filled by agent response
                    expected_output=expected_output,
                    context=context,
                    retrieval_context=item.get('retrieved_context', [])
                )

                # Add additional metadata
                if item.get('category') or item.get('difficulty') or item.get('source'):
                    test_case.additional_metadata = {
                        'category': item.get('category'),
                        'difficulty': item.get('difficulty'),
                        'source': item.get('source'),
                        'original_index': i
                    }

                test_cases.append(test_case)

            except Exception as e:
                logger.error(f"Error converting USER_QUERY item {i}: {e}")
                # Create minimal test case to avoid breaking the evaluation
                test_cases.append(TestCase(
                    input=f"Error processing item {i}",
                    actual_output=None,
                    expected_output="Error in data",
                    context=[f"Original item had error: {str(e)}"]
                ))

        return test_cases

    async def _convert_qa_dataset(
            self,
            dataset_content: List[Dict[str, Any]]
    ) -> List[TestCase]:
        """Convert QUESTION_ANSWER dataset to TestCases."""
        test_cases = []

        for i, item in enumerate(dataset_content):
            try:
                question = (
                        item.get('question') or
                        item.get('query') or
                        item.get('input', '')
                )

                answer = (
                        item.get('answer') or
                        item.get('expected_answer') or
                        item.get('ground_truth', '')
                )

                context = item.get('supporting_context', item.get('context', []))
                if isinstance(context, str):
                    context = [context]
                elif context is None:
                    context = []

                test_case = TestCase(
                    input=question,
                    actual_output=None,  # Will be filled by agent response
                    expected_output=answer,
                    context=context,
                    retrieval_context=item.get('retrieved_context', [])
                )

                # Add QA-specific metadata
                if any(key in item for key in ['category', 'difficulty_level', 'source', 'topic']):
                    test_case.additional_metadata = {
                        'category': item.get('category'),
                        'difficulty': item.get('difficulty_level'),
                        'source': item.get('source'),
                        'topic': item.get('topic'),
                        'original_index': i
                    }

                test_cases.append(test_case)

            except Exception as e:
                logger.error(f"Error converting QUESTION_ANSWER item {i}: {e}")
                test_cases.append(TestCase(
                    input=f"Error processing QA item {i}",
                    actual_output=None,
                    expected_output="Error in data",
                    context=[f"Original item had error: {str(e)}"]
                ))

        return test_cases

    async def _convert_context_dataset(
            self,
            dataset_content: List[Dict[str, Any]]
    ) -> List[TestCase]:
        """Convert CONTEXT dataset to TestCases."""
        test_cases = []

        for i, item in enumerate(dataset_content):
            try:
                query = (
                        item.get('query') or
                        item.get('question') or
                        item.get('input', '')
                )

                contexts = item.get('contexts', item.get('context', []))
                if isinstance(contexts, str):
                    contexts = [contexts]
                elif contexts is None:
                    contexts = []

                expected_output = (
                        item.get('expected_output') or
                        item.get('expected_answer') or
                        item.get('ground_truth')
                )

                test_case = TestCase(
                    input=query,
                    actual_output=None,  # Will be filled by agent response
                    expected_output=expected_output,
                    context=contexts,
                    retrieval_context=item.get('retrieved_contexts', [])
                )

                # Add context-specific metadata
                if 'retrieval_score' in item or 'context_relevance' in item:
                    test_case.additional_metadata = {
                        'retrieval_score': item.get('retrieval_score'),
                        'context_relevance': item.get('context_relevance'),
                        'original_index': i
                    }

                test_cases.append(test_case)

            except Exception as e:
                logger.error(f"Error converting CONTEXT item {i}: {e}")
                test_cases.append(TestCase(
                    input=f"Error processing context item {i}",
                    actual_output=None,
                    expected_output="Error in data",
                    context=[f"Original item had error: {str(e)}"]
                ))

        return test_cases

    async def _convert_conversation_dataset(
            self,
            dataset_content: List[Dict[str, Any]]
    ) -> List[TestCase]:
        """Convert CONVERSATION dataset to TestCases."""
        test_cases = []

        for i, item in enumerate(dataset_content):
            try:
                # Handle conversation format - could be multiple turns
                conversation = item.get('conversation', item.get('messages', []))

                if isinstance(conversation, list) and len(conversation) > 0:
                    # Take the last user message as input
                    user_messages = [msg for msg in conversation if msg.get('role') == 'user']
                    assistant_messages = [msg for msg in conversation if msg.get('role') == 'assistant']

                    if user_messages:
                        query = user_messages[-1].get('content', '')
                        expected_output = assistant_messages[-1].get('content', '') if assistant_messages else None

                        # Use previous conversation as context
                        context_messages = conversation[:-1] if len(conversation) > 1 else []
                        context = [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in
                                   context_messages]
                    else:
                        query = item.get('input', f"Conversation item {i}")
                        expected_output = item.get('expected_output')
                        context = []
                else:
                    # Fallback for non-standard conversation format
                    query = item.get('input', item.get('query', f"Conversation item {i}"))
                    expected_output = item.get('expected_output', item.get('response'))
                    context = item.get('context', [])

                test_case = TestCase(
                    input=query,
                    actual_output=None,  # Will be filled by agent response
                    expected_output=expected_output,
                    context=context if isinstance(context, list) else [context] if context else []
                )

                # Add conversation-specific metadata
                test_case.additional_metadata = {
                    'conversation_length': len(conversation) if isinstance(conversation, list) else 1,
                    'original_index': i
                }

                test_cases.append(test_case)

            except Exception as e:
                logger.error(f"Error converting CONVERSATION item {i}: {e}")
                test_cases.append(TestCase(
                    input=f"Error processing conversation item {i}",
                    actual_output=None,
                    expected_output="Error in data",
                    context=[f"Original item had error: {str(e)}"]
                ))

        return test_cases

    async def _convert_custom_dataset(
            self,
            dataset_content: List[Dict[str, Any]]
    ) -> List[TestCase]:
        """Convert CUSTOM dataset to TestCases with flexible field mapping."""
        test_cases = []

        # Common field names to try for each component
        input_fields = ['input', 'query', 'question', 'prompt', 'text']
        output_fields = ['output', 'expected_output', 'answer', 'expected_answer', 'ground_truth', 'response']
        context_fields = ['context', 'contexts', 'supporting_context', 'background', 'documents']

        for i, item in enumerate(dataset_content):
            try:
                # Try to find input field
                query = None
                for field in input_fields:
                    if field in item and item[field]:
                        query = str(item[field])
                        break

                if not query:
                    logger.warning(f"No input field found in custom dataset item {i}")
                    query = f"Custom dataset item {i}"

                # Try to find expected output field
                expected_output = None
                for field in output_fields:
                    if field in item and item[field]:
                        expected_output = str(item[field])
                        break

                # Try to find context field
                context = []
                for field in context_fields:
                    if field in item and item[field]:
                        ctx = item[field]
                        if isinstance(ctx, list):
                            context = ctx
                        elif isinstance(ctx, str):
                            context = [ctx]
                        break

                test_case = TestCase(
                    input=query,
                    actual_output=None,  # Will be filled by agent response
                    expected_output=expected_output,
                    context=context
                )

                # Add all remaining fields as metadata
                metadata = {k: v for k, v in item.items()
                            if k not in input_fields + output_fields + context_fields}
                if metadata:
                    test_case.additional_metadata = {
                        **metadata,
                        'original_index': i
                    }

                test_cases.append(test_case)

            except Exception as e:
                logger.error(f"Error converting CUSTOM dataset item {i}: {e}")
                test_cases.append(TestCase(
                    input=f"Error processing custom item {i}",
                    actual_output=None,
                    expected_output="Error in data",
                    context=[f"Original item had error: {str(e)}"]
                ))

        return test_cases

    def validate_dataset_for_deepeval(self, dataset_content: List[Dict[str, Any]], metrics: List[str]) -> Dict[
        str, Any]:
        """Validate that dataset is compatible with selected DeepEval metrics."""
        validation_results = {
            'compatible': True,
            'warnings': [],
            'requirements': [],
            'statistics': {
                'total_items': len(dataset_content),
                'items_with_context': 0,
                'items_with_expected_output': 0,
                'items_with_input': 0
            }
        }

        # Analyze dataset content
        for item in dataset_content:
            # Check for input data
            if any(field in item for field in ['input', 'query', 'question']):
                validation_results['statistics']['items_with_input'] += 1

            # Check for context data
            if any(field in item for field in ['context', 'contexts', 'supporting_context']):
                validation_results['statistics']['items_with_context'] += 1

            # Check for expected output
            if any(field in item for field in ['expected_output', 'answer', 'ground_truth']):
                validation_results['statistics']['items_with_expected_output'] += 1

        # Check metric requirements
        for metric in metrics:
            if metric == 'faithfulness':
                if validation_results['statistics']['items_with_context'] == 0:
                    validation_results['warnings'].append(
                        "Faithfulness metric requires context data. No context found in dataset."
                    )

            elif metric == 'answer_relevancy':
                if validation_results['statistics']['items_with_input'] == 0:
                    validation_results['compatible'] = False
                    validation_results['requirements'].append(
                        "Answer relevancy requires input questions."
                    )

            elif metric in ['contextual_precision', 'contextual_recall']:
                if validation_results['statistics']['items_with_expected_output'] == 0:
                    validation_results['warnings'].append(
                        f"{metric} works better with expected outputs for comparison."
                    )
                if validation_results['statistics']['items_with_context'] == 0:
                    validation_results['warnings'].append(
                        f"{metric} requires context data for proper evaluation."
                    )

        return validation_results
