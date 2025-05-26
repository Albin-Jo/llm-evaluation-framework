import logging
from typing import Any, Dict, List

from backend.app.db.models.orm import Dataset, DatasetType

# Configure logging
logger = logging.getLogger(__name__)

# Only import deepeval if available
try:
    from deepeval.dataset import EvaluationDataset
    from deepeval.test_case import LLMTestCase

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    LLMTestCase = None
    EvaluationDataset = None


class DatasetAdapter:
    """Converts dataset formats to DeepEval TestCases."""

    def __init__(self):
        """Initialize the dataset adapter."""
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("DeepEval library is required for DatasetAdapter")

    async def convert_to_deepeval_dataset(
            self, dataset: Dataset, dataset_content: List[Dict[str, Any]]
    ) -> EvaluationDataset:
        """
        Convert dataset to DeepEval format.

        Args:
            dataset: Dataset model instance
            dataset_content: List of dataset items

        Returns:
            EvaluationDataset: Converted dataset for DeepEval

        Raises:
            ValueError: If dataset type is unsupported
        """
        logger.info(
            f"Converting dataset {dataset.id} (type: {dataset.type}) to DeepEval format"
        )

        conversion_methods = {
            DatasetType.USER_QUERY: self._convert_user_query_dataset,
            DatasetType.QUESTION_ANSWER: self._convert_qa_dataset,
            DatasetType.CONTEXT: self._convert_context_dataset,
            DatasetType.CONVERSATION: self._convert_conversation_dataset,
            DatasetType.CUSTOM: self._convert_custom_dataset,
        }

        converter = conversion_methods.get(dataset.type)
        if not converter:
            raise ValueError(f"Unsupported dataset type: {dataset.type}")

        test_cases = await converter(dataset_content)

        logger.info(f"Converted {len(test_cases)} items to DeepEval TestCases")
        return EvaluationDataset(test_cases=test_cases)

    async def _convert_user_query_dataset(
            self, dataset_content: List[Dict[str, Any]]
    ) -> List[LLMTestCase]:
        """Convert USER_QUERY dataset to TestCases."""
        test_cases = []
        query_fields = ["query", "question", "user_query", "input"]
        answer_fields = [
            "expected_answer",
            "expected_output",
            "answer",
            "ground_truth",
        ]

        for i, item in enumerate(dataset_content):
            try:
                query = self._extract_field_value(item, query_fields, "")
                expected_output = self._extract_field_value(item, answer_fields)
                context = self._normalize_context(item.get("context", []))

                test_case = LLMTestCase(
                    input=query,
                    actual_output=None,  # Will be filled by agent response
                    expected_output=expected_output,
                    context=context,
                    retrieval_context=item.get("retrieved_context", []),
                )

                # Add metadata if available
                metadata_fields = ["category", "difficulty", "source"]
                metadata = self._extract_metadata(item, metadata_fields, i)
                if metadata:
                    test_case.additional_metadata = metadata

                test_cases.append(test_case)

            except Exception as e:
                logger.error(f"Error converting USER_QUERY item {i}: {e}")
                test_cases.append(self._create_error_test_case(i, e))

        return test_cases

    async def _convert_qa_dataset(
            self, dataset_content: List[Dict[str, Any]]
    ) -> List[LLMTestCase]:
        """Convert QUESTION_ANSWER dataset to TestCases."""
        test_cases = []
        question_fields = ["question", "query", "input"]
        answer_fields = ["answer", "expected_answer", "ground_truth"]

        for i, item in enumerate(dataset_content):
            try:
                question = self._extract_field_value(item, question_fields, "")
                answer = self._extract_field_value(item, answer_fields, "")
                context = self._normalize_context(
                    item.get("supporting_context", item.get("context", []))
                )

                test_case = LLMTestCase(
                    input=question,
                    actual_output=None,  # Will be filled by agent response
                    expected_output=answer,
                    context=context,
                    retrieval_context=item.get("retrieved_context", []),
                )

                # Add QA-specific metadata
                metadata_fields = ["category", "difficulty_level", "source", "topic"]
                metadata = self._extract_metadata(item, metadata_fields, i)
                if metadata:
                    test_case.additional_metadata = metadata

                test_cases.append(test_case)

            except Exception as e:
                logger.error(f"Error converting QUESTION_ANSWER item {i}: {e}")
                test_cases.append(self._create_error_test_case(i, e, "QA"))

        return test_cases

    async def _convert_context_dataset(
            self, dataset_content: List[Dict[str, Any]]
    ) -> List[LLMTestCase]:
        """Convert CONTEXT dataset to TestCases."""
        test_cases = []
        query_fields = ["query", "question", "input"]
        output_fields = ["expected_output", "expected_answer", "ground_truth"]

        for i, item in enumerate(dataset_content):
            try:
                query = self._extract_field_value(item, query_fields, "")
                expected_output = self._extract_field_value(item, output_fields)

                contexts = item.get("contexts", item.get("context", []))
                contexts = self._normalize_context(contexts)

                test_case = LLMTestCase(
                    input=query,
                    actual_output=None,  # Will be filled by agent response
                    expected_output=expected_output,
                    context=contexts,
                    retrieval_context=item.get("retrieved_contexts", []),
                )

                # Add context-specific metadata
                context_metadata = {}
                if "retrieval_score" in item:
                    context_metadata["retrieval_score"] = item["retrieval_score"]
                if "context_relevance" in item:
                    context_metadata["context_relevance"] = item["context_relevance"]

                if context_metadata:
                    context_metadata["original_index"] = i
                    test_case.additional_metadata = context_metadata

                test_cases.append(test_case)

            except Exception as e:
                logger.error(f"Error converting CONTEXT item {i}: {e}")
                test_cases.append(self._create_error_test_case(i, e, "context"))

        return test_cases

    async def _convert_conversation_dataset(
            self, dataset_content: List[Dict[str, Any]]
    ) -> List[LLMTestCase]:
        """Convert CONVERSATION dataset to TestCases."""
        test_cases = []

        for i, item in enumerate(dataset_content):
            try:
                conversation = item.get("conversation", item.get("messages", []))

                if isinstance(conversation, list) and conversation:
                    query, expected_output, context = self._process_conversation(
                        conversation
                    )
                else:
                    # Fallback for non-standard conversation format
                    query = item.get(
                        "input", item.get("query", f"Conversation item {i}")
                    )
                    expected_output = item.get(
                        "expected_output", item.get("response")
                    )
                    context = self._normalize_context(item.get("context", []))

                test_case = LLMTestCase(
                    input=query,
                    actual_output=None,  # Will be filled by agent response
                    expected_output=expected_output,
                    context=context,
                )

                # Add conversation-specific metadata
                conversation_length = (
                    len(conversation) if isinstance(conversation, list) else 1
                )
                test_case.additional_metadata = {
                    "conversation_length": conversation_length,
                    "original_index": i,
                }

                test_cases.append(test_case)

            except Exception as e:
                logger.error(f"Error converting CONVERSATION item {i}: {e}")
                test_cases.append(self._create_error_test_case(i, e, "conversation"))

        return test_cases

    async def _convert_custom_dataset(
            self, dataset_content: List[Dict[str, Any]]
    ) -> List[LLMTestCase]:
        """Convert CUSTOM dataset to TestCases with flexible field mapping."""
        test_cases = []

        # Common field names to try for each component
        input_fields = ["input", "query", "question", "prompt", "text"]
        output_fields = [
            "output",
            "expected_output",
            "answer",
            "expected_answer",
            "ground_truth",
            "response",
        ]
        context_fields = [
            "context",
            "contexts",
            "supporting_context",
            "background",
            "documents",
        ]

        for i, item in enumerate(dataset_content):
            try:
                # Try to find input field
                query = self._extract_field_value(item, input_fields)
                if not query:
                    logger.warning(f"No input field found in custom dataset item {i}")
                    query = f"Custom dataset item {i}"

                # Try to find expected output field
                expected_output = self._extract_field_value(item, output_fields)

                # Try to find context field
                context = []
                for field in context_fields:
                    if field in item and item[field]:
                        context = self._normalize_context(item[field])
                        break

                test_case = LLMTestCase(
                    input=query,
                    actual_output=None,  # Will be filled by agent response
                    expected_output=expected_output,
                    context=context,
                )

                # Add all remaining fields as metadata
                excluded_fields = set(input_fields + output_fields + context_fields)
                metadata = {
                    k: v for k, v in item.items() if k not in excluded_fields
                }
                if metadata:
                    metadata["original_index"] = i
                    test_case.additional_metadata = metadata

                test_cases.append(test_case)

            except Exception as e:
                logger.error(f"Error converting CUSTOM dataset item {i}: {e}")
                test_cases.append(self._create_error_test_case(i, e, "custom"))

        return test_cases

    def validate_dataset_for_deepeval(
            self, dataset_content: List[Dict[str, Any]], metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Validate dataset compatibility with selected DeepEval metrics.

        Args:
            dataset_content: List of dataset items to validate
            metrics: List of metric names to validate against

        Returns:
            Dict containing validation results and statistics
        """
        validation_results = {
            "compatible": True,
            "warnings": [],
            "requirements": [],
            "statistics": {
                "total_items": len(dataset_content),
                "items_with_context": 0,
                "items_with_expected_output": 0,
                "items_with_input": 0,
            },
        }

        # Analyze dataset content
        input_fields = {"input", "query", "question"}
        context_fields = {"context", "contexts", "supporting_context"}
        output_fields = {"expected_output", "answer", "ground_truth"}

        for item in dataset_content:
            item_keys = set(item.keys())

            if item_keys & input_fields:
                validation_results["statistics"]["items_with_input"] += 1

            if item_keys & context_fields:
                validation_results["statistics"]["items_with_context"] += 1

            if item_keys & output_fields:
                validation_results["statistics"]["items_with_expected_output"] += 1

        # Check metric requirements
        self._validate_metrics_requirements(metrics, validation_results)

        return validation_results

    def _extract_field_value(
            self, item: Dict[str, Any], field_names: List[str], default: Any = None
    ) -> Any:
        """Extract value from item using list of possible field names."""
        for field_name in field_names:
            if field_name in item and item[field_name]:
                return str(item[field_name])
        return default

    def _normalize_context(self, context: Any) -> List[str]:
        """Normalize context to list of strings."""
        if isinstance(context, list):
            return context
        elif isinstance(context, str):
            return [context]
        elif context is None:
            return []
        else:
            return [str(context)]

    def _extract_metadata(
            self, item: Dict[str, Any], metadata_fields: List[str], index: int
    ) -> Dict[str, Any]:
        """Extract metadata from item."""
        metadata = {}
        for field in metadata_fields:
            if field in item:
                # Map difficulty_level to difficulty for consistency
                key = "difficulty" if field == "difficulty_level" else field
                metadata[key] = item[field]

        if metadata:
            metadata["original_index"] = index

        return metadata

    def _create_error_test_case(
            self, index: int, error: Exception, dataset_type: str = ""
    ) -> LLMTestCase:
        """Create a test case for error handling."""
        error_prefix = f"Error processing {dataset_type} item" if dataset_type else "Error processing item"

        return LLMTestCase(
            input=f"{error_prefix} {index}",
            actual_output='',
            expected_output="Error in data",
            context=[f"Original item had error: {str(error)}"],
        )

    def _process_conversation(
            self, conversation: List[Dict[str, Any]]
    ) -> tuple[str, str, List[str]]:
        """Process conversation to extract query, expected output, and context."""
        user_messages = [msg for msg in conversation if msg.get("role") == "user"]
        assistant_messages = [
            msg for msg in conversation if msg.get("role") == "assistant"
        ]

        if user_messages:
            query = user_messages[-1].get("content", "")
            expected_output = (
                assistant_messages[-1].get("content", "") if assistant_messages else None
            )

            # Use previous conversation as context
            context_messages = conversation[:-1] if len(conversation) > 1 else []
            context = [
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in context_messages
            ]
        else:
            query = ""
            expected_output = None
            context = []

        return query, expected_output, context

    def _validate_metrics_requirements(
            self, metrics: List[str], validation_results: Dict[str, Any]
    ) -> None:
        """Validate that dataset meets metric requirements."""
        stats = validation_results["statistics"]

        metric_validators = {
            "faithfulness": lambda: self._check_context_requirement(
                stats, validation_results, "Faithfulness metric requires context data"
            ),
            "answer_relevancy": lambda: self._check_input_requirement(
                stats, validation_results, "Answer relevancy requires input questions"
            ),
            "contextual_precision": lambda: self._check_context_and_output_requirements(
                stats, validation_results, "contextual_precision"
            ),
            "contextual_recall": lambda: self._check_context_and_output_requirements(
                stats, validation_results, "contextual_recall"
            ),
        }

        for metric in metrics:
            validator = metric_validators.get(metric)
            if validator:
                validator()

    def _check_context_requirement(
            self, stats: Dict[str, int], validation_results: Dict[str, Any], message: str
    ) -> None:
        """Check if context requirement is met."""
        if stats["items_with_context"] == 0:
            validation_results["warnings"].append(
                f"{message}. No context found in dataset."
            )

    def _check_input_requirement(
            self, stats: Dict[str, int], validation_results: Dict[str, Any], message: str
    ) -> None:
        """Check if input requirement is met."""
        if stats["items_with_input"] == 0:
            validation_results["compatible"] = False
            validation_results["requirements"].append(message)

    def _check_context_and_output_requirements(
            self, stats: Dict[str, int], validation_results: Dict[str, Any], metric_name: str
    ) -> None:
        """Check context and output requirements for contextual metrics."""
        if stats["items_with_expected_output"] == 0:
            validation_results["warnings"].append(
                f"{metric_name} works better with expected outputs for comparison."
            )
        if stats["items_with_context"] == 0:
            validation_results["warnings"].append(
                f"{metric_name} requires context data for proper evaluation."
            )
