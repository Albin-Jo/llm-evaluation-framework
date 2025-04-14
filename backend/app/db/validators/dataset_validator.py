# backend/app/db/validators/dataset_validator.py
import json
import logging
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, validator

from backend.app.db.models.orm import DatasetType

logger = logging.getLogger(__name__)


class BaseDatasetValidator(BaseModel):
    """Base validator for all dataset types."""

    @classmethod
    def validate_dataset_file(cls, file_content: str, dataset_type: DatasetType) -> Dict[str, Any]:
        """
        Validate a dataset file against the schema for the specified dataset type.

        Args:
            file_content: String content of the dataset file
            dataset_type: Type of the dataset

        Returns:
            Dict[str, Any]: Validation results

        Raises:
            ValueError: If the dataset is invalid
        """
        try:
            # Parse JSON content
            data = json.loads(file_content)

            # Select appropriate validator
            validator_class = DATASET_TYPE_VALIDATORS.get(dataset_type)
            if not validator_class:
                logger.warning(f"No validator found for dataset type {dataset_type}")
                return {
                    "valid": True,
                    "warning": f"No specific validation performed for {dataset_type} datasets"
                }

            # Process based on JSON structure
            if isinstance(data, list):
                # Validate each item and collect errors
                all_errors = []
                for i, item in enumerate(data):
                    try:
                        validator_class.model_validate(item)
                    except Exception as e:
                        all_errors.append(f"Item {i}: {str(e)}")

                if all_errors:
                    raise ValueError(f"Dataset validation failed with errors: {', '.join(all_errors)}")

                return {
                    "valid": True,
                    "count": len(data)
                }

            elif isinstance(data, dict):
                # Check for containers
                container_fields = ["items", "data", "samples", "examples"]
                for field in container_fields:
                    if field in data and isinstance(data[field], list):
                        items = data[field]
                        # Validate each item and collect errors
                        all_errors = []
                        for i, item in enumerate(items):
                            try:
                                validator_class.model_validate(item)
                            except Exception as e:
                                all_errors.append(f"Item {i}: {str(e)}")

                        if all_errors:
                            raise ValueError(f"Dataset validation failed with errors: {', '.join(all_errors)}")

                        return {
                            "valid": True,
                            "count": len(items)
                        }

                # If no container found, validate the dict itself
                validator_class.model_validate(data)
                return {
                    "valid": True,
                    "count": 1
                }

            else:
                raise ValueError(f"Unexpected data type: {type(data)}. Expected JSON object or array.")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Dataset validation failed: {str(e)}")


class UserQueryDatasetValidator(BaseDatasetValidator):
    """Validator for USER_QUERY dataset type."""
    query: str = Field(..., description="The user query or question")
    context: Optional[Union[str, List[str]]] = Field(None, description="Context information")

    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v


class ContextDatasetValidator(BaseDatasetValidator):
    """Validator for CONTEXT dataset type."""
    context: Union[str, List[str]] = Field(..., description="Context information")
    reference: Optional[str] = Field(None, description="Reference answer")
    query: Optional[str] = Field(None, description="Optional query")

    @validator('context')
    def context_not_empty(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError("Context cannot be empty")
        elif isinstance(v, list) and (not v or all(not c.strip() for c in v if isinstance(c, str))):
            raise ValueError("Context list cannot be empty")
        return v


class QuestionAnswerDatasetValidator(BaseDatasetValidator):
    """Validator for QUESTION_ANSWER dataset type."""
    query: str = Field(..., description="The question")
    ground_truth: str = Field(..., description="The reference answer")
    context: Optional[Union[str, List[str]]] = Field(None, description="Optional context information")

    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v

    @validator('ground_truth')
    def ground_truth_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Ground truth cannot be empty")
        return v


class MessageModel(BaseModel):
    """Model for conversation messages."""
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant')")
    content: str = Field(..., description="Content of the message")


class ConversationDatasetValidator(BaseDatasetValidator):
    """Validator for CONVERSATION dataset type."""
    messages: List[Dict[str, str]] = Field(..., description="List of conversation messages")
    context: Optional[Union[str, List[str]]] = Field(None, description="Optional context information")

    @validator('messages')
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("Messages list cannot be empty")

        # Check that we have at least one user message
        has_user = False
        for msg in v:
            if isinstance(msg, dict) and msg.get('role', '').lower() in ('user', 'human'):
                has_user = True
                break

        if not has_user:
            raise ValueError("Conversation must contain at least one user message")

        return v


class CustomDatasetValidator(BaseDatasetValidator):
    """Validator for CUSTOM dataset type with flexible schema."""

    # Allow any fields
    class Config:
        extra = "allow"


# Mapping of dataset types to validators
DATASET_TYPE_VALIDATORS = {
    DatasetType.USER_QUERY: UserQueryDatasetValidator,
    DatasetType.CONTEXT: ContextDatasetValidator,
    DatasetType.QUESTION_ANSWER: QuestionAnswerDatasetValidator,
    DatasetType.CONVERSATION: ConversationDatasetValidator,
    DatasetType.CUSTOM: CustomDatasetValidator
}


def validate_dataset_schema(file_content: str, dataset_type: DatasetType) -> Dict[str, Any]:
    """
    Validate a dataset file against its schema.

    Args:
        file_content: String content of the dataset file
        dataset_type: Type of the dataset

    Returns:
        Dict[str, Any]: Validation results with schema details
    """
    # Get validation results
    validation_results = BaseDatasetValidator.validate_dataset_file(file_content, dataset_type)

    # Add schema information
    validation_results["schema"] = get_dataset_schema(dataset_type)

    return validation_results


def get_dataset_schema(dataset_type: DatasetType) -> Dict[str, Any]:
    """
    Get the schema definition for a dataset type.

    Args:
        dataset_type: Type of the dataset

    Returns:
        Dict[str, Any]: Schema definition
    """
    # Get required and optional fields for the dataset type
    validator_class = DATASET_TYPE_VALIDATORS.get(dataset_type)
    if not validator_class:
        return {
            "required_fields": [],
            "optional_fields": [],
            "description": f"No schema defined for {dataset_type} datasets"
        }

    schema = validator_class.model_json_schema()

    # Extract required and optional fields
    required_fields = schema.get("required", [])

    # Get all properties excluding required ones
    all_properties = set(schema.get("properties", {}).keys())
    optional_fields = all_properties - set(required_fields)

    # Get descriptions for fields
    field_descriptions = {}
    for field_name, field_info in schema.get("properties", {}).items():
        field_descriptions[field_name] = field_info.get("description", "")

    return {
        "required_fields": required_fields,
        "optional_fields": list(optional_fields),
        "field_descriptions": field_descriptions,
        "description": schema.get("description", f"Schema for {dataset_type} datasets")
    }


def get_supported_metrics_for_schema(dataset_schema: Dict[str, Any]) -> Set[str]:
    """
    Get the set of metrics that can be supported by a dataset with the given schema.

    Args:
        dataset_schema: Schema definition with required and optional fields

    Returns:
        Set[str]: Set of supported metric names
    """
    from backend.app.evaluation.metrics.ragas_metrics import METRIC_REQUIREMENTS

    # Combine required and optional fields
    available_fields = set(dataset_schema.get("required_fields", []))
    available_fields.update(dataset_schema.get("optional_fields", []))

    # Map field names to standardized names
    field_mapping = {
        "query": "query",
        "question": "query",
        "user_query": "query",
        "context": "context",
        "retrieved_context": "context",
        "retrieved_contexts": "context",
        "ground_truth": "ground_truth",
        "reference": "ground_truth",
        "answer": "answer",
        "response": "answer"
    }

    standardized_fields = set()
    for field in available_fields:
        if field in field_mapping:
            standardized_fields.add(field_mapping[field])

    # Check which metrics can be calculated
    supported_metrics = set()
    for metric_name, requirements in METRIC_REQUIREMENTS.items():
        required_fields = set(requirements.get("required_fields", []))
        if required_fields.issubset(standardized_fields):
            supported_metrics.add(metric_name)

    return supported_metrics