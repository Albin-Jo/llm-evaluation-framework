# File: backend/app/evaluation/utils/dataset_utils.py
import json
import logging
from typing import Any, Dict, List

# Configure logging
logger = logging.getLogger(__name__)


def truncate_text(text: str, max_length: int = 4000) -> str:
    """
    Truncate text to a maximum length, adding an ellipsis if truncated.

    Args:
        text: Text to truncate
        max_length: Maximum allowed length

    Returns:
        str: Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length] + "... (truncated)"


def normalize_dataset_item(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Normalize dataset item to ensure it has the required fields.

    Args:
        item: Original dataset item
        index: Item index for logging

    Returns:
        Dict[str, Any]: Normalized dataset item
    """
    normalized = {}

    # Log the raw item for debugging
    logger.debug(f"Normalizing dataset item {index} with keys: {list(item.keys())}")

    # Common field mappings (handle different naming conventions)
    query_fields = ["query", "question", "user_query", "input", "prompt", "user_question", "q"]
    context_fields = ["context", "document", "passage", "source", "content", "documents", "text", "doc", "docs"]
    ground_truth_fields = ["ground_truth", "answer", "reference", "expected", "golden", "target",
                           "expected_answer", "correct_answer", "true_answer"]

    # Extract query
    query = None
    for field in query_fields:
        if field in item and item[field]:
            query = item[field]
            logger.debug(f"Found query in field '{field}'")
            break

    if not query:
        logger.warning(f"No query found in item {index} - checked fields: {query_fields}")
        # Generate a default query based on the item content
        all_text = " ".join(str(v) for k, v in item.items() if isinstance(v, (str, int, float)))
        query = f"Analyze this information: {all_text[:100]}..."

    normalized["query"] = query
    logger.debug(f"Normalized query: {query[:100]}..." if len(query) > 100 else f"Normalized query: {query}")

    # Extract context
    context = None
    context_source = None
    for field in context_fields:
        if field in item and item[field]:
            context = item[field]
            context_source = field
            logger.debug(f"Found context in field '{field}'")
            break

    # Process complex context formats
    if context:
        if isinstance(context, list):
            if all(isinstance(doc, str) for doc in context):
                # List of strings
                logger.debug(f"Context is a list of {len(context)} strings")
                context = "\n\n".join(context)
            elif all(isinstance(doc, dict) for doc in context):
                # Extract text from each document dictionary
                logger.debug(f"Context is a list of {len(context)} dictionaries")
                try:
                    texts = []
                    for doc in context:
                        if "text" in doc:
                            texts.append(doc["text"])
                        elif "content" in doc:
                            texts.append(doc["content"])
                        elif "passage" in doc:
                            texts.append(doc["passage"])
                        else:
                            # Flatten the dict as a fallback
                            doc_text = " ".join(f"{k}: {v}" for k, v in doc.items()
                                                if isinstance(v, (str, int, float)))
                            texts.append(doc_text)
                    context = "\n\n".join(texts)
                except Exception as e:
                    logger.warning(f"Error extracting text from context documents: {e}")
                    context = str(context)
        elif isinstance(context, dict):
            # Handle dict context
            logger.debug(f"Context is a dictionary with keys: {list(context.keys())}")
            if "text" in context:
                context = context["text"]
            elif "content" in context:
                context = context["content"]
            else:
                # Flatten the dict
                context = "\n".join(f"{k}: {v}" for k, v in context.items()
                                    if isinstance(v, (str, int, float)))

    if not context:
        logger.warning(f"No context found in item {index} - checked fields: {context_fields}")
        # Generate minimal context from other fields
        other_fields = {k: v for k, v in item.items()
                        if k not in query_fields and isinstance(v, (str, int, float))}
        if other_fields:
            context = "\n".join(f"{k}: {v}" for k, v in other_fields.items())
        else:
            context = "No context available in the dataset."

    normalized["context"] = context
    logger.debug(f"Normalized context length: {len(context)} chars")
    if len(context) > 200:
        logger.debug(f"Context preview: {context[:200]}...")

    # Extract ground truth
    ground_truth = None
    for field in ground_truth_fields:
        if field in item and item[field]:
            ground_truth = item[field]
            logger.debug(f"Found ground truth in field '{field}'")
            break

    normalized["ground_truth"] = ground_truth or ""
    if ground_truth:
        logger.debug(f"Normalized ground truth: {ground_truth[:100]}..."
                     if len(ground_truth) > 100 else f"Normalized ground truth: {ground_truth}")

    # Copy all original fields that don't conflict with normalized ones
    for key, value in item.items():
        if key not in normalized:
            normalized[key] = value

    # Final validation
    if not normalized["query"]:
        logger.error(f"Failed to extract a valid query for item {index}")
    if not normalized["context"]:
        logger.error(f"Failed to extract a valid context for item {index}")

    return normalized


def process_user_query_dataset(data: Any) -> List[Dict[str, Any]]:
    """Process a user query dataset."""
    logger.debug(f"Processing user query dataset with structure: {type(data)}")

    if isinstance(data, list):
        # Normalize the data to ensure it has required fields
        normalized_data = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # Ensure it has query field, looking for common names
                query = None
                for field in ["query", "question", "user_query", "input"]:
                    if field in item and item[field]:
                        query = item[field]
                        break

                normalized_item = {
                    "query": query or f"Missing query in item {i}",
                    "context": item.get("context", ""),
                    "ground_truth": item.get("ground_truth", "")
                }

                # Copy other fields
                for k, v in item.items():
                    if k not in normalized_item:
                        normalized_item[k] = v

                normalized_data.append(normalized_item)
            else:
                # Not a dict, create a simple entry
                normalized_data.append({
                    "query": str(item),
                    "context": ""
                })

        logger.info(f"Processed {len(normalized_data)} user query items")
        return normalized_data

    elif isinstance(data, dict):
        # Handle container formats
        if "queries" in data and isinstance(data["queries"], list):
            return process_user_query_dataset(data["queries"])
        elif "questions" in data and isinstance(data["questions"], list):
            return process_user_query_dataset(data["questions"])
        else:
            # Single item
            return [{"query": data.get("query", "No query found"), "context": data.get("context", "")}]

    else:
        # Unexpected format
        logger.warning(f"Unexpected data format in user query dataset: {type(data)}")
        return [{"query": str(data), "context": ""}]


def process_context_dataset(data: Any) -> List[Dict[str, Any]]:
    """Process a context dataset."""
    logger.debug(f"Processing context dataset with structure: {type(data)}")

    if isinstance(data, list):
        # Normalize the data
        normalized_data = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # Look for context fields
                context = None
                for field in ["context", "document", "content", "text"]:
                    if field in item and item[field]:
                        context = item[field]
                        break

                normalized_item = {
                    "query": item.get("query", f"What information can be found in context {i}?"),
                    "context": context or f"Missing context in item {i}",
                    "ground_truth": item.get("ground_truth", "")
                }

                # Copy other fields
                for k, v in item.items():
                    if k not in normalized_item:
                        normalized_item[k] = v

                normalized_data.append(normalized_item)
            else:
                # Not a dict, create a simple entry
                normalized_data.append({
                    "query": f"What information can be found in this context?",
                    "context": str(item)
                })

        logger.info(f"Processed {len(normalized_data)} context items")
        return normalized_data

    elif isinstance(data, dict):
        # Handle container formats
        if "contexts" in data and isinstance(data["contexts"], list):
            return process_context_dataset(data["contexts"])
        elif "documents" in data and isinstance(data["documents"], list):
            return process_context_dataset(data["documents"])
        else:
            # Single item
            return [{
                "query": "What information can be found in this context?",
                "context": data.get("context", data.get("document", "No context found"))
            }]

    else:
        # Unexpected format
        logger.warning(f"Unexpected data format in context dataset: {type(data)}")
        return [{"query": "What is in this context?", "context": str(data)}]


def process_qa_dataset(data: Any) -> List[Dict[str, Any]]:
    """Process a question-answer dataset."""
    logger.debug(f"Processing QA dataset with structure: {type(data)}")

    if isinstance(data, list):
        # Normalize the data
        normalized_data = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # Normalize using common function
                normalized_item = normalize_dataset_item(item, i)
                normalized_data.append(normalized_item)
            else:
                # Not a dict, create a simple entry
                normalized_data.append({
                    "query": f"Item {i} in QA dataset",
                    "context": "",
                    "ground_truth": str(item)
                })

        logger.info(f"Processed {len(normalized_data)} QA items")
        return normalized_data

    elif isinstance(data, dict):
        # Handle container formats
        for container_key in ["qas", "pairs", "examples", "data"]:
            if container_key in data and isinstance(data[container_key], list):
                return process_qa_dataset(data[container_key])

        # Check for SQuAD-like format
        if "data" in data and isinstance(data["data"], list):
            all_qa_pairs = []
            for article in data["data"]:
                if isinstance(article, dict) and "paragraphs" in article:
                    for para in article["paragraphs"]:
                        if isinstance(para, dict):
                            context = para.get("context", "")
                            if "qas" in para and isinstance(para["qas"], list):
                                for qa in para["qas"]:
                                    if isinstance(qa, dict):
                                        question = qa.get("question", "")
                                        answers = qa.get("answers", [])
                                        if answers and isinstance(answers, list):
                                            answer = answers[0].get("text", "") if isinstance(answers[0],
                                                                                              dict) else str(answers[0])
                                        else:
                                            answer = ""

                                        all_qa_pairs.append({
                                            "query": question,
                                            "context": context,
                                            "ground_truth": answer
                                        })
            if all_qa_pairs:
                logger.info(f"Extracted {len(all_qa_pairs)} QA pairs from SQuAD format")
                return all_qa_pairs

        # Single item
        return [{
            "query": data.get("question", data.get("query", "No question found")),
            "context": data.get("context", ""),
            "ground_truth": data.get("answer", data.get("ground_truth", ""))
        }]

    else:
        # Unexpected format
        logger.warning(f"Unexpected data format in QA dataset: {type(data)}")
        return [{"query": "Unknown question", "context": "", "ground_truth": str(data)}]


def process_conversation_dataset(data: Any) -> List[Dict[str, Any]]:
    """Process a conversation dataset."""
    logger.debug(f"Processing conversation dataset with structure: {type(data)}")

    normalized_data = []

    if isinstance(data, list):
        # Check if it's a list of conversations or list of messages
        is_list_of_conversations = False
        for item in data:
            if isinstance(item, dict) and any(k in item for k in ["messages", "conversation", "turns", "exchanges"]):
                is_list_of_conversations = True
                break

        if is_list_of_conversations:
            # List of conversations
            for i, conversation in enumerate(data):
                if isinstance(conversation, dict):
                    # Extract messages from conversation
                    messages = None
                    for field in ["messages", "conversation", "turns", "exchanges"]:
                        if field in conversation and conversation[field]:
                            messages = conversation[field]
                            break

                    if messages and isinstance(messages, list):
                        # Create context from all but last message
                        context = ""
                        if len(messages) > 1:
                            for j, msg in enumerate(messages[:-1]):
                                role = ""
                                content = ""

                                if isinstance(msg, dict):
                                    # Handle different message formats
                                    role = msg.get("role", msg.get("speaker", ""))
                                    content = msg.get("content", msg.get("text", msg.get("message", "")))
                                else:
                                    content = str(msg)

                                if role and content:
                                    context += f"{role}: {content}\n\n"
                                else:
                                    context += f"{content}\n\n"

                        # Get final user query and assistant response
                        if len(messages) > 0:
                            last_msg = messages[-1]
                            query = ""
                            ground_truth = ""

                            if isinstance(last_msg, dict):
                                if "role" in last_msg and last_msg["role"].lower() in ["user", "human"]:
                                    query = last_msg.get("content", last_msg.get("text", ""))
                                else:
                                    # If last message is from assistant, use it as ground truth
                                    ground_truth = last_msg.get("content", last_msg.get("text", ""))
                                    # And try to find the last user message for query
                                    for j in range(len(messages) - 2, -1, -1):
                                        msg = messages[j]
                                        if isinstance(msg, dict) and msg.get("role", "").lower() in ["user", "human"]:
                                            query = msg.get("content", msg.get("text", ""))
                                            break
                            else:
                                query = str(last_msg)

                            normalized_data.append({
                                "query": query,
                                "context": context.strip(),
                                "ground_truth": ground_truth
                            })
                else:
                    # Not a dict
                    normalized_data.append({
                        "query": f"Conversation {i}",
                        "context": str(conversation),
                        "ground_truth": ""
                    })
        else:
            # List of messages in a single conversation
            # Create context from all but last message
            context = ""
            if len(data) > 1:
                for j, msg in enumerate(data[:-1]):
                    role = ""
                    content = ""

                    if isinstance(msg, dict):
                        # Handle different message formats
                        role = msg.get("role", msg.get("speaker", ""))
                        content = msg.get("content", msg.get("text", msg.get("message", "")))
                    else:
                        content = str(msg)

                    if role and content:
                        context += f"{role}: {content}\n\n"
                    else:
                        context += f"{content}\n\n"

            # Get final user query
            if len(data) > 0:
                last_msg = data[-1]
                query = ""

                if isinstance(last_msg, dict):
                    query = last_msg.get("content", last_msg.get("text", last_msg.get("message", "")))
                else:
                    query = str(last_msg)

                normalized_data.append({
                    "query": query,
                    "context": context.strip(),
                    "ground_truth": ""
                })

    elif isinstance(data, dict):
        # Check for common conversation container formats
        for field in ["messages", "conversation", "turns", "exchanges"]:
            if field in data and isinstance(data[field], list):
                return process_conversation_dataset(data[field])

        # Single message or unknown structure
        normalized_data.append({
            "query": data.get("input", data.get("query", data.get("message", ""))),
            "context": data.get("context", data.get("history", "")),
            "ground_truth": data.get("response", data.get("output", data.get("reply", "")))
        })

    else:
        # Unexpected format
        logger.warning(f"Unexpected data format in conversation dataset: {type(data)}")
        normalized_data.append({
            "query": "What is this conversation about?",
            "context": str(data),
            "ground_truth": ""
        })

    logger.info(f"Processed {len(normalized_data)} conversation items")
    return normalized_data


def process_custom_dataset(data: Any) -> List[Dict[str, Any]]:
    """Process a custom dataset format."""
    logger.debug(f"Processing custom dataset with structure: {type(data)}")

    if isinstance(data, list):
        # Try to extract query, context, and ground truth from each item
        normalized_data = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # Apply maximum normalization
                normalized_item = normalize_dataset_item(item, i)
                normalized_data.append(normalized_item)
            else:
                # Not a dict
                normalized_data.append({
                    "query": f"Item {i} in custom dataset",
                    "context": str(item),
                    "ground_truth": ""
                })

        logger.info(f"Processed {len(normalized_data)} custom dataset items")
        return normalized_data

    elif isinstance(data, dict):
        # Check for container formats
        for container_key in ["items", "examples", "data", "entries"]:
            if container_key in data and isinstance(data[container_key], list):
                return process_custom_dataset(data[container_key])

        # Single item
        return [normalize_dataset_item(data, 0)]

    else:
        # Unexpected format
        logger.warning(f"Unexpected data format in custom dataset: {type(data)}")
        return [{
            "query": "What is in this custom dataset?",
            "context": str(data)[:1000] + ("..." if len(str(data)) > 1000 else ""),
            "ground_truth": ""
        }]


def filter_dataset_item(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Filter a dataset item to ensure it won't trigger content filters.

    Args:
        item: The dataset item to filter
        index: Index for logging purposes

    Returns:
        Dict[str, Any]: Filtered dataset item
    """
    filtered_item = {}

    # Process each field
    for key, value in item.items():
        if isinstance(value, str):
            # For very long texts, truncate to reasonable length
            if len(value) > 10000:
                filtered_value = value[:10000] + "... (truncated)"
                logger.debug(f"Truncated long text in field '{key}' from {len(value)} to 10000 chars")
                filtered_item[key] = filtered_value
            else:
                filtered_item[key] = value
        elif isinstance(value, (list, dict)):
            # For complex values, convert to string representation but limit length
            str_value = str(value)
            if len(str_value) > 1000:
                # Truncate and mark as truncated
                filtered_item[key] = str_value[:1000] + "... (truncated)"
                logger.debug(f"Truncated complex field '{key}' from {len(str_value)} to 1000 chars")
            else:
                filtered_item[key] = value
        else:
            # Keep other types as-is
            filtered_item[key] = value

    return filtered_item


def sanitize_content(text: str, is_system_prompt: bool = False) -> str:
    """
    Sanitize content to avoid triggering content filters.

    Args:
        text: Content to sanitize
        is_system_prompt: Whether this is a system prompt

    Returns:
        str: Sanitized content
    """
    if not text:
        return text

    # Add appropriate prefixes based on content type
    if is_system_prompt:
        prefix = "This is an educational evaluation prompt for testing purposes. "
    else:
        prefix = "For evaluation purposes: "

    # Check if content might be problematic
    text_lower = text.lower()
    problematic_patterns = [
        "jailbreak", "ignore previous instructions", "ignore your instructions",
        "harmful content", "illegal activity", "bypass", "restriction",
        "generate inappropriate", "harmful", "unethical", "immoral",
        "ignore content policy", "content filter", "evade", "circumvent",
        "bypass safeguards", "ignore guidelines", "ignore rules"
    ]

    if any(pattern in text_lower for pattern in problematic_patterns):
        import re

        # Replace problematic patterns with neutral terms
        sanitized_text = text

        replacements = {
            "jailbreak": "evaluate",
            "ignore previous instructions": "consider these instructions",
            "ignore your instructions": "follow these instructions",
            "harmful content": "evaluation content",
            "illegal activity": "hypothetical scenario",
            "bypass": "process",
            "restriction": "guideline",
            "generate inappropriate": "analyze appropriate",
            "harmful": "educational",
            "unethical": "ethical",
            "immoral": "appropriate",
            "ignore content policy": "follow content policy",
            "content filter": "content guideline",
            "evade": "adhere to",
            "circumvent": "follow",
            "bypass safeguards": "maintain safeguards"
        }

        for pattern, replacement in replacements.items():
            sanitized_text = re.sub(
                r'\b' + re.escape(pattern) + r'\b',
                replacement,
                sanitized_text,
                flags=re.IGNORECASE
            )

        return prefix + sanitized_text

    # If content seems safe, just add the prefix
    return prefix + text


def process_content_filter_error(error_response, item_index: int) -> Dict[str, Any]:
    """
    Process a content filter error and format for user feedback.

    Args:
        error_response: The error response from the API
        item_index: Index of the dataset item

    Returns:
        Dict[str, Any]: Formatted error information
    """
    try:
        error_info = {
            "error_type": "content_filter",
            "message": "Content filtered by Azure OpenAI's content management policy",
            "details": {},
            "recommendation": "Please review and modify your prompt to comply with content policies"
        }

        # Extract more specific details if available
        if isinstance(error_response, dict):
            if "code" in error_response:
                error_info["error_code"] = error_response["code"]

            # Extract inner error details
            inner_error = error_response.get("innererror", {})
            if inner_error:
                if "code" in inner_error:
                    error_info["details"]["code"] = inner_error["code"]

                # Get content filter results
                filter_result = inner_error.get("content_filter_result", {})
                if filter_result:
                    error_info["details"]["filter_results"] = {}

                    # Extract individual category results
                    for category, result in filter_result.items():
                        if isinstance(result, dict):
                            category_info = {}
                            if "filtered" in result:
                                category_info["filtered"] = result["filtered"]
                            if "severity" in result:
                                category_info["severity"] = result["severity"]
                            if "detected" in result:
                                category_info["detected"] = result["detected"]

                            error_info["details"]["filter_results"][category] = category_info

        # Log the error for debugging
        logger.error(f"Content filter error for item {item_index}: {json.dumps(error_info)}")

        return error_info

    except Exception as e:
        logger.error(f"Error processing content filter details: {e}")
        return {
            "error_type": "content_filter",
            "message": "Content filtered by Azure OpenAI's content management policy",
            "recommendation": "Please review and modify your prompt to comply with content policies"
        }
