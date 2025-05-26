import logging
from typing import Any, Dict, List, Optional

from jinja2 import Template, TemplateError

from backend.app.db.models.orm import Prompt
from backend.app.services.agent_clients.base import AgentClient

# Configure logging
logger = logging.getLogger(__name__)

# Template variable mappings
INPUT_VARIABLES = ["user_input", "user_query", "query", "question", "input"]
CONTEXT_VARIABLES = ["context", "contexts", "supporting_context"]
EVALUATION_KEYWORDS = ["accurate", "precise", "relevant", "helpful", "clear"]


class PromptAdapter:
    """Integrates prompt system with DeepEval evaluations."""

    def __init__(self):
        """Initialize the prompt adapter."""
        pass

    async def apply_prompt_to_agent_client(
            self,
            agent_client: AgentClient,
            prompt: Prompt,
            test_input: str,
            context: Optional[List[str]] = None,
    ) -> str:
        """
        Apply prompt template to generate agent response for DeepEval.

        Args:
            agent_client: Agent client to use for generation
            prompt: Prompt template to apply
            test_input: Input text for the prompt
            context: Optional context list

        Returns:
            str: Generated response from the agent

        Raises:
            Exception: If prompt application fails
        """
        try:
            rendered_prompt = await self._render_prompt_template(
                prompt, test_input, context or []
            )

            response = await agent_client.process_query(
                query=test_input,
                context="\n".join(context) if context else None,
                system_message=rendered_prompt,
            )

            return self._extract_response_text(response)

        except Exception as e:
            logger.error(f"Error applying prompt template: {e}")
            return f"Error applying prompt: {str(e)}"

    def _extract_response_text(self, response: Any) -> str:
        """Extract text response from agent client response."""
        if isinstance(response, dict):
            if not response.get("success", True):
                error = response.get("error", "Unknown error")
                logger.warning(f"Agent client returned error: {error}")
                return f"Error: {error}"
            return response.get("answer", "")

        return str(response)

    async def _render_prompt_template(
            self, prompt: Prompt, user_input: str, context: List[str]
    ) -> str:
        """
        Render prompt template with actual values.

        Args:
            prompt: Prompt object containing template and parameters
            user_input: User input text
            context: List of context strings

        Returns:
            str: Rendered prompt template
        """
        try:
            template_vars = self._build_template_variables(
                user_input, context, prompt.parameters or {}
            )

            # Use Jinja2 for template rendering
            template = Template(prompt.content)
            rendered = template.render(**template_vars)

            logger.debug(
                f"Rendered prompt template. Original length: {len(prompt.content)}, "
                f"Rendered length: {len(rendered)}"
            )

            return rendered

        except TemplateError as e:
            logger.error(f"Jinja2 template error: {e}")
            return self._simple_template_render(
                prompt.content, user_input, context, prompt.parameters or {}
            )
        except Exception as e:
            logger.error(f"Error rendering prompt template: {e}")
            return self._simple_template_render(
                prompt.content, user_input, context, prompt.parameters or {}
            )

    def _build_template_variables(
            self, user_input: str, context: List[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build template variables dictionary."""
        context_str = "\n".join(context) if context else ""

        return {
            # Input aliases
            "user_input": user_input,
            "user_query": user_input,
            "query": user_input,
            "question": user_input,
            "input": user_input,

            # Context handling
            "context": context_str,
            "contexts": context,
            "context_list": context,
            "supporting_context": context_str,

            # Metadata
            "context_count": len(context),
            "has_context": bool(context),

            # Custom parameters
            **parameters,
        }

    def _simple_template_render(
            self,
            template_content: str,
            user_input: str,
            context: List[str],
            parameters: Dict[str, Any],
    ) -> str:
        """Simple template rendering fallback using string replacement."""
        try:
            replacements = self._build_replacement_dict(user_input, context, parameters)

            rendered = template_content
            for placeholder, value in replacements.items():
                rendered = rendered.replace(placeholder, value)

            return rendered

        except Exception as e:
            logger.error(f"Error in simple template rendering: {e}")
            return template_content

    def _build_replacement_dict(
            self, user_input: str, context: List[str], parameters: Dict[str, Any]
    ) -> Dict[str, str]:
        """Build replacement dictionary for simple template rendering."""
        context_str = "\n".join(context) if context else ""

        base_replacements = {
            "{user_input}": user_input,
            "{user_query}": user_input,
            "{query}": user_input,
            "{question}": user_input,
            "{input}": user_input,
            "{context}": context_str,
            "{contexts}": context_str,
            "{supporting_context}": context_str,
        }

        # Add parameter replacements
        param_replacements = {f"{{{k}}}": str(v) for k, v in parameters.items()}

        return {**base_replacements, **param_replacements}

    async def prepare_system_message(
            self, prompt: Prompt, evaluation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Prepare system message with evaluation-specific context.

        Args:
            prompt: Prompt object
            evaluation_context: Optional evaluation context

        Returns:
            str: Prepared system message
        """
        try:
            system_message = prompt.content

            if evaluation_context:
                evaluation_prefix = self._create_evaluation_prefix(evaluation_context)
                system_message = f"{evaluation_prefix}\n\n{system_message}"

            return system_message

        except Exception as e:
            logger.error(f"Error preparing system message: {e}")
            return prompt.content

    def _create_evaluation_prefix(self, evaluation_context: Dict[str, Any]) -> str:
        """Create evaluation-specific prefix for system message."""
        evaluation_type = evaluation_context.get("evaluation_type", "general")
        metrics = evaluation_context.get("metrics", [])

        prefix_parts = [
            "You are being evaluated for response quality.",
            f"Evaluation type: {evaluation_type}",
        ]

        if metrics:
            instructions = self._get_metric_instructions(metrics)
            if instructions:
                prefix_parts.append("Focus on:")
                prefix_parts.extend(instructions)

        return "\n".join(prefix_parts)

    def _get_metric_instructions(self, metrics: List[str]) -> List[str]:
        """Get evaluation instructions for specific metrics."""
        metric_instructions = {
            "faithfulness": "- Stay grounded in the provided context",
            "answer_relevancy": "- Ensure your response directly addresses the question",
            "contextual_precision": "- Use context information precisely and accurately",
            "contextual_recall": "- Include all relevant information from the context",
            "hallucination": "- Do not fabricate information not present in the context",
            "toxicity": "- Maintain respectful and appropriate language",
            "bias": "- Provide balanced and unbiased responses",
        }

        return [
            metric_instructions[metric]
            for metric in metrics
            if metric in metric_instructions
        ]

    def validate_prompt_compatibility(self, prompt: Prompt) -> Dict[str, Any]:
        """
        Validate prompt compatibility with DeepEval.

        Args:
            prompt: Prompt to validate

        Returns:
            Dict containing validation results and suggestions
        """
        validation_result = {
            "compatible": True,
            "warnings": [],
            "suggestions": [],
            "template_variables": [],
        }

        try:
            template_content = prompt.content.lower()

            # Extract template variables
            validation_result["template_variables"] = self._extract_template_variables(
                prompt.content
            )

            # Validate input variables
            self._validate_input_variables(template_content, validation_result)

            # Validate context variables  
            self._validate_context_variables(template_content, validation_result)

            # Validate prompt length
            self._validate_prompt_length(prompt.content, validation_result)

            # Check for evaluation-friendly instructions
            self._validate_evaluation_keywords(template_content, validation_result)

        except Exception as e:
            logger.error(f"Error validating prompt compatibility: {e}")
            validation_result["compatible"] = False
            validation_result["warnings"].append(f"Error during validation: {str(e)}")

        return validation_result

    def _extract_template_variables(self, content: str) -> List[str]:
        """Extract template variables from prompt content."""
        import re
        variables = re.findall(r'\{([^}]+)\}', content)
        return list(set(variables))

    def _validate_input_variables(
            self, template_content: str, validation_result: Dict[str, Any]
    ) -> None:
        """Validate presence of input variables."""
        input_placeholders = [f"{{{var}}}" for var in INPUT_VARIABLES]
        has_input_var = any(var in template_content for var in input_placeholders)

        if not has_input_var:
            validation_result["warnings"].append(
                "Prompt doesn't contain common input variables. "
                "Consider adding {user_input}, {query}, or {question} placeholders."
            )

    def _validate_context_variables(
            self, template_content: str, validation_result: Dict[str, Any]
    ) -> None:
        """Validate presence of context variables."""
        context_placeholders = [f"{{{var}}}" for var in CONTEXT_VARIABLES]
        has_context_var = any(var in template_content for var in context_placeholders)

        if not has_context_var:
            validation_result["suggestions"].append(
                "Consider adding {context} or {contexts} placeholders for RAG evaluations."
            )

    def _validate_prompt_length(
            self, content: str, validation_result: Dict[str, Any]
    ) -> None:
        """Validate prompt length."""
        if len(content) > 4000:
            validation_result["warnings"].append(
                "Prompt is quite long. Consider shortening it to avoid token limits."
            )

    def _validate_evaluation_keywords(
            self, template_content: str, validation_result: Dict[str, Any]
    ) -> None:
        """Validate presence of evaluation-friendly keywords."""
        has_evaluation_keywords = any(
            keyword in template_content for keyword in EVALUATION_KEYWORDS
        )

        if not has_evaluation_keywords:
            validation_result["suggestions"].append(
                "Consider adding instructions for accuracy, relevance, or helpfulness "
                "to improve evaluation scores."
            )

    async def create_evaluation_optimized_prompt(
            self, base_prompt: Prompt, metrics: List[str], dataset_type: str
    ) -> str:
        """
        Create an evaluation-optimized version of the prompt.

        Args:
            base_prompt: Base prompt to optimize
            metrics: List of metrics to optimize for
            dataset_type: Type of dataset being evaluated

        Returns:
            str: Optimized prompt content
        """
        try:
            optimized_content = base_prompt.content

            # Add metric-specific instructions
            metric_instructions = self._get_metric_specific_instructions(metrics)
            if metric_instructions:
                instruction_text = (
                        "\n\nEvaluation Guidelines:\n" + "\n".join(metric_instructions)
                )
                optimized_content += instruction_text

            # Add dataset-specific instructions
            dataset_instructions = self._get_dataset_specific_instructions(dataset_type)
            if dataset_instructions:
                optimized_content += f"\n\nDataset Context: {dataset_instructions}"

            return optimized_content

        except Exception as e:
            logger.error(f"Error creating evaluation-optimized prompt: {e}")
            return base_prompt.content

    def _get_metric_specific_instructions(self, metrics: List[str]) -> List[str]:
        """Get specific instructions for each metric."""
        instruction_map = {
            "faithfulness": (
                "- Base your response strictly on the provided context. "
                "Do not add information not present in the context."
            ),
            "answer_relevancy": (
                "- Ensure your response directly and completely addresses the user's question."
            ),
            "contextual_precision": (
                "- Use context information accurately and precisely. "
                "Avoid misrepresenting context details."
            ),
            "contextual_recall": (
                "- Include all relevant information from the context that helps answer the question."
            ),
            "hallucination": (
                "- Do not fabricate, assume, or infer information beyond what is explicitly provided."
            ),
            "toxicity": (
                "- Maintain professional, respectful, and appropriate language throughout your response."
            ),
            "bias": (
                "- Provide balanced, objective responses without favoring particular viewpoints unfairly."
            ),
            "coherence": (
                "- Structure your response logically with clear connections between ideas."
            ),
            "correctness": (
                "- Ensure factual accuracy in your response based on the available information."
            ),
            "completeness": (
                "- Address all aspects of the question comprehensively."
            ),
        }

        return [instruction_map[metric] for metric in metrics if metric in instruction_map]

    def _get_dataset_specific_instructions(self, dataset_type: str) -> str:
        """Get dataset-specific instructions."""
        instructions = {
            "user_query": "Respond to user queries naturally and helpfully.",
            "question_answer": "Provide accurate, complete answers to the questions asked.",
            "context": "Use the provided context to inform your response appropriately.",
            "conversation": "Maintain conversational flow and context from previous exchanges.",
            "custom": "Follow the specific requirements indicated by the data format.",
        }

        return instructions.get(dataset_type, "Respond appropriately to the given input.")
