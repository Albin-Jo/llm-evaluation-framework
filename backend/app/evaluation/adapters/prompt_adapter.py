import logging
from typing import Dict, Any, List, Optional

from jinja2 import Template, TemplateError

from backend.app.db.models.orm import Prompt
from backend.app.services.agent_clients.base import AgentClient

# Configure logging
logger = logging.getLogger(__name__)


class PromptAdapter:
    """Integrates your prompt system with DeepEval evaluations."""

    def __init__(self):
        pass

    async def apply_prompt_to_agent_client(
            self,
            agent_client: AgentClient,
            prompt: Prompt,
            test_input: str,
            context: Optional[List[str]] = None
    ) -> str:
        """Apply your prompt template to generate agent response for DeepEval."""

        try:
            # Render your prompt template with parameters
            rendered_prompt = await self._render_prompt_template(
                prompt,
                test_input,
                context or []
            )

            # Use your existing agent client with the rendered prompt
            response = await agent_client.process_query(
                query=test_input,
                context='\n'.join(context) if context else None,
                system_message=rendered_prompt
            )

            # Extract the answer from the response
            if isinstance(response, dict):
                answer = response.get('answer', '')
                success = response.get('success', True)

                if not success:
                    error = response.get('error', 'Unknown error')
                    logger.warning(f"Agent client returned error: {error}")
                    return f"Error: {error}"

                return answer
            else:
                # Handle case where response is a string
                return str(response)

        except Exception as e:
            logger.error(f"Error applying prompt template: {e}")
            return f"Error applying prompt: {str(e)}"

    async def _render_prompt_template(
            self,
            prompt: Prompt,
            user_input: str,
            context: List[str]
    ) -> str:
        """Render prompt template with actual values."""

        try:
            template_content = prompt.content
            parameters = prompt.parameters or {}

            # Standard template variables
            template_vars = {
                'user_input': user_input,
                'user_query': user_input,  # Alias
                'query': user_input,  # Another alias
                'question': user_input,  # Yet another alias
                'input': user_input,  # Generic alias

                # Context handling
                'context': '\n'.join(context) if context else '',
                'contexts': context,
                'context_list': context,
                'supporting_context': '\n'.join(context) if context else '',

                # Metadata
                'context_count': len(context),
                'has_context': bool(context),

                # Include any custom parameters from the prompt
                **parameters
            }

            # Use Jinja2 for template rendering (supports your existing templates)
            template = Template(template_content)
            rendered = template.render(**template_vars)

            logger.debug(f"Rendered prompt template. Original length: {len(template_content)}, "
                         f"Rendered length: {len(rendered)}")

            return rendered

        except TemplateError as e:
            logger.error(f"Jinja2 template error: {e}")
            # Fallback to simple string formatting
            return self._simple_template_render(prompt.content, user_input, context, prompt.parameters or {})

        except Exception as e:
            logger.error(f"Error rendering prompt template: {e}")
            # Return template with basic substitution as fallback
            return self._simple_template_render(prompt.content, user_input, context, prompt.parameters or {})

    def _simple_template_render(
            self,
            template_content: str,
            user_input: str,
            context: List[str],
            parameters: Dict[str, Any]
    ) -> str:
        """Simple template rendering fallback using string replacement."""

        try:
            # Create replacement dictionary
            replacements = {
                '{user_input}': user_input,
                '{user_query}': user_input,
                '{query}': user_input,
                '{question}': user_input,
                '{input}': user_input,
                '{context}': '\n'.join(context) if context else '',
                '{contexts}': '\n'.join(context) if context else '',
                '{supporting_context}': '\n'.join(context) if context else '',
                **{f'{{{k}}}': str(v) for k, v in parameters.items()}
            }

            # Apply replacements
            rendered = template_content
            for placeholder, value in replacements.items():
                rendered = rendered.replace(placeholder, value)

            return rendered

        except Exception as e:
            logger.error(f"Error in simple template rendering: {e}")
            # Ultimate fallback - return original template
            return template_content

    async def prepare_system_message(
            self,
            prompt: Prompt,
            evaluation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Prepare system message with evaluation-specific context."""

        try:
            # Base system message from prompt
            system_message = prompt.content

            # Add evaluation context if provided
            if evaluation_context:
                evaluation_prefix = self._create_evaluation_prefix(evaluation_context)
                system_message = f"{evaluation_prefix}\n\n{system_message}"

            return system_message

        except Exception as e:
            logger.error(f"Error preparing system message: {e}")
            return prompt.content

    def _create_evaluation_prefix(self, evaluation_context: Dict[str, Any]) -> str:
        """Create evaluation-specific prefix for system message."""

        evaluation_type = evaluation_context.get('evaluation_type', 'general')
        metrics = evaluation_context.get('metrics', [])

        prefix_parts = [
            "You are being evaluated for response quality.",
            f"Evaluation type: {evaluation_type}"
        ]

        if metrics:
            metric_descriptions = {
                'faithfulness': 'Stay grounded in the provided context',
                'answer_relevancy': 'Ensure your response directly addresses the question',
                'contextual_precision': 'Use context information precisely and accurately',
                'contextual_recall': 'Include all relevant information from the context',
                'hallucination': 'Do not fabricate information not present in the context',
                'toxicity': 'Maintain respectful and appropriate language',
                'bias': 'Provide balanced and unbiased responses'
            }

            relevant_instructions = []
            for metric in metrics:
                if metric in metric_descriptions:
                    relevant_instructions.append(f"- {metric_descriptions[metric]}")

            if relevant_instructions:
                prefix_parts.append("Focus on:")
                prefix_parts.extend(relevant_instructions)

        return "\n".join(prefix_parts)

    def validate_prompt_compatibility(self, prompt: Prompt) -> Dict[str, Any]:
        """Validate prompt compatibility with DeepEval."""

        validation_result = {
            'compatible': True,
            'warnings': [],
            'suggestions': [],
            'template_variables': []
        }

        try:
            # Check for common template variables
            template_content = prompt.content.lower()

            # Extract potential template variables
            import re
            variables = re.findall(r'\{([^}]+)\}', prompt.content)
            validation_result['template_variables'] = list(set(variables))

            # Check for input variables
            input_vars = ['user_input', 'user_query', 'query', 'question', 'input']
            has_input_var = any(var in template_content for var in [f'{{{var}}}' for var in input_vars])

            if not has_input_var:
                validation_result['warnings'].append(
                    "Prompt doesn't contain common input variables. "
                    "Consider adding {user_input}, {query}, or {question} placeholders."
                )

            # Check for context variables
            context_vars = ['context', 'contexts', 'supporting_context']
            has_context_var = any(var in template_content for var in [f'{{{var}}}' for var in context_vars])

            if not has_context_var:
                validation_result['suggestions'].append(
                    "Consider adding {context} or {contexts} placeholders for RAG evaluations."
                )

            # Check prompt length
            if len(prompt.content) > 4000:
                validation_result['warnings'].append(
                    "Prompt is quite long. Consider shortening it to avoid token limits."
                )

            # Check for evaluation-friendly instructions
            evaluation_keywords = ['accurate', 'precise', 'relevant', 'helpful', 'clear']
            has_evaluation_keywords = any(keyword in template_content for keyword in evaluation_keywords)

            if not has_evaluation_keywords:
                validation_result['suggestions'].append(
                    "Consider adding instructions for accuracy, relevance, or helpfulness "
                    "to improve evaluation scores."
                )

        except Exception as e:
            logger.error(f"Error validating prompt compatibility: {e}")
            validation_result['compatible'] = False
            validation_result['warnings'].append(f"Error during validation: {str(e)}")

        return validation_result

    async def create_evaluation_optimized_prompt(
            self,
            base_prompt: Prompt,
            metrics: List[str],
            dataset_type: str
    ) -> str:
        """Create an evaluation-optimized version of the prompt."""

        try:
            # Start with base prompt
            optimized_content = base_prompt.content

            # Add metric-specific instructions
            metric_instructions = self._get_metric_specific_instructions(metrics)

            if metric_instructions:
                instruction_text = "\n\nEvaluation Guidelines:\n" + "\n".join(metric_instructions)
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

        instructions = []

        metric_instruction_map = {
            'faithfulness': "- Base your response strictly on the provided context. Do not add information not present in the context.",
            'answer_relevancy': "- Ensure your response directly and completely addresses the user's question.",
            'contextual_precision': "- Use context information accurately and precisely. Avoid misrepresenting context details.",
            'contextual_recall': "- Include all relevant information from the context that helps answer the question.",
            'hallucination': "- Do not fabricate, assume, or infer information beyond what is explicitly provided.",
            'toxicity': "- Maintain professional, respectful, and appropriate language throughout your response.",
            'bias': "- Provide balanced, objective responses without favoring particular viewpoints unfairly.",
            'coherence': "- Structure your response logically with clear connections between ideas.",
            'correctness': "- Ensure factual accuracy in your response based on the available information.",
            'completeness': "- Address all aspects of the question comprehensively."
        }

        for metric in metrics:
            if metric in metric_instruction_map:
                instructions.append(metric_instruction_map[metric])

        return instructions

    def _get_dataset_specific_instructions(self, dataset_type: str) -> str:
        """Get dataset-specific instructions."""

        dataset_instructions = {
            'user_query': "Respond to user queries naturally and helpfully.",
            'question_answer': "Provide accurate, complete answers to the questions asked.",
            'context': "Use the provided context to inform your response appropriately.",
            'conversation': "Maintain conversational flow and context from previous exchanges.",
            'custom': "Follow the specific requirements indicated by the data format."
        }

        return dataset_instructions.get(dataset_type, "Respond appropriately to the given input.")
