# File: backend/app/utils/deepeval_sample.py

import json
import logging
import time
import ssl
import warnings
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Disable DeepEval telemetry and analytics (fixes PostHog SSL errors)
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["POSTHOG_OPT_OUT"] = "1"

# DeepEval imports
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    BiasMetric,
    ToxicityMetric,
    HallucinationMetric
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

# LangChain imports for Azure OpenAI
from langchain_openai import AzureChatOpenAI


@dataclass
class EvaluationConfig:
    """Configuration class for evaluation settings"""
    azure_openai_key: str
    azure_openai_endpoint: str
    azure_openai_deployment: str
    azure_openai_version: str
    threshold: float = 0.7
    include_reason: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    output_file: Optional[str] = None
    fix_ssl: bool = True
    quiet_mode: bool = False


class EnhancedLogger:
    """Enhanced logging utility with better error handling"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        if config.enable_logging:
            self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration with third-party library suppression"""
        if self.config.quiet_mode:
            logging.basicConfig(
                level=logging.ERROR,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('deepeval_evaluation.log', encoding='utf-8')
                ]
            )
        else:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('deepeval_evaluation.log', encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )

        self.logger = logging.getLogger(__name__)

        # Suppress verbose third-party loggers
        third_party_loggers = [
            'urllib3.connectionpool',
            'httpx',
            'backoff',
            'posthog',
            'requests.packages.urllib3.connectionpool',
            'deepeval'
        ]

        for logger_name in third_party_loggers:
            logging.getLogger(logger_name).setLevel(logging.ERROR)

        warnings.filterwarnings("ignore")

        if not self.config.quiet_mode:
            self.logger.info("Logging configured successfully")

    def info(self, message: str):
        if self.config.enable_logging and not self.config.quiet_mode:
            self.logger.info(message)

    def error(self, message: str):
        if self.config.enable_logging:
            self.logger.error(message)

    def warning(self, message: str):
        if self.config.enable_logging and not self.config.quiet_mode:
            self.logger.warning(message)


class IntegratedAzureOpenAI(DeepEvalBaseLLM):
    """Integrated Azure OpenAI model that works with your platform's agent system"""

    def __init__(self, config: EvaluationConfig, logger: EnhancedLogger):
        self.config = config
        self.logger = logger
        self.model = None

    def load_model(self):
        """Load the Azure OpenAI model"""
        try:
            if not self.model:
                self.model = AzureChatOpenAI(
                    openai_api_version=self.config.azure_openai_version,
                    azure_deployment=self.config.azure_openai_deployment,
                    azure_endpoint=self.config.azure_openai_endpoint,
                    openai_api_key=self.config.azure_openai_key,
                    request_timeout=60,
                    max_retries=3,
                    temperature=0.0  # Consistent evaluation
                )
                self.logger.info("Successfully loaded Azure OpenAI model")
            return self.model
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def generate(self, prompt: str) -> str:
        """Generate response with error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                model = self.load_model()
                response = model.invoke(prompt)
                return response.content
            except Exception as e:
                self.logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All generation attempts failed: {str(e)}")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    async def a_generate(self, prompt: str) -> str:
        """Async generation with error handling"""
        try:
            model = self.load_model()
            res = await model.ainvoke(prompt)
            return res.content
        except Exception as e:
            self.logger.error(f"Async generation failed: {str(e)}")
            raise

    def get_model_name(self):
        return "Integrated Azure OpenAI Model"


class MetricsFactory:
    """Factory class for creating evaluation metrics with better error handling"""

    def __init__(self, config: EvaluationConfig, model: IntegratedAzureOpenAI, logger: EnhancedLogger):
        self.config = config
        self.model = model
        self.logger = logger

    def create_comprehensive_metrics(self) -> Dict[str, Any]:
        """Create comprehensive metrics that align with your platform's capabilities"""
        metrics = {}

        try:
            # Core metrics that work well with most datasets
            metrics['answer_relevancy'] = {
                'metric': AnswerRelevancyMetric(
                    threshold=self.config.threshold,
                    model=self.model,
                    include_reason=self.config.include_reason
                ),
                'description': 'Measures how relevant the answer is to the given question',
                'category': 'relevance'
            }

            metrics['faithfulness'] = {
                'metric': FaithfulnessMetric(
                    threshold=self.config.threshold,
                    model=self.model,
                    include_reason=self.config.include_reason
                ),
                'description': 'Measures how well the answer sticks to the information in the context',
                'category': 'groundedness'
            }

            # Contextual metrics for RAG evaluation
            metrics['contextual_precision'] = {
                'metric': ContextualPrecisionMetric(
                    threshold=self.config.threshold,
                    model=self.model,
                    include_reason=self.config.include_reason
                ),
                'description': 'Measures whether nodes in retrieval context are relevant to the given input',
                'category': 'retrieval'
            }

            metrics['contextual_recall'] = {
                'metric': ContextualRecallMetric(
                    threshold=self.config.threshold,
                    model=self.model,
                    include_reason=self.config.include_reason
                ),
                'description': 'Measures how much of the expected output can be attributed to the retrieval context',
                'category': 'retrieval'
            }

            metrics['contextual_relevancy'] = {
                'metric': ContextualRelevancyMetric(
                    threshold=self.config.threshold,
                    model=self.model,
                    include_reason=self.config.include_reason
                ),
                'description': 'Measures how relevant the retrieval context is to the given input',
                'category': 'retrieval'
            }

            # Safety metrics
            metrics['hallucination'] = {
                'metric': HallucinationMetric(
                    threshold=0.3,  # Lower threshold for hallucination detection
                    model=self.model
                ),
                'description': 'Detects hallucinated information not present in the context',
                'category': 'safety'
            }

            metrics['bias'] = {
                'metric': BiasMetric(
                    threshold=0.3,
                    model=self.model,
                    include_reason=self.config.include_reason
                ),
                'description': 'Detects bias in model outputs across various dimensions',
                'category': 'safety'
            }

            metrics['toxicity'] = {
                'metric': ToxicityMetric(
                    threshold=0.3,
                    model=self.model,
                    include_reason=self.config.include_reason
                ),
                'description': 'Detects toxic, harmful, or inappropriate content',
                'category': 'safety'
            }

            self.logger.info(f"Successfully created {len(metrics)} evaluation metrics")

        except Exception as e:
            self.logger.error(f"Failed to create metrics: {str(e)}")
            raise

        return metrics

    def get_metrics_for_dataset_type(self, dataset_type: str) -> Dict[str, Any]:
        """Get appropriate metrics for specific dataset types"""
        all_metrics = self.create_comprehensive_metrics()

        # Map dataset types to appropriate metrics
        dataset_metric_map = {
            'user_query': ['answer_relevancy', 'hallucination', 'bias', 'toxicity'],
            'question_answer': ['answer_relevancy', 'faithfulness', 'contextual_precision', 'contextual_recall'],
            'context': ['faithfulness', 'contextual_precision', 'contextual_recall', 'contextual_relevancy'],
            'conversation': ['answer_relevancy', 'bias', 'toxicity'],
            'custom': list(all_metrics.keys())
        }

        selected_metric_names = dataset_metric_map.get(dataset_type, list(all_metrics.keys()))
        return {name: all_metrics[name] for name in selected_metric_names if name in all_metrics}


class EvaluationRunner:
    """Main evaluation runner that integrates with your platform"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = EnhancedLogger(config)

        if config.fix_ssl:
            self._fix_ssl_issues()

        self.azure_model = IntegratedAzureOpenAI(config, self.logger)
        self.metrics_factory = MetricsFactory(config, self.azure_model, self.logger)

    def _fix_ssl_issues(self):
        """Fix SSL certificate issues"""
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            self.logger.info("SSL verification disabled")
        except Exception as e:
            self.logger.warning(f"Could not fix SSL issues: {str(e)}")

    def create_test_cases_from_platform_format(self, dataset_items: List[Dict[str, Any]]) -> List[LLMTestCase]:
        """Convert your platform's dataset format to DeepEval test cases"""
        test_cases = []

        for i, item in enumerate(dataset_items):
            try:
                # Extract fields using your platform's field mapping
                query = item.get("query", item.get("question", item.get("input", "")))
                context = item.get("context", item.get("contexts", []))
                expected_output = item.get("ground_truth", item.get("expected_answer", item.get("answer", "")))

                # Normalize context to list
                if isinstance(context, str):
                    context = [context] if context else []
                elif not isinstance(context, list):
                    context = []

                # Create test case
                test_case = LLMTestCase(
                    input=query,
                    expected_output=expected_output,
                    context=context,
                    # actual_output will be filled by agent response
                    actual_output=None
                )

                test_cases.append(test_case)

            except Exception as e:
                self.logger.error(f"Error creating test case {i}: {e}")
                # Create a minimal test case for error handling
                test_cases.append(LLMTestCase(
                    input=f"Error processing item {i}",
                    expected_output="Error",
                    context=["Error in processing"],
                    actual_output="Processing error occurred"
                ))

        return test_cases

    def simulate_agent_responses(self, test_cases: List[LLMTestCase]) -> List[LLMTestCase]:
        """Simulate agent responses for testing (replace with actual agent calls)"""
        for i, test_case in enumerate(test_cases):
            try:
                # In your real integration, this would call your agent client
                # For now, we'll simulate realistic responses

                if test_case.input and test_case.context:
                    # Simulate a response based on context
                    context_text = " ".join(test_case.context) if test_case.context else ""
                    simulated_response = f"Based on the provided information: {context_text[:100]}... the answer to '{test_case.input}' is derived from the context."
                else:
                    simulated_response = f"This is a simulated response to: {test_case.input}"

                test_case.actual_output = simulated_response

            except Exception as e:
                self.logger.error(f"Error simulating response for test case {i}: {e}")
                test_case.actual_output = f"Error generating response: {str(e)}"

        return test_cases

    def run_evaluation_compatible_with_platform(
            self,
            dataset_items: List[Dict[str, Any]],
            dataset_type: str = "custom",
            custom_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run evaluation in a format compatible with your platform"""

        start_time = time.time()
        self.logger.info(f"Starting evaluation with {len(dataset_items)} items")

        try:
            # Convert to DeepEval format
            test_cases = self.create_test_cases_from_platform_format(dataset_items)

            # Simulate agent responses (replace with actual agent calls)
            test_cases_with_outputs = self.simulate_agent_responses(test_cases)

            # Get appropriate metrics
            if custom_metrics:
                all_metrics = self.metrics_factory.create_comprehensive_metrics()
                metrics_dict = {name: all_metrics[name] for name in custom_metrics if name in all_metrics}
            else:
                metrics_dict = self.metrics_factory.get_metrics_for_dataset_type(dataset_type)

            # Validate test cases against metrics
            validated_metrics = self._validate_test_cases_for_metrics(test_cases_with_outputs, metrics_dict)

            if not validated_metrics:
                raise ValueError("No metrics can be used with the provided test cases")

            # Extract metrics for evaluation
            metrics = [m['metric'] for m in validated_metrics.values()]

            self.logger.info(
                f"Running evaluation with {len(metrics)} metrics on {len(test_cases_with_outputs)} test cases")

            # Run DeepEval evaluation
            results = evaluate(test_cases=test_cases_with_outputs, metrics=metrics)

            evaluation_time = time.time() - start_time

            # Convert to platform-compatible format
            platform_results = self._convert_to_platform_format(
                results, validated_metrics, evaluation_time, dataset_items
            )

            # Save results if requested
            if self.config.output_file:
                self._save_results(platform_results)

            return platform_results

        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise

    def _validate_test_cases_for_metrics(self, test_cases: List[LLMTestCase], metrics_dict: Dict) -> Dict[str, Any]:
        """Validate test cases against metric requirements"""
        validated_metrics = {}

        for metric_name, metric_info in metrics_dict.items():
            can_use_metric = True

            # Check if test cases have required fields for this metric
            for test_case in test_cases:
                if metric_name in ['contextual_precision', 'contextual_recall', 'contextual_relevancy']:
                    if not test_case.context or len(test_case.context) == 0:
                        can_use_metric = False
                        break
                elif metric_name == 'faithfulness':
                    if not test_case.context:
                        can_use_metric = False
                        break
                elif metric_name == 'hallucination':
                    if not test_case.context:
                        can_use_metric = False
                        break

            if can_use_metric:
                validated_metrics[metric_name] = metric_info
                self.logger.info(f"[OK] {metric_name} validated successfully")
            else:
                self.logger.warning(f"[SKIP] Skipping {metric_name} due to missing required fields")

        return validated_metrics

    def _convert_to_platform_format(
            self,
            deepeval_results: Any,
            metrics_dict: Dict,
            evaluation_time: float,
            original_dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convert DeepEval results to your platform's expected format"""

        # Calculate summary statistics
        total_tests = len(deepeval_results.test_results) if hasattr(deepeval_results, 'test_results') else 0
        successful_tests = 0

        # Process individual results
        processed_results = []
        metric_performance = {}

        if hasattr(deepeval_results, 'test_results'):
            for i, test_result in enumerate(deepeval_results.test_results):
                # Convert individual result
                result_dict = {
                    "id": f"result_{i}",
                    "dataset_sample_id": str(i),
                    "input_data": {
                        "query": original_dataset[i].get("query", "") if i < len(original_dataset) else "",
                        "context": original_dataset[i].get("context", "") if i < len(original_dataset) else "",
                        "ground_truth": original_dataset[i].get("ground_truth", "") if i < len(original_dataset) else ""
                    },
                    "output_data": {
                        "answer": test_result.input if hasattr(test_result, 'input') else ""
                    },
                    "overall_score": 0.0,
                    "passed": test_result.success if hasattr(test_result, 'success') else False,
                    "metric_scores": []
                }

                # Process metrics for this result
                total_score = 0
                metric_count = 0

                if hasattr(test_result, 'metrics_data'):
                    for metric_data in test_result.metrics_data:
                        metric_name = metric_data.name if hasattr(metric_data, 'name') else "unknown"
                        metric_score = metric_data.score if hasattr(metric_data, 'score') else 0
                        metric_success = metric_data.success if hasattr(metric_data, 'success') else False

                        # Add to result
                        result_dict["metric_scores"].append({
                            "name": metric_name,
                            "value": metric_score,
                            "success": metric_success,
                            "reason": getattr(metric_data, 'reason', '') if hasattr(metric_data, 'reason') else ''
                        })

                        # Track for summary
                        if metric_name not in metric_performance:
                            metric_performance[metric_name] = {'scores': [], 'successes': 0, 'total': 0}

                        metric_performance[metric_name]['scores'].append(metric_score)
                        metric_performance[metric_name]['total'] += 1
                        if metric_success:
                            metric_performance[metric_name]['successes'] += 1

                        total_score += metric_score
                        metric_count += 1

                # Calculate overall score for this result
                if metric_count > 0:
                    result_dict["overall_score"] = total_score / metric_count

                if result_dict["passed"]:
                    successful_tests += 1

                processed_results.append(result_dict)

        # Calculate performance analysis
        performance_analysis = {}
        for metric_name, perf_data in metric_performance.items():
            total_cases = len(perf_data['scores'])
            avg_score = sum(perf_data['scores']) / total_cases if total_cases > 0 else 0
            success_rate = (perf_data['successes'] / total_cases) * 100 if total_cases > 0 else 0

            performance_analysis[metric_name] = {
                'average_score': round(avg_score, 3),
                'success_rate': round(success_rate, 1),
                'total_cases': total_cases,
                'min_score': min(perf_data['scores']) if perf_data['scores'] else 0,
                'max_score': max(perf_data['scores']) if perf_data['scores'] else 0
            }

        # Generate recommendations
        recommendations = self._generate_recommendations(performance_analysis)

        return {
            'evaluation_summary': {
                'total_test_cases': total_tests,
                'evaluation_time_seconds': round(evaluation_time, 2),
                'overall_success_rate': (successful_tests / total_tests) if total_tests > 0 else 0,
                'metrics_analyzed': len(metrics_dict)
            },
            'metrics_descriptions': {
                name: {
                    'description': info['description'],
                    'category': info.get('category', 'general')
                }
                for name, info in metrics_dict.items()
            },
            'performance_analysis': performance_analysis,
            'recommendations': recommendations,
            'detailed_results': processed_results
        }

    def _generate_recommendations(self, performance_analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on results"""
        recommendations = []

        for metric_name, analysis in performance_analysis.items():
            success_rate = analysis['success_rate']
            avg_score = analysis['average_score']

            if success_rate < 50:
                if 'answer_relevancy' in metric_name:
                    recommendations.append(
                        f"Low {metric_name} ({success_rate:.1f}%). Improve prompt engineering to ensure responses directly address questions."
                    )
                elif 'faithfulness' in metric_name:
                    recommendations.append(
                        f"Low {metric_name} ({success_rate:.1f}%). Enhance context grounding and reduce hallucinations."
                    )
                elif 'bias' in metric_name:
                    recommendations.append(
                        f"Bias detected ({success_rate:.1f}%). Review training data and implement bias mitigation."
                    )
                elif 'toxicity' in metric_name:
                    recommendations.append(
                        f"Toxicity detected ({success_rate:.1f}%). Implement content filtering and safety measures."
                    )
                elif 'contextual' in metric_name:
                    recommendations.append(
                        f"Poor {metric_name} ({success_rate:.1f}%). Improve retrieval system and context relevance."
                    )
            elif success_rate > 80:
                recommendations.append(
                    f"Excellent {metric_name} performance ({success_rate:.1f}%). Current approach is working well."
                )

        if not recommendations:
            recommendations.append("Overall performance looks good across all metrics.")

        return recommendations

    def _save_results(self, results: Dict[str, Any]):
        """Save results to file"""
        try:
            output_path = Path(self.config.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"Results saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")


def main():
    """
    Main execution function demonstrating platform integration
    """
    # Configuration with your Azure OpenAI credentials
    config = EvaluationConfig(
        azure_openai_key="your-azure-openai-key-here",
        azure_openai_endpoint="your-azure-openai-endpoint-here",
        azure_openai_deployment="your-deployment-name",
        azure_openai_version="2024-02-01",
        threshold=0.7,
        include_reason=True,
        enable_logging=True,
        log_level="INFO",
        output_file="platform_evaluation_results.json",
        fix_ssl=True,
        quiet_mode=False
    )

    # Sample dataset in your platform's format
    platform_dataset = [
        {
            "query": "What is the capital of France?",
            "context": "France is a country in Western Europe. Its capital is Paris, which is known as the City of Light.",
            "ground_truth": "The capital of France is Paris.",
            "category": "geography"
        },
        {
            "query": "How does photosynthesis work?",
            "context": "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water.",
            "ground_truth": "Photosynthesis is a process where plants use sunlight, water, and carbon dioxide to create oxygen and energy.",
            "category": "science"
        },
        {
            "query": "What are the benefits of exercise?",
            "context": "Regular physical activity improves cardiovascular health, strengthens muscles, and enhances mental well-being.",
            "ground_truth": "Exercise improves cardiovascular health, strengthens muscles, and enhances mental well-being.",
            "category": "health"
        }
    ]

    try:
        runner = EvaluationRunner(config)

        print("\nStarting Platform-Compatible DeepEval Evaluation...")
        print("=" * 60)

        # Run evaluation
        results = runner.run_evaluation_compatible_with_platform(
            dataset_items=platform_dataset,
            dataset_type="question_answer",
            custom_metrics=["answer_relevancy", "faithfulness", "contextual_precision"]
        )

        # Display results in platform format
        print("\n" + "=" * 80)
        print("PLATFORM-COMPATIBLE DEEPEVAL EVALUATION RESULTS")
        print("=" * 80)

        summary = results['evaluation_summary']
        print(f"Total Test Cases: {summary['total_test_cases']}")
        print(f"Evaluation Time: {summary['evaluation_time_seconds']:.2f} seconds")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.1%}")
        print(f"Metrics Analyzed: {summary['metrics_analyzed']}")

        print(f"\nPERFORMANCE ANALYSIS:")
        for metric_name, analysis in results['performance_analysis'].items():
            status = "[PASS]" if analysis['success_rate'] > 70 else "[WARN]" if analysis[
                                                                                    'success_rate'] > 30 else "[FAIL]"
            print(
                f"{status} {metric_name}: {analysis['average_score']:.3f} avg, {analysis['success_rate']:.1f}% success")

        print(f"\nRECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"  • {rec}")

        print(f"\nDetailed results saved to: {config.output_file}")
        print("=" * 80)
        print("Evaluation completed successfully!")

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure Azure OpenAI credentials are correct")
        print("2. Check internet connectivity")
        print("3. Verify DeepEval installation")
        raise


if __name__ == "__main__":
    main()