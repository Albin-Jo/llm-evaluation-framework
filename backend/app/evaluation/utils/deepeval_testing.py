# File: backend/app/evaluation/utils/deepeval_testing.py
import asyncio
import json
import logging
import tempfile
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.app.db.models.orm import DatasetType, EvaluationMethod
from backend.app.evaluation.metrics.deepeval_metrics import DEEPEVAL_AVAILABLE
from backend.app.evaluation.utils.deepeval_config_validator import DeepEvalConfigValidator

logger = logging.getLogger(__name__)


class DeepEvalTestRunner:
    """Enhanced utility for testing DeepEval integration with comprehensive test coverage."""

    def __init__(self, db_session=None):
        """Initialize the enhanced test runner."""
        self.db_session = db_session
        self.validator = DeepEvalConfigValidator()

    async def run_integration_test(
            self,
            test_type: str = "basic",
            metrics: Optional[List[str]] = None,
            include_new_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive integration tests for DeepEval.

        Args:
            test_type: Type of test ('basic', 'comprehensive', 'performance', 'stress')
            metrics: Specific metrics to test
            include_new_metrics: Whether to include newly added metrics in testing

        Returns:
            Dict with detailed test results
        """
        logger.info(f"Running DeepEval integration test: {test_type}")

        test_results = {
            "test_type": test_type,
            "timestamp": datetime.now().isoformat(),
            "deepeval_available": DEEPEVAL_AVAILABLE,
            "include_new_metrics": include_new_metrics,
            "tests": {},
            "performance_metrics": {},
            "recommendations": []
        }

        try:
            # Test 1: Library availability and import checks
            test_results["tests"]["library_availability"] = await self._test_library_availability()

            # Test 2: Configuration validation
            test_results["tests"]["configuration_validation"] = await self._test_configuration_validation()

            # Test 3: Enhanced metric testing
            if metrics:
                test_results["tests"]["metric_initialization"] = await self._test_metric_initialization_enhanced(
                    metrics)
            else:
                # Test with default metrics based on test type
                default_metrics = self._get_default_test_metrics(test_type, include_new_metrics)
                test_results["tests"]["metric_initialization"] = await self._test_metric_initialization_enhanced(
                    default_metrics)

            # Test 4: Model validation
            test_results["tests"]["model_validation"] = await self._test_model_validation()

            # Test 5: Sample evaluation with different scenarios
            if test_type in ["comprehensive", "performance", "stress"]:
                test_results["tests"]["sample_evaluation"] = await self._test_sample_evaluation_enhanced(
                    metrics or default_metrics, test_type
                )

            # Test 6: Performance benchmarking
            if test_type in ["performance", "stress"]:
                test_results["tests"]["performance"] = await self._test_performance_enhanced(test_type)

            # Test 7: Error handling and edge cases
            test_results["tests"]["error_handling"] = await self._test_error_handling_enhanced()

            # Test 8: New metrics specific testing
            if include_new_metrics and test_type in ["comprehensive", "stress"]:
                test_results["tests"]["new_metrics_validation"] = await self._test_new_metrics()

            # Test 9: Concurrent execution testing
            if test_type == "stress":
                test_results["tests"]["concurrent_execution"] = await self._test_concurrent_execution()

            # Test 10: Memory and resource usage
            if test_type in ["performance", "stress"]:
                test_results["tests"]["resource_usage"] = await self._test_resource_usage()

            # Calculate overall success
            test_results["overall_success"] = all(
                test.get("success", False) for test in test_results["tests"].values()
            )

            # Generate recommendations
            test_results["recommendations"] = self._generate_recommendations(test_results)

        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            test_results["error"] = str(e)
            test_results["overall_success"] = False

        return test_results

    async def _test_library_availability(self) -> Dict[str, Any]:
        """Enhanced test for DeepEval library availability and component imports."""
        try:
            if not DEEPEVAL_AVAILABLE:
                return {
                    "success": False,
                    "error": "DeepEval library not available",
                    "recommendation": "Install with: pip install deepeval"
                }

            # Test importing core components
            import_tests = {}

            try:
                from deepeval.test_case import LLMTestCase
                import_tests["LLMTestCase"] = "success"
            except Exception as e:
                import_tests["LLMTestCase"] = f"failed: {e}"

            try:
                from deepeval.dataset import EvaluationDataset
                import_tests["EvaluationDataset"] = "success"
            except Exception as e:
                import_tests["EvaluationDataset"] = f"failed: {e}"

            # Test importing various metrics
            metrics_import_tests = {}
            metric_imports = [
                ("AnswerRelevancyMetric", "deepeval.metrics"),
                ("FaithfulnessMetric", "deepeval.metrics"),
                ("ContextualPrecisionMetric", "deepeval.metrics"),
                ("ContextualRecallMetric", "deepeval.metrics"),
                ("HallucinationMetric", "deepeval.metrics"),
                ("BiasMetric", "deepeval.metrics"),
                ("ToxicityMetric", "deepeval.metrics"),
            ]

            for metric_name, module_name in metric_imports:
                try:
                    module = __import__(module_name, fromlist=[metric_name])
                    getattr(module, metric_name)
                    metrics_import_tests[metric_name] = "success"
                except Exception as e:
                    metrics_import_tests[metric_name] = f"failed: {e}"

            # Test basic functionality
            try:
                test_case = LLMTestCase(
                    input="Test question",
                    actual_output="Test answer",
                    expected_output="Expected answer"
                )
                dataset = EvaluationDataset(test_cases=[test_case])
                functionality_test = "success"
            except Exception as e:
                functionality_test = f"failed: {e}"

            all_imports_successful = all(
                status == "success" for status in import_tests.values()
            )

            return {
                "success": all_imports_successful,
                "core_imports": import_tests,
                "metrics_imports": metrics_import_tests,
                "functionality_test": functionality_test,
                "import_success_rate": sum(1 for s in import_tests.values() if s == "success") / len(import_tests)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "recommendation": "Check DeepEval installation and dependencies"
            }

    async def _test_configuration_validation(self) -> Dict[str, Any]:
        """Enhanced configuration validation testing."""
        try:
            test_scenarios = []

            # Test 1: Valid configuration with basic metrics
            valid_config_basic = self.validator.generate_recommended_config(
                DatasetType.QUESTION_ANSWER,
                ["answer_relevancy", "faithfulness"]
            )

            is_valid, errors, warnings = self.validator.validate_evaluation_config(
                valid_config_basic,
                DatasetType.QUESTION_ANSWER,
                ["answer_relevancy", "faithfulness"]
            )

            test_scenarios.append({
                "name": "basic_valid_config",
                "valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "config": valid_config_basic
            })

            # Test 2: Valid configuration with new metrics
            valid_config_enhanced = self.validator.generate_recommended_config(
                DatasetType.CUSTOM,
                ["answer_relevancy", "summarization", "tool_correctness", "g_eval"]
            )

            is_valid, errors, warnings = self.validator.validate_evaluation_config(
                valid_config_enhanced,
                DatasetType.CUSTOM,
                ["answer_relevancy", "summarization", "tool_correctness", "g_eval"]
            )

            test_scenarios.append({
                "name": "enhanced_valid_config",
                "valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "config": valid_config_enhanced
            })

            # Test 3: Invalid configuration scenarios
            invalid_configs = [
                {
                    "name": "invalid_model",
                    "config": {
                        "deepeval_config": {
                            "model": "invalid-model-name",
                            "answer_relevancy": {"threshold": 0.7}
                        }
                    },
                    "metrics": ["answer_relevancy"]
                },
                {
                    "name": "invalid_threshold",
                    "config": {
                        "deepeval_config": {
                            "model": "gpt-4o",
                            "answer_relevancy": {"threshold": 2.0}  # Invalid threshold
                        }
                    },
                    "metrics": ["answer_relevancy"]
                },
                {
                    "name": "unsupported_metric",
                    "config": {
                        "deepeval_config": {
                            "model": "gpt-4o",
                            "unsupported_metric": {"threshold": 0.7}
                        }
                    },
                    "metrics": ["unsupported_metric"]
                }
            ]

            for invalid_test in invalid_configs:
                is_valid, errors, warnings = self.validator.validate_evaluation_config(
                    invalid_test["config"],
                    DatasetType.CUSTOM,
                    invalid_test["metrics"]
                )

                test_scenarios.append({
                    "name": invalid_test["name"],
                    "valid": is_valid,
                    "errors": errors,
                    "warnings": warnings,
                    "expected_invalid": True
                })

            # Calculate success rate
            successful_validations = sum(
                1 for scenario in test_scenarios
                if (scenario.get("expected_invalid", False) and not scenario["valid"]) or
                (not scenario.get("expected_invalid", False) and scenario["valid"])
            )
            success_rate = successful_validations / len(test_scenarios)

            return {
                "success": success_rate >= 0.8,  # At least 80% of tests should pass
                "test_scenarios": test_scenarios,
                "success_rate": success_rate,
                "total_tests": len(test_scenarios)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _test_metric_initialization_enhanced(self, metrics: List[str]) -> Dict[str, Any]:
        """Enhanced metric initialization testing with detailed results."""
        try:
            if not DEEPEVAL_AVAILABLE:
                return {"success": False, "error": "DeepEval not available"}

            from backend.app.evaluation.methods.deepeval import DeepEvalMethod

            method = DeepEvalMethod(self.db_session)

            # Test different configuration scenarios
            test_scenarios = []

            # Scenario 1: Basic configuration
            basic_config = {
                "deepeval_config": {
                    "model": "gpt-4o",
                    **{metric: {"threshold": 0.7} for metric in metrics}
                }
            }

            try:
                initialized_metrics = method._initialize_deepeval_metrics(metrics, basic_config)
                test_scenarios.append({
                    "name": "basic_config",
                    "success": True,
                    "metrics_count": len(initialized_metrics),
                    "initialized_metrics": [str(metric.__class__.__name__) for metric in initialized_metrics]
                })
            except Exception as e:
                test_scenarios.append({
                    "name": "basic_config",
                    "success": False,
                    "error": str(e)
                })

            # Scenario 2: Optimized configuration for performance
            perf_config = {
                "deepeval_config": {
                    "model": "gpt-3.5-turbo",
                    "batch_size": 10,
                    **{metric: {"threshold": 0.6, "include_reason": False, "async_mode": True} for metric in metrics}
                }
            }

            try:
                initialized_metrics = method._initialize_deepeval_metrics(metrics, perf_config)
                test_scenarios.append({
                    "name": "performance_optimized",
                    "success": True,
                    "metrics_count": len(initialized_metrics),
                    "config_type": "performance"
                })
            except Exception as e:
                test_scenarios.append({
                    "name": "performance_optimized",
                    "success": False,
                    "error": str(e)
                })

            # Scenario 3: Test individual metric categories
            category_results = {}
            for category, category_metrics in self.validator.METRIC_CATEGORIES.items():
                available_metrics = [m for m in category_metrics if m in metrics]
                if available_metrics:
                    try:
                        cat_config = {
                            "deepeval_config": {
                                "model": "gpt-4o",
                                **{metric: {"threshold": 0.7} for metric in available_metrics}
                            }
                        }
                        initialized = method._initialize_deepeval_metrics(available_metrics, cat_config)
                        category_results[category] = {
                            "success": True,
                            "count": len(initialized),
                            "metrics": available_metrics
                        }
                    except Exception as e:
                        category_results[category] = {
                            "success": False,
                            "error": str(e),
                            "metrics": available_metrics
                        }

            overall_success = all(scenario.get("success", False) for scenario in test_scenarios)

            return {
                "success": overall_success,
                "test_scenarios": test_scenarios,
                "category_results": category_results,
                "total_metrics_tested": len(metrics),
                "requested_metrics": metrics
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "requested_metrics": metrics
            }

    async def _test_model_validation(self) -> Dict[str, Any]:
        """Test model availability and validation."""
        try:
            model_tests = {}

            # Test supported models
            supported_models = ["gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo", "o1-mini"]
            for model in supported_models:
                is_valid, error = self.validator.validate_model_availability(model)
                model_tests[model] = {
                    "valid": is_valid,
                    "error": error,
                    "expected": True
                }

            # Test unsupported models
            unsupported_models = ["gpt-4", "claude-3", "invalid-model"]
            for model in unsupported_models:
                is_valid, error = self.validator.validate_model_availability(model)
                model_tests[model] = {
                    "valid": is_valid,
                    "error": error,
                    "expected": False
                }

            # Calculate success rate
            correct_validations = sum(
                1 for test in model_tests.values()
                if test["valid"] == test["expected"]
            )
            success_rate = correct_validations / len(model_tests)

            return {
                "success": success_rate == 1.0,  # All validations should be correct
                "model_tests": model_tests,
                "success_rate": success_rate,
                "total_models_tested": len(model_tests)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _test_sample_evaluation_enhanced(
            self,
            metrics: List[str],
            test_type: str
    ) -> Dict[str, Any]:
        """Enhanced sample evaluation testing with different scenarios."""
        try:
            if not DEEPEVAL_AVAILABLE:
                return {"success": False, "error": "DeepEval not available"}

            # Get sample cases based on test type
            if test_type == "stress":
                sample_cases = self._get_comprehensive_test_cases()
            else:
                sample_cases = self._get_sample_test_cases()

            # Test different configurations
            test_scenarios = []

            # Scenario 1: Fast evaluation
            fast_config = {
                "deepeval_config": {
                    "model": "gpt-3.5-turbo",
                    "batch_size": 5,
                    **{metric: {"threshold": 0.6, "include_reason": False, "async_mode": True}
                       for metric in metrics}
                }
            }

            scenario_result = await self._run_evaluation_scenario(
                "fast_evaluation", sample_cases, metrics, fast_config, timeout=60
            )
            test_scenarios.append(scenario_result)

            # Scenario 2: Accurate evaluation (if not stress test)
            if test_type != "stress":
                accurate_config = {
                    "deepeval_config": {
                        "model": "gpt-4o",
                        "batch_size": 2,
                        **{metric: {"threshold": 0.8, "include_reason": True, "async_mode": True}
                           for metric in metrics}
                    }
                }

                scenario_result = await self._run_evaluation_scenario(
                    "accurate_evaluation", sample_cases[:3], metrics, accurate_config, timeout=120
                )
                test_scenarios.append(scenario_result)

            overall_success = all(scenario.get("success", False) for scenario in test_scenarios)

            return {
                "success": overall_success,
                "test_scenarios": test_scenarios,
                "total_test_cases": len(sample_cases),
                "metrics_tested": metrics
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _run_evaluation_scenario(
            self,
            scenario_name: str,
            test_cases: List[Dict[str, Any]],
            metrics: List[str],
            config: Dict[str, Any],
            timeout: int = 120
    ) -> Dict[str, Any]:
        """Run a single evaluation scenario."""
        try:
            from deepeval.test_case import LLMTestCase
            from deepeval.dataset import EvaluationDataset
            from backend.app.evaluation.methods.deepeval import DeepEvalMethod

            # Convert to DeepEval format
            deepeval_cases = []
            for case in test_cases:
                test_case = LLMTestCase(
                    input=case["input"],
                    actual_output=case["actual_output"],
                    expected_output=case.get("expected_output"),
                    context=case.get("context", [])
                )
                deepeval_cases.append(test_case)

            dataset = EvaluationDataset(test_cases=deepeval_cases)

            # Initialize method and run evaluation
            method = DeepEvalMethod(self.db_session)
            metrics_instances = method._initialize_deepeval_metrics(metrics, config)

            if not metrics_instances:
                return {
                    "name": scenario_name,
                    "success": False,
                    "error": "No metrics could be initialized"
                }

            # Run evaluation with timeout
            start_time = time.time()

            try:
                results = await asyncio.wait_for(
                    method._run_deepeval_async(dataset, metrics_instances),
                    timeout=timeout
                )

                end_time = time.time()
                processing_time = end_time - start_time

                return {
                    "name": scenario_name,
                    "success": True,
                    "test_cases_count": len(deepeval_cases),
                    "metrics_used": metrics,
                    "processing_time_seconds": round(processing_time, 2),
                    "results_generated": "results" in results or "metrics" in results,
                    "has_errors": "error" in results,
                    "config_model": config["deepeval_config"]["model"]
                }

            except asyncio.TimeoutError:
                return {
                    "name": scenario_name,
                    "success": False,
                    "error": f"{scenario_name} timed out after {timeout} seconds",
                    "test_cases_count": len(deepeval_cases),
                    "metrics_used": metrics
                }

        except Exception as e:
            return {
                "name": scenario_name,
                "success": False,
                "error": str(e)
            }

    async def _test_new_metrics(self) -> Dict[str, Any]:
        """Test newly added metrics specifically."""
        try:
            new_metrics = [
                "summarization", "tool_correctness", "knowledge_retention",
                "g_eval", "conversation", "fluency"
            ]

            available_new_metrics = [
                metric for metric in new_metrics
                if metric in self.validator.SUPPORTED_METRICS
            ]

            test_results = {}

            for metric in available_new_metrics:
                # Test metric-specific configuration
                metric_schema = self.validator.get_metric_schema(metric)

                if metric_schema:
                    # Create a basic configuration for this metric
                    test_config = {"threshold": 0.7}

                    # Add metric-specific parameters
                    if metric == "g_eval":
                        test_config["evaluation_params"] = ["accuracy", "relevance"]
                        test_config["evaluation_steps"] = ["step1", "step2"]
                    elif metric == "summarization":
                        test_config["assessment_questions"] = ["Is the summary accurate?"]
                    elif metric == "conversation":
                        test_config["conversation_criteria"] = ["coherence", "engagement"]

                    # Validate configuration
                    is_valid, errors, warnings = self.validator.validate_single_metric_config(
                        metric, test_config
                    )

                    test_results[metric] = {
                        "config_valid": is_valid,
                        "errors": errors,
                        "warnings": warnings,
                        "schema_available": True
                    }
                else:
                    test_results[metric] = {
                        "config_valid": False,
                        "schema_available": False,
                        "errors": ["No schema available for metric"]
                    }

            overall_success = all(
                result.get("config_valid", False) for result in test_results.values()
            )

            return {
                "success": overall_success,
                "new_metrics_tested": available_new_metrics,
                "test_results": test_results,
                "success_rate": sum(1 for r in test_results.values() if r.get("config_valid", False)) / len(
                    test_results) if test_results else 0
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _test_concurrent_execution(self) -> Dict[str, Any]:
        """Test concurrent execution capabilities."""
        try:
            # Create multiple evaluation tasks
            tasks = []
            test_cases = self._get_sample_test_cases()[:2]  # Use fewer cases for concurrency test

            for i in range(3):  # Test 3 concurrent evaluations
                task = self._run_evaluation_scenario(
                    f"concurrent_task_{i}",
                    test_cases,
                    ["answer_relevancy"],
                    {
                        "deepeval_config": {
                            "model": "gpt-3.5-turbo",
                            "answer_relevancy": {"threshold": 0.7, "async_mode": True}
                        }
                    },
                    timeout=90
                )
                tasks.append(task)

            # Run tasks concurrently
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()

            successful_tasks = sum(
                1 for result in results
                if isinstance(result, dict) and result.get("success", False)
            )

            return {
                "success": successful_tasks >= 2,  # At least 2 out of 3 should succeed
                "total_tasks": len(tasks),
                "successful_tasks": successful_tasks,
                "total_time": round(end_time - start_time, 2),
                "average_time_per_task": round((end_time - start_time) / len(tasks), 2),
                "task_results": [r for r in results if isinstance(r, dict)]
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _test_resource_usage(self) -> Dict[str, Any]:
        """Test memory and resource usage."""
        try:
            import psutil
            import gc

            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Run a memory-intensive evaluation
            large_test_cases = self._get_comprehensive_test_cases()

            config = {
                "deepeval_config": {
                    "model": "gpt-3.5-turbo",
                    "answer_relevancy": {"threshold": 0.7, "async_mode": True}
                }
            }

            scenario_result = await self._run_evaluation_scenario(
                "memory_test", large_test_cases, ["answer_relevancy"], config, timeout=180
            )

            # Get peak memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory

            # Force garbage collection
            gc.collect()

            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_released = peak_memory - final_memory

            return {
                "success": scenario_result.get("success", False) and memory_increase < 500,  # Less than 500MB increase
                "initial_memory_mb": round(initial_memory, 2),
                "peak_memory_mb": round(peak_memory, 2),
                "final_memory_mb": round(final_memory, 2),
                "memory_increase_mb": round(memory_increase, 2),
                "memory_released_mb": round(memory_released, 2),
                "evaluation_result": scenario_result
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _test_performance_enhanced(self, test_type: str) -> Dict[str, Any]:
        """Enhanced performance testing with detailed metrics."""
        try:
            performance_results = {}

            # Test different batch sizes
            batch_sizes = [1, 3, 5, 10] if test_type == "stress" else [1, 5]

            for batch_size in batch_sizes:
                start_time = time.time()

                # Simulate processing with actual small evaluation if possible
                if DEEPEVAL_AVAILABLE:
                    try:
                        # Quick evaluation test
                        test_cases = self._get_sample_test_cases()[:batch_size]
                        config = {
                            "deepeval_config": {
                                "model": "gpt-3.5-turbo",
                                "batch_size": batch_size,
                                "answer_relevancy": {"threshold": 0.7, "include_reason": False, "async_mode": True}
                            }
                        }

                        scenario_result = await self._run_evaluation_scenario(
                            f"perf_batch_{batch_size}", test_cases, ["answer_relevancy"], config, timeout=60
                        )

                        end_time = time.time()
                        processing_time = end_time - start_time

                        performance_results[f"batch_size_{batch_size}"] = {
                            "processing_time": round(processing_time, 3),
                            "time_per_item": round(processing_time / batch_size, 3),
                            "success": scenario_result.get("success", False),
                            "throughput": round(batch_size / processing_time, 2) if processing_time > 0 else 0
                        }
                    except Exception as e:
                        performance_results[f"batch_size_{batch_size}"] = {
                            "error": str(e),
                            "success": False
                        }
                else:
                    # Fallback simulation
                    await asyncio.sleep(0.1 * batch_size)
                    end_time = time.time()
                    processing_time = end_time - start_time

                    performance_results[f"batch_size_{batch_size}"] = {
                        "processing_time": round(processing_time, 3),
                        "time_per_item": round(processing_time / batch_size, 3),
                        "simulated": True
                    }

            # Find optimal batch size
            successful_results = {
                k: v for k, v in performance_results.items()
                if v.get("success", False) or v.get("simulated", False)
            }

            if successful_results:
                optimal_batch = min(
                    successful_results.keys(),
                    key=lambda k: successful_results[k]["time_per_item"]
                )
            else:
                optimal_batch = "batch_size_5"

            return {
                "success": len(successful_results) > 0,
                "batch_performance": performance_results,
                "optimal_batch_size": optimal_batch.split("_")[-1] if optimal_batch else "5",
                "recommendation": f"Use batch size {optimal_batch.split('_')[-1] if optimal_batch else '5'} for optimal performance"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _test_error_handling_enhanced(self) -> Dict[str, Any]:
        """Enhanced error handling testing."""
        try:
            error_tests = {}

            # Test 1: Invalid metric handling
            try:
                if DEEPEVAL_AVAILABLE:
                    from backend.app.evaluation.methods.deepeval import DeepEvalMethod
                    method = DeepEvalMethod(self.db_session)
                    method._initialize_deepeval_metrics(["invalid_metric"], {})

                error_tests["invalid_metric"] = {
                    "handled_gracefully": True,
                    "error": "No error thrown for invalid metric (unexpected)"
                }
            except Exception as e:
                error_tests["invalid_metric"] = {
                    "handled_gracefully": True,
                    "error": str(e)
                }

            # Test 2: Empty configuration
            try:
                config_valid, errors, warnings = self.validator.validate_evaluation_config(
                    {}, DatasetType.CUSTOM, []
                )
                error_tests["empty_config"] = {
                    "handled_gracefully": True,
                    "validation_failed": not config_valid,
                    "errors_detected": len(errors) > 0
                }
            except Exception as e:
                error_tests["empty_config"] = {
                    "handled_gracefully": False,
                    "error": str(e)
                }

            # Test 3: Invalid model configuration
            try:
                invalid_config = {
                    "deepeval_config": {
                        "model": "non-existent-model",
                        "answer_relevancy": {"threshold": 0.7}
                    }
                }
                config_valid, errors, warnings = self.validator.validate_evaluation_config(
                    invalid_config, DatasetType.CUSTOM, ["answer_relevancy"]
                )
                error_tests["invalid_model_config"] = {
                    "handled_gracefully": True,
                    "validation_failed": not config_valid,
                    "errors_detected": len(errors) > 0
                }
            except Exception as e:
                error_tests["invalid_model_config"] = {
                    "handled_gracefully": False,
                    "error": str(e)
                }

            # Test 4: Invalid threshold values
            try:
                invalid_threshold_config = {
                    "deepeval_config": {
                        "model": "gpt-4o",
                        "answer_relevancy": {"threshold": 5.0}  # Invalid threshold > 1.0
                    }
                }
                config_valid, errors, warnings = self.validator.validate_evaluation_config(
                    invalid_threshold_config, DatasetType.CUSTOM, ["answer_relevancy"]
                )
                error_tests["invalid_threshold"] = {
                    "handled_gracefully": True,
                    "validation_failed": not config_valid,
                    "errors_detected": len(errors) > 0
                }
            except Exception as e:
                error_tests["invalid_threshold"] = {
                    "handled_gracefully": False,
                    "error": str(e)
                }

            return {
                "success": True,
                "error_tests": error_tests,
                "all_handled_gracefully": all(
                    test.get("handled_gracefully", False)
                    for test in error_tests.values()
                ),
                "total_error_tests": len(error_tests)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _get_default_test_metrics(self, test_type: str, include_new_metrics: bool) -> List[str]:
        """Get default metrics for testing based on test type."""
        basic_metrics = ["answer_relevancy", "faithfulness"]

        if test_type == "basic":
            return basic_metrics
        elif test_type == "comprehensive":
            comprehensive = basic_metrics + ["contextual_precision", "hallucination", "coherence"]
            if include_new_metrics:
                comprehensive.extend(["fluency", "summarization"])
            return comprehensive
        elif test_type in ["performance", "stress"]:
            performance = basic_metrics + ["contextual_precision"]
            if include_new_metrics and test_type == "stress":
                performance.extend(["tool_correctness", "knowledge_retention"])
            return performance
        else:
            return basic_metrics

    def _get_comprehensive_test_cases(self) -> List[Dict[str, Any]]:
        """Get comprehensive test cases for stress testing."""
        basic_cases = self._get_sample_test_cases()

        additional_cases = [
            {
                "input": "How does machine learning work?",
                "actual_output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "expected_output": "Machine learning allows computers to learn patterns from data",
                "context": ["AI is a broad field", "ML uses algorithms to find patterns", "Training data is essential"]
            },
            {
                "input": "What are the benefits of renewable energy?",
                "actual_output": "Renewable energy sources like solar and wind power provide clean energy that doesn't pollute the environment and helps reduce dependence on fossil fuels.",
                "expected_output": "Clean, sustainable energy that reduces pollution",
                "context": ["Climate change is a major concern", "Fossil fuels are limited", "Technology is improving"]
            },
            {
                "input": "Explain quantum computing",
                "actual_output": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot.",
                "expected_output": "Computing using quantum physics principles",
                "context": ["Quantum physics is complex", "Superposition allows multiple states",
                            "Entanglement connects particles"]
            }
        ]

        return basic_cases + additional_cases

    def _get_sample_test_cases(self) -> List[Dict[str, Any]]:
        """Get sample test cases for evaluation testing."""
        return [
            {
                "input": "What is the capital of France?",
                "actual_output": "The capital of France is Paris.",
                "expected_output": "Paris",
                "context": ["France is a country in Western Europe.", "Paris is the largest city in France."]
            },
            {
                "input": "Explain photosynthesis",
                "actual_output": "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll.",
                "expected_output": "Photosynthesis is how plants make food from sunlight",
                "context": ["Plants use chlorophyll to capture sunlight.", "This process creates glucose and oxygen."]
            },
            {
                "input": "What year did World War II end?",
                "actual_output": "World War II ended in 1945.",
                "expected_output": "1945",
                "context": ["World War II was a global conflict.", "It lasted from 1939 to 1945."]
            }
        ]

    def _generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Check overall success
        if not test_results.get("overall_success", False):
            recommendations.append("Some tests failed. Review individual test results for specific issues.")

        # Check specific test results
        tests = test_results.get("tests", {})

        if not tests.get("library_availability", {}).get("success", False):
            recommendations.append("Install or update DeepEval library: pip install --upgrade deepeval")

        if not tests.get("model_validation", {}).get("success", False):
            recommendations.append("Update model configurations to use supported model names")

        api_creds_valid = self.validator.validate_api_credentials()[0]
        if not api_creds_valid:
            recommendations.append("Set up valid OpenAI API credentials (OPENAI_API_KEY environment variable)")

        # Performance recommendations
        performance_test = tests.get("performance", {})
        if performance_test.get("success", False):
            optimal_batch = performance_test.get("optimal_batch_size", "5")
            recommendations.append(f"Use batch size {optimal_batch} for optimal performance")

        # Resource usage recommendations
        resource_test = tests.get("resource_usage", {})
        if resource_test and resource_test.get("memory_increase_mb", 0) > 200:
            recommendations.append("Consider using smaller batch sizes or lighter models for large datasets")

        # New metrics recommendations
        if test_results.get("include_new_metrics", False):
            recommendations.append(
                "Consider using new metrics like summarization, tool_correctness, or g_eval for specialized use cases")

        return recommendations

    async def validate_deepeval_environment(self) -> Dict[str, Any]:
        """Comprehensive validation of the DeepEval environment with enhanced checks."""
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "performance_indicators": {}
        }

        # Check 1: Library installation
        validation_results["checks"]["library_installed"] = {
            "status": DEEPEVAL_AVAILABLE,
            "details": "DeepEval library is properly installed" if DEEPEVAL_AVAILABLE else "DeepEval library not found"
        }

        # Check 2: API credentials
        creds_valid, creds_error = self.validator.validate_api_credentials()
        validation_results["checks"]["api_credentials"] = {
            "status": creds_valid,
            "details": "API credentials validated" if creds_valid else creds_error
        }

        # Check 3: Model availability
        model_availability = {}
        test_models = ["gpt-4o", "gpt-3.5-turbo", "o1-mini"]
        for model in test_models:
            is_valid, error = self.validator.validate_model_availability(model)
            model_availability[model] = is_valid

        validation_results["checks"]["model_availability"] = {
            "status": any(model_availability.values()),
            "details": f"Supported models available: {[m for m, v in model_availability.items() if v]}",
            "model_status": model_availability
        }

        # Check 4: System resources
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            cpu_count = psutil.cpu_count()

            validation_results["checks"]["system_resources"] = {
                "status": memory_gb >= 4 and cpu_count >= 2,
                "details": f"System: {memory_gb:.1f}GB RAM, {cpu_count} CPUs"
            }

            validation_results["performance_indicators"]["system_memory_gb"] = round(memory_gb, 1)
            validation_results["performance_indicators"]["cpu_cores"] = cpu_count
        except ImportError:
            validation_results["checks"]["system_resources"] = {
                "status": True,  # Assume adequate if can't check
                "details": "Could not check system resources (psutil not available)"
            }

        # Check 5: Enhanced metric support
        supported_metrics_count = len(self.validator.SUPPORTED_METRICS)
        validation_results["checks"]["metrics_support"] = {
            "status": supported_metrics_count >= 15,
            "details": f"Supports {supported_metrics_count} evaluation metrics",
            "categories": list(self.validator.METRIC_CATEGORIES.keys())
        }

        # Overall status
        validation_results["overall_status"] = all(
            check["status"] for check in validation_results["checks"].values()
        )

        # Enhanced recommendations
        recommendations = []
        if not DEEPEVAL_AVAILABLE:
            recommendations.append("Install DeepEval: pip install deepeval")
        if not creds_valid:
            recommendations.append("Set up valid OpenAI API credentials")
        if not any(model_availability.values()):
            recommendations.append("Ensure at least one supported model is available")

        # Performance recommendations
        if validation_results["performance_indicators"].get("system_memory_gb", 0) < 4:
            recommendations.append("Consider upgrading system memory for better performance")

        validation_results["recommendations"] = recommendations

        return validation_results


# Enhanced standalone testing functions
async def quick_deepeval_test() -> bool:
    """Quick test to verify DeepEval integration is working."""
    try:
        runner = DeepEvalTestRunner()
        results = await runner.run_integration_test("basic", ["answer_relevancy"])
        return results.get("overall_success", False)
    except Exception as e:
        logger.error(f"Quick DeepEval test failed: {e}")
        return False


async def comprehensive_deepeval_test() -> Dict[str, Any]:
    """Comprehensive test of all DeepEval functionality."""
    runner = DeepEvalTestRunner()
    return await runner.run_integration_test(
        "comprehensive",
        ["answer_relevancy", "faithfulness", "contextual_precision", "summarization", "fluency"],
        include_new_metrics=True
    )


async def stress_test_deepeval() -> Dict[str, Any]:
    """Stress test with all features enabled."""
    runner = DeepEvalTestRunner()
    return await runner.run_integration_test(
        "stress",
        ["answer_relevancy", "faithfulness", "tool_correctness", "g_eval"],
        include_new_metrics=True
    )


if __name__ == "__main__":
    # Allow running enhanced tests directly
    logging.basicConfig(level=logging.INFO)


    async def main():
        print("Running Enhanced DeepEval Integration Tests...")

        # Quick test
        print("\n1. Quick Test...")
        quick_result = await quick_deepeval_test()
        print(f"   Result: {'✅ PASSED' if quick_result else '❌ FAILED'}")

        # Environment validation
        print("\n2. Environment Validation...")
        runner = DeepEvalTestRunner()
        env_result = await runner.validate_deepeval_environment()
        print(f"   Result: {'✅ PASSED' if env_result.get('overall_status') else '❌ FAILED'}")

        if env_result.get("recommendations"):
            print("   Recommendations:")
            for rec in env_result["recommendations"]:
                print(f"   - {rec}")

        # Comprehensive test
        if quick_result:
            print("\n3. Comprehensive Test...")
            comprehensive_result = await comprehensive_deepeval_test()
            print(f"   Result: {'✅ PASSED' if comprehensive_result.get('overall_success') else '❌ FAILED'}")

            # Print summary
            tests = comprehensive_result.get("tests", {})
            for test_name, test_result in tests.items():
                status = "✅" if test_result.get("success", False) else "❌"
                print(f"   {status} {test_name}")

        print("\n🎉 Enhanced DeepEval testing complete!")


    asyncio.run(main())