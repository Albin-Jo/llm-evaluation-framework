# File: tests/evaluation/test_metrics_registry.py
import pytest

from app.evaluation.metrics.registry import MetricsRegistry
# pytestmark = pytest.mark.skipif(
#     True,  # Change to False when ready to enable these tests
#     reason="Database tests are currently disabled"
# )

def test_metrics_registry():
    """Test the metrics registry."""

    # Define test metrics
    def test_metric1(answer, context):
        return 0.5

    def test_metric2(answer, query):
        return 0.8

    # Register metrics
    MetricsRegistry.register(
        "test_metric1",
        test_metric1,
        "Test metric 1 description",
        1.0
    )

    MetricsRegistry.register(
        "test_metric2",
        test_metric2,
        "Test metric 2 description",
        0.5
    )

    # Get metric
    metric1 = MetricsRegistry.get("test_metric1")

    # Check metric
    assert metric1 is not None
    assert metric1["func"] == test_metric1
    assert metric1["description"] == "Test metric 1 description"
    assert metric1["default_weight"] == 1.0

    # List metrics
    metrics = MetricsRegistry.list_metrics()

    # Check metrics
    assert "test_metric1" in metrics
    assert "test_metric2" in metrics
    assert metrics["test_metric1"]["description"] == "Test metric 1 description"
    assert metrics["test_metric2"]["default_weight"] == 0.5

    # Get non-existent metric
    non_existent = MetricsRegistry.get("non_existent")
    assert non_existent is None