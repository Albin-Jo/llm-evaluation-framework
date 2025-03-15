# # File: tests/workers/test_tasks.py
# import pytest
# from unittest.mock import patch, AsyncMock
# from uuid import UUID
#
# from app.workers.tasks import run_evaluation_task, _run_evaluation
# from app.models.orm.models import EvaluationStatus
#
# # Add to the top of your test files that interact with the database:
# # pytestmark = pytest.mark.skipif(
# #     True,  # Change to False when ready to enable these tests
# #     reason="Database tests are currently disabled"
# # )
# @pytest.mark.asyncio
# async def test_run_evaluation(
#         db_session_sync, test_evaluation_sync, mock_httpx_client
# ):
#     """Test running an evaluation."""
#     # Mock the evaluation service
#     evaluation_service_mock = AsyncMock()
#     evaluation_service_mock.get_evaluation.return_value = test_evaluation_sync
#     evaluation_service_mock.start_evaluation.return_value = test_evaluation_sync
#     evaluation_service_mock.complete_evaluation.return_value = test_evaluation_sync
#
#     # Mock the method handler
#     method_handler_mock = AsyncMock()
#     method_handler_mock.run_evaluation.return_value = []
#     evaluation_service_mock.get_evaluation_method_handler.return_value = method_handler_mock
#
#     # Patch the service
#     with patch('app.services.evaluation_service.EvaluationService', return_value=evaluation_service_mock):
#         # Run the evaluation
#         result = await _run_evaluation(test_evaluation_sync.id)
#
#         # Check the result
#         assert "completed successfully" in result
#
#         # Verify service calls
#         evaluation_service_mock.get_evaluation.assert_called_with(test_evaluation_sync.id)
#         evaluation_service_mock.start_evaluation.assert_called_with(test_evaluation_sync.id)
#         evaluation_service_mock.get_evaluation_method_handler.assert_called_with(test_evaluation_sync.method)
#         method_handler_mock.run_evaluation.assert_called_with(test_evaluation_sync)
#         evaluation_service_mock.complete_evaluation.assert_called_with(test_evaluation_sync.id, success=True)
#
#
# @pytest.mark.asyncio
# async def test_run_evaluation_error(
#         db_session_sync, test_evaluation_sync, mock_httpx_client
# ):
#     """Test running an evaluation with error."""
#     # Mock the evaluation service
#     evaluation_service_mock = AsyncMock()
#     evaluation_service_mock.get_evaluation.return_value = test_evaluation_sync
#     evaluation_service_mock.start_evaluation.return_value = test_evaluation_sync
#     evaluation_service_mock.complete_evaluation.return_value = test_evaluation_sync
#
#     # Mock the method handler to raise an exception
#     method_handler_mock = AsyncMock()
#     method_handler_mock.run_evaluation.side_effect = Exception("Test error")
#     evaluation_service_mock.get_evaluation_method_handler.return_value = method_handler_mock
#
#     # Patch the service
#     with patch('app.services.evaluation_service.EvaluationService', return_value=evaluation_service_mock):
#         # Run the evaluation
#         result = await _run_evaluation(test_evaluation_sync.id)
#
#         # Check the result
#         assert "Error running evaluation" in result
#         assert "Test error" in result
#
#         # Verify service calls
#         evaluation_service_mock.get_evaluation.assert_called_with(test_evaluation_sync.id)
#         evaluation_service_mock.start_evaluation.assert_called_with(test_evaluation_sync.id)
#         evaluation_service_mock.get_evaluation_method_handler.assert_called_with(test_evaluation_sync.method)
#         method_handler_mock.run_evaluation.assert_called_with(test_evaluation_sync)
#         evaluation_service_mock.complete_evaluation.assert_called_with(test_evaluation_sync.id, success=False)
#
#
# @pytest.mark.asyncio
# async def test_run_evaluation_already_completed(
#         db_session_sync, test_evaluation_sync, mock_httpx_client
# ):
#     """Test running an evaluation that's already completed."""
#     # Mock the evaluation service
#     evaluation_service_mock = AsyncMock()
#
#     # Set evaluation status to completed
#     from app.models.orm.models import Evaluation, EvaluationStatus
#     from unittest.mock import MagicMock
#     completed_eval = MagicMock(spec=Evaluation)
#     completed_eval.id = test_evaluation_sync.id
#     completed_eval.status = EvaluationStatus.COMPLETED
#     evaluation_service_mock.get_evaluation.return_value = completed_eval
#
#     # Patch the service
#     with patch('app.services.evaluation_service.EvaluationService', return_value=evaluation_service_mock):
#         # Run the evaluation
#         result = await _run_evaluation(test_evaluation_sync.id)
#
#         # Check the result
#         assert "already in completed status" in result
#
#         # Verify service calls
#         evaluation_service_mock.get_evaluation.assert_called_with(test_evaluation_sync.id)
#         evaluation_service_mock.start_evaluation.assert_not_called()