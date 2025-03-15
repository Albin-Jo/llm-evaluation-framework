# # File: tests/integration/test_evaluation_pipeline.py
# import pytest
# import asyncio
# import json
# import os
# from uuid import uuid4
# from unittest.mock import patch, AsyncMock, MagicMock
#
# from app.models.orm.models import User, UserRole, MicroAgent, Dataset, DatasetType
# from app.models.orm.models import Prompt, Evaluation, EvaluationMethod, EvaluationStatus
# from app.services.evaluation_service import EvaluationService
# from app.schema.evaluation_schema import EvaluationCreate
# from app.workers.tasks import _run_evaluation
#
#
# @pytest.mark.asyncio
# async def test_evaluation_pipeline(
#         db_session_sync, test_user_sync, settings
# ):
#     """Test the entire evaluation pipeline."""
#     # Create test entities using mocks
#     microagent = MagicMock(spec=MicroAgent)
#     microagent.id = uuid4()
#     microagent.name = "Test Integration Agent"
#     microagent.api_endpoint = "http://localhost:8000/test-integration-agent"
#
#     dataset = MagicMock(spec=Dataset)
#     dataset.id = uuid4()
#     dataset.name = "Test Integration Dataset"
#     dataset.file_path = "test_data/integration_dataset.json"
#
#     prompt = MagicMock(spec=Prompt)
#     prompt.id = uuid4()
#     prompt.name = "Test Integration Prompt"
#     prompt.content = "Answer the following question: {query}\nContext: {context}"
#
#     # Setup evaluation service mock
#     evaluation_service = MagicMock(spec=EvaluationService)
#
#     # Setup evaluation
#     evaluation = MagicMock(spec=Evaluation)
#     evaluation.id = uuid4()
#     evaluation.name = "Test Integration Evaluation"
#     evaluation.method = EvaluationMethod.RAGAS
#     evaluation.status = EvaluationStatus.PENDING
#     evaluation.created_by_id = test_user_sync.id
#     evaluation.micro_agent_id = microagent.id
#     evaluation.dataset_id = dataset.id
#     evaluation.prompt_id = prompt.id
#     evaluation.config = {"metrics": ["faithfulness", "answer_relevancy", "context_relevancy"]}
#
#     # Mock evaluation service methods with AsyncMock
#     evaluation_service.get_evaluation = AsyncMock(return_value=evaluation)
#     evaluation_service.start_evaluation = AsyncMock(return_value=evaluation)
#
#     # Create a mock for the evaluation method handler
#     method_handler = AsyncMock()
#     method_handler.run_evaluation = AsyncMock(return_value=[])
#     evaluation_service.get_evaluation_method_handler = AsyncMock(return_value=method_handler)
#
#     evaluation_service.create_evaluation_result = AsyncMock()
#     evaluation_service.complete_evaluation = AsyncMock(return_value=evaluation)
#
#     # Run test with patched service
#     with patch('app.services.evaluation_service.EvaluationService', return_value=evaluation_service):
#         # Run the evaluation
#         result = await _run_evaluation(evaluation.id)
#
#         # Check result
#         assert "completed successfully" in result
#
#         # Verify service calls
#         evaluation_service.get_evaluation.assert_called_with(evaluation.id)
#         evaluation_service.start_evaluation.assert_called_with(evaluation.id)
#         evaluation_service.get_evaluation_method_handler.assert_called_with(evaluation.method)
#         method_handler.run_evaluation.assert_called_with(evaluation)
#         evaluation_service.complete_evaluation.assert_called_with(evaluation.id, success=True)