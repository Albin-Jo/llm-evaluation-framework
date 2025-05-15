import logging
from typing import Optional
from uuid import UUID

from sqlalchemy import select, func

# Configure logger
logger = logging.getLogger(__name__)


async def update_evaluation_progress(evaluation_id: UUID, processed_items: int, total_items: Optional[int] = None):
    """
    Update evaluation progress directly in the database.
    This ensures progress is visible across all processes.
    """
    try:
        # Create a new database session for this update
        from backend.app.db.session import db_session
        from backend.app.db.repositories.base import BaseRepository
        from backend.app.db.models.orm import Evaluation, Dataset

        async with db_session() as session:
            # Use repository to update the record
            repo = BaseRepository(Evaluation, session)

            # Update processed_items in database
            update_data = {"processed_items": processed_items}

            # If dataset has no row_count yet but we know the total, update it
            if total_items is not None:
                # Get the evaluation to check if we need to update its dataset
                evaluation = await repo.get(evaluation_id)
                if evaluation and evaluation.dataset_id:
                    dataset_repo = BaseRepository(Dataset, session)
                    dataset = await dataset_repo.get(evaluation.dataset_id)
                    if dataset and (not dataset.row_count or dataset.row_count == 0):
                        await dataset_repo.update(dataset.id, {"row_count": total_items})
                        logger.info(f"Updated dataset {dataset.id} row_count to {total_items}")

            # Update the evaluation
            await repo.update(evaluation_id, update_data)

            # Commit the changes immediately
            await session.commit()

            logger.debug(f"Updated evaluation {evaluation_id} progress: {processed_items} items processed")

    except Exception as e:
        # Log error but don't fail the evaluation if progress update fails
        logger.error(f"Error updating evaluation progress in database: {e}", exc_info=True)


async def get_evaluation_result_count(db_session, evaluation_id: UUID) -> int:
    """
    Get the number of evaluation results for an evaluation.

    Args:
        db_session: Database session
        evaluation_id: Evaluation ID

    Returns:
        int: Number of evaluation results
    """
    try:
        from backend.app.db.models.orm import EvaluationResult

        query = select(func.count()).select_from(EvaluationResult).where(
            EvaluationResult.evaluation_id == evaluation_id
        )
        result = await db_session.execute(query)
        return result.scalar_one_or_none() or 0
    except Exception as e:
        logger.error(f"Error getting evaluation result count: {e}")
        return 0
