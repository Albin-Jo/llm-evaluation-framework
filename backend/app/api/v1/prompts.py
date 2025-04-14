from typing import List, Optional
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import JSONResponse

from backend.app.db.models.orm import Prompt, PromptTemplate
from backend.app.db.repositories.base import BaseRepository
from backend.app.db.schema.prompt_schema import (
    PromptCreate, PromptResponse, PromptUpdate
)
from backend.app.db.session import get_db

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=PromptResponse)
async def create_prompt(
        prompt_data: PromptCreate,
        db: AsyncSession = Depends(get_db)
):
    """
    Create a new prompt.
    """
    logger.info(f"Creating new prompt with template_id: {prompt_data.template_id}")

    # Check if template exists provided
    if prompt_data.template_id:
        logger.debug(f"Verifying template existence with ID: {prompt_data.template_id}")
        template_repo = BaseRepository(PromptTemplate, db)
        template = await template_repo.get(prompt_data.template_id)
        if not template:
            logger.warning(f"Prompt template with ID {prompt_data.template_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt template with ID {prompt_data.template_id} not found"
            )
        logger.debug(f"Template with ID {prompt_data.template_id} found")

    prompt_repo = BaseRepository(Prompt, db)

    # Create prompt
    prompt_dict = prompt_data.model_dump()
    logger.debug(f"Creating prompt with data: {prompt_dict}")

    try:
        prompt = await prompt_repo.create(prompt_dict)
        logger.info(f"Successfully created prompt with ID: {prompt.id}")
        return prompt
    except Exception as e:
        logger.error(f"Failed to create prompt: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create prompt"
        )


@router.get("/{prompt_id}", response_model=PromptResponse)
async def get_prompt(
        prompt_id: UUID,
        db: AsyncSession = Depends(get_db)
):
    """
    Get prompt by ID.
    """
    logger.info(f"Retrieving prompt with ID: {prompt_id}")

    prompt_repo = BaseRepository(Prompt, db)
    prompt = await prompt_repo.get(prompt_id)

    if not prompt:
        logger.warning(f"Prompt with ID {prompt_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt with ID {prompt_id} not found"
        )

    logger.info(f"Successfully retrieved prompt with ID: {prompt_id}")
    return prompt


@router.get("/", response_model=List[PromptResponse])
async def list_prompts(
        skip: int = 0,
        limit: int = 100,
        is_public: Optional[bool] = None,
        template_id: Optional[UUID] = None,
        db: AsyncSession = Depends(get_db)
):
    """
    List prompts with optional filtering.
    """
    logger.info(f"Listing prompts with skip={skip}, limit={limit}, is_public={is_public}, template_id={template_id}")

    filters = {}

    # Add filters if provided
    if is_public is not None:
        filters["is_public"] = is_public
    if template_id:
        filters["template_id"] = template_id

    logger.debug(f"Applied filters: {filters}")

    # Get prompts
    prompt_repo = BaseRepository(Prompt, db)

    try:
        prompts = await prompt_repo.get_multi(skip=skip, limit=limit, filters=filters)
        logger.info(f"Successfully retrieved {len(prompts)} prompts")
        return prompts
    except Exception as e:
        logger.error(f"Failed to retrieve prompts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve prompts"
        )


@router.put("/{prompt_id}", response_model=PromptResponse)
async def update_prompt(
        prompt_id: UUID,
        prompt_data: PromptUpdate,
        db: AsyncSession = Depends(get_db)
):
    """
    Update prompt by ID.
    """
    logger.info(f"Updating prompt with ID: {prompt_id}")

    prompt_repo = BaseRepository(Prompt, db)
    prompt = await prompt_repo.get(prompt_id)

    if not prompt:
        logger.warning(f"Prompt with ID {prompt_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt with ID {prompt_id} not found"
        )

    # Check if new template exists if provided
    if prompt_data.template_id and prompt_data.template_id != prompt.template_id:
        logger.debug(f"Verifying new template existence with ID: {prompt_data.template_id}")
        template_repo = BaseRepository(PromptTemplate, db)
        template = await template_repo.get(prompt_data.template_id)
        if not template:
            logger.warning(f"Prompt template with ID {prompt_data.template_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt template with ID {prompt_data.template_id} not found"
            )
        logger.debug(f"Template with ID {prompt_data.template_id} found")

    # Update the prompt
    update_data = {
        k: v for k, v in prompt_data.model_dump().items() if v is not None
    }

    if not update_data:
        logger.info(f"No data to update for prompt with ID: {prompt_id}")
        return prompt

    logger.debug(f"Updating prompt {prompt_id} with data: {update_data}")

    try:
        updated_prompt = await prompt_repo.update(prompt_id, update_data)
        logger.info(f"Successfully updated prompt with ID: {prompt_id}")
        return updated_prompt
    except Exception as e:
        logger.error(f"Failed to update prompt {prompt_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update prompt"
        )


@router.delete("/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prompt(
        prompt_id: UUID,
        db: AsyncSession = Depends(get_db)
):
    """
    Delete prompt by ID.
    """
    logger.info(f"Deleting prompt with ID: {prompt_id}")

    prompt_repo = BaseRepository(Prompt, db)
    prompt = await prompt_repo.get(prompt_id)

    if not prompt:
        logger.warning(f"Prompt with ID {prompt_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt with ID {prompt_id} not found"
        )

    # Delete the prompt
    try:
        success = await prompt_repo.delete(prompt_id)
        if not success:
            logger.error(f"Failed to delete prompt with ID: {prompt_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete prompt"
            )

        logger.info(f"Successfully deleted prompt with ID: {prompt_id}")
        return JSONResponse(
            content={"detail": f"Prompt with ID {prompt_id} successfully deleted"}
        )
    except Exception as e:
        logger.error(f"Exception when deleting prompt {prompt_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete prompt: {str(e)}"
        )