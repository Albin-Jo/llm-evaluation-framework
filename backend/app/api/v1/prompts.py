from typing import Dict, List, Optional, Any, Annotated
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import JSONResponse

from backend.app.db.models.orm import Prompt, PromptTemplate
from backend.app.db.repositories.base import BaseRepository
from backend.app.db.schema.prompt_schema import (
    PromptCreate, PromptResponse, PromptUpdate
)
from backend.app.db.session import get_db
from backend.app.api.dependencies.rate_limiter import rate_limit

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=PromptResponse)
async def create_prompt(
        prompt_data: PromptCreate,
        db: AsyncSession = Depends(get_db),
        _: None = Depends(rate_limit(max_requests=20, period_seconds=60))
):
    """
    Create a new prompt.

    This endpoint creates a new prompt with the provided data.
    If a template_id is provided, it verifies that the template exists.

    - **prompt_data**: Required prompt data including name, content, and optional template_id

    Returns the created prompt object.
    """
    logger.info(f"Creating new prompt with template_id: {prompt_data.template_id}")

    # Check if template exists if provided
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
            detail=f"Failed to create prompt: {str(e)}"
        )


@router.get("/{prompt_id}", response_model=PromptResponse)
async def get_prompt(
        prompt_id: Annotated[UUID, Path(description="The ID of the prompt to retrieve")],
        db: AsyncSession = Depends(get_db)
):
    """
    Get prompt by ID.

    This endpoint retrieves a specific prompt by its unique identifier.

    - **prompt_id**: The unique identifier of the prompt

    Returns the prompt object if found.
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
    # Convert model to dict to avoid serialization issues
    return prompt


@router.get("/", response_model=Dict[str, Any])
async def list_prompts(
        skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
        limit: Annotated[int, Query(ge=1, le=100, description="Maximum number of records to return")] = 100,
        is_public: Annotated[Optional[bool], Query(description="Filter by public status")] = None,
        template_id: Annotated[Optional[UUID], Query(description="Filter by template ID")] = None,
        name: Annotated[Optional[str], Query(description="Filter by name (case-insensitive, partial match)")] = None,
        sort_by: Annotated[Optional[str], Query(description="Field to sort by")] = "created_at",
        sort_dir: Annotated[Optional[str], Query(description="Sort direction (asc or desc)")] = "desc",
        db: AsyncSession = Depends(get_db)
):
    """
    List prompts with optional filtering, sorting and pagination.

    This endpoint returns both the prompts array and a total count for pagination purposes.

    - **skip**: Number of records to skip (for pagination)
    - **limit**: Maximum number of records to return
    - **is_public**: Optional filter by public status (true/false)
    - **template_id**: Optional filter by template ID
    - **name**: Optional filter by prompt name (case-insensitive, supports partial matching)
    - **sort_by**: Field to sort results by (default: created_at)
    - **sort_dir**: Sort direction, either "asc" or "desc" (default: desc)

    Returns a dictionary containing the list of prompts and the total count.
    """
    logger.info(
        f"Listing prompts with skip={skip}, limit={limit}, is_public={is_public}, template_id={template_id}, name={name}")

    filters = {}

    # Add filters if provided
    if is_public is not None:
        filters["is_public"] = is_public
    if template_id:
        filters["template_id"] = template_id
    if name:
        filters["name"] = name

    # Validate sort_by parameter
    valid_sort_fields = ["created_at", "updated_at", "name", "version", "is_public"]
    if sort_by not in valid_sort_fields:
        logger.warning(f"Invalid sort field: {sort_by}, defaulting to created_at")
        sort_by = "created_at"

    # Validate sort_dir parameter
    if sort_dir.lower() not in ["asc", "desc"]:
        logger.warning(f"Invalid sort direction: {sort_dir}, defaulting to desc")
        sort_dir = "desc"

    from sqlalchemy import asc, desc
    # Apply sorting
    sort_expr = None
    if sort_by:
        if sort_dir.lower() == "asc":
            sort_expr = asc(getattr(Prompt, sort_by))
        else:
            sort_expr = desc(getattr(Prompt, sort_by))

    logger.debug(f"Applied filters: {filters}, sorting by {sort_by} {sort_dir}")

    # Get prompts
    prompt_repo = BaseRepository(Prompt, db)

    try:
        # Get total count for pagination
        count_query = select(func.count()).select_from(Prompt)

        # Apply filters to count query
        filter_conditions = []
        if filters:
            for field, value in filters.items():
                if hasattr(Prompt, field):
                    # Handle special case for string fields with LIKE operation
                    if isinstance(value, str) and field not in ["status", "method"]:
                        filter_conditions.append(getattr(Prompt, field).ilike(f"%{value}%"))
                    else:
                        filter_conditions.append(getattr(Prompt, field) == value)

        # Apply filter conditions if any
        if filter_conditions:
            count_query = count_query.where(and_(*filter_conditions))

        # Execute count query
        result = await db.execute(count_query)
        total_count = result.scalar_one_or_none() or 0

        # Get prompts with pagination
        prompts = await prompt_repo.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            sort=sort_expr,
            load_relationships=["template"]  # Optionally load template relationship
        )

        # Convert SQLAlchemy models to dictionaries to avoid serialization issues
        prompt_dicts = []
        for prompt in prompts:
            # Use the to_dict method from ModelMixin
            prompt_dict = prompt.to_dict()

            # Add template information if available
            if hasattr(prompt, 'template') and prompt.template:
                prompt_dict['template'] = prompt.template.to_dict() if prompt.template else None

            prompt_dicts.append(prompt_dict)

        logger.info(f"Successfully retrieved {len(prompts)} prompts from total of {total_count}")

        # Return both results and total count
        return {
            "items": prompt_dicts,
            "total": total_count
        }
    except Exception as e:
        logger.error(f"Failed to retrieve prompts: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve prompts: {str(e)}"
        )


@router.put("/{prompt_id}", response_model=PromptResponse)
async def update_prompt(
        prompt_id: Annotated[UUID, Path(description="The ID of the prompt to update")],
        prompt_data: PromptUpdate,
        db: AsyncSession = Depends(get_db)
):
    """
    Update prompt by ID.

    This endpoint updates an existing prompt with the provided data.
    If a new template_id is provided, it verifies that the template exists.

    - **prompt_id**: The unique identifier of the prompt to update
    - **prompt_data**: The prompt data to update

    Returns the updated prompt object.
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
            detail=f"Failed to update prompt: {str(e)}"
        )


@router.delete("/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prompt(
        prompt_id: Annotated[UUID, Path(description="The ID of the prompt to delete")],
        db: AsyncSession = Depends(get_db)
):
    """
    Delete prompt by ID.

    This endpoint completely removes a prompt. This operation cannot be undone.

    - **prompt_id**: The unique identifier of the prompt to delete

    Returns no content on success (HTTP 204).
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
            status_code=status.HTTP_204_NO_CONTENT,
            content=None
        )
    except Exception as e:
        logger.error(f"Exception when deleting prompt {prompt_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete prompt: {str(e)}"
        )