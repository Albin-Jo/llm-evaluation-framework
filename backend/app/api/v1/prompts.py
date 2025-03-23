# File: app/api/v1/prompts.py
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.repositories.base import BaseRepository
from backend.app.db.session import get_db
from backend.app.db.models.orm.models import Prompt, PromptTemplate, User
from backend.app.db.schema.prompt_schema import (
    PromptCreate, PromptResponse, PromptTemplateCreate,
    PromptTemplateResponse, PromptTemplateUpdate, PromptUpdate
)
from backend.app.services.auth import get_current_active_user, get_current_admin_user

router = APIRouter()


# Prompt Template Endpoints

@router.post("/templates/", response_model=PromptTemplateResponse)
async def create_prompt_template(
        template_data: PromptTemplateCreate,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Create a new prompt template.
    """
    # Only admins can create public templates
    if template_data.is_public and current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can create public templates"
        )

    template_repo = BaseRepository(PromptTemplate, db)

    # Create prompt template
    template_dict = template_data.model_dump()

    template = await template_repo.create(template_dict)
    return template


@router.get("/templates/", response_model=List[PromptTemplateResponse])
async def list_prompt_templates(
        skip: int = 0,
        limit: int = 100,
        is_public: Optional[bool] = None,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    List prompt templates.
    """
    filters = {}

    # Add filters if provided
    if is_public is not None:
        filters["is_public"] = is_public

    template_repo = BaseRepository(PromptTemplate, db)
    templates = await template_repo.get_multi(skip=skip, limit=limit, filters=filters)

    return templates


@router.get("/templates/{template_id}", response_model=PromptTemplateResponse)
async def get_prompt_template(
        template_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Get prompt template by ID.
    """
    template_repo = BaseRepository(PromptTemplate, db)
    template = await template_repo.get(template_id)

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt template with ID {template_id} not found"
        )

    return template


@router.put("/templates/{template_id}", response_model=PromptTemplateResponse)
async def update_prompt_template(
        template_id: UUID,
        template_data: PromptTemplateUpdate,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Update prompt template by ID.
    """
    template_repo = BaseRepository(PromptTemplate, db)
    template = await template_repo.get(template_id)

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt template with ID {template_id} not found"
        )

    # Only admins can update public templates or change a template's public status
    if (template.is_public or (template_data.is_public is not None)) and current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can update public templates or change a template's public status"
        )

    # Update the template
    update_data = {
        k: v for k, v in template_data.model_dump().items() if v is not None
    }

    if not update_data:
        return template

    updated_template = await template_repo.update(template_id, update_data)
    return updated_template


@router.delete("/templates/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prompt_template(
        template_id: UUID,
        current_user: User = Depends(get_current_admin_user),  # Only admins can delete templates
        db: AsyncSession = Depends(get_db)
):
    """
    Delete prompt template by ID. Admin only.
    """
    template_repo = BaseRepository(PromptTemplate, db)
    template = await template_repo.get(template_id)

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt template with ID {template_id} not found"
        )

    # Delete the template
    success = await template_repo.delete(template_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete prompt template"
        )


# Prompt Endpoints

@router.post("/", response_model=PromptResponse)
async def create_prompt(
        prompt_data: PromptCreate,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Create a new prompt.
    """
    # Only admins can create public prompts
    if prompt_data.is_public and current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can create public prompts"
        )

    # Check if template exists if provided
    if prompt_data.template_id:
        template_repo = BaseRepository(PromptTemplate, db)
        template = await template_repo.get(prompt_data.template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt template with ID {prompt_data.template_id} not found"
            )

    prompt_repo = BaseRepository(Prompt, db)

    # Create prompt
    prompt_dict = prompt_data.model_dump()
    prompt_dict["owner_id"] = current_user.id

    prompt = await prompt_repo.create(prompt_dict)
    return prompt


@router.get("/", response_model=List[PromptResponse])
async def list_prompts(
        skip: int = 0,
        limit: int = 100,
        is_public: Optional[bool] = None,
        template_id: Optional[UUID] = None,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    List prompts with optional filtering.
    """
    filters = {}

    # Add filters if provided
    if is_public is not None:
        filters["is_public"] = is_public
    if template_id:
        filters["template_id"] = template_id

    # Get prompts
    prompt_repo = BaseRepository(Prompt, db)

    if current_user.role.value == "admin":
        # Admins can see all prompts
        prompts = await prompt_repo.get_multi(skip=skip, limit=limit, filters=filters)
    else:
        # Regular users can see their own prompts and public prompts
        prompts_owned = await prompt_repo.get_multi(
            skip=0, limit=None, filters={"owner_id": current_user.id, **filters}
        )

        # If is_public filter is explicitly set to False, don't fetch public prompts
        if is_public is False:
            return prompts_owned

        # Get public prompts not owned by the user
        public_filters = {"is_public": True, **filters}
        prompts_public = await prompt_repo.get_multi(
            skip=0, limit=None, filters=public_filters
        )

        # Combine and paginate manually
        all_prompts = prompts_owned + [
            p for p in prompts_public if p.owner_id != current_user.id
        ]

        # Apply pagination
        prompts = all_prompts[skip:skip + limit]

    return prompts


@router.get("/{prompt_id}", response_model=PromptResponse)
async def get_prompt(
        prompt_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Get prompt by ID.
    """
    prompt_repo = BaseRepository(Prompt, db)
    prompt = await prompt_repo.get(prompt_id)

    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt with ID {prompt_id} not found"
        )

    # Check if user has permission to view this prompt
    if (
            prompt.owner_id != current_user.id
            and not prompt.is_public
            and current_user.role.value != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    return prompt


@router.put("/{prompt_id}", response_model=PromptResponse)
async def update_prompt(
        prompt_id: UUID,
        prompt_data: PromptUpdate,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Update prompt by ID.
    """
    prompt_repo = BaseRepository(Prompt, db)
    prompt = await prompt_repo.get(prompt_id)

    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt with ID {prompt_id} not found"
        )

    # Check if user has permission to update this prompt
    if prompt.owner_id != current_user.id and current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Only admins can change a prompt's public status
    if prompt_data.is_public is not None and prompt_data.is_public != prompt.is_public and current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can change a prompt's public status"
        )

    # Check if new template exists if provided
    if prompt_data.template_id and prompt_data.template_id != prompt.template_id:
        template_repo = BaseRepository(PromptTemplate, db)
        template = await template_repo.get(prompt_data.template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt template with ID {prompt_data.template_id} not found"
            )

    # Update the prompt
    update_data = {
        k: v for k, v in prompt_data.model_dump().items() if v is not None
    }

    if not update_data:
        return prompt

    updated_prompt = await prompt_repo.update(prompt_id, update_data)
    return updated_prompt


@router.delete("/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prompt(
        prompt_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Delete prompt by ID.
    """
    prompt_repo = BaseRepository(Prompt, db)
    prompt = await prompt_repo.get(prompt_id)

    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt with ID {prompt_id} not found"
        )

    # Check if user has permission to delete this prompt
    if prompt.owner_id != current_user.id and current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Delete the prompt
    success = await prompt_repo.delete(prompt_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete prompt"
        )


@router.post("/{prompt_id}/render", response_model=Dict[str, str])
async def render_prompt(
        prompt_id: UUID,
        variables: Dict[str, str],
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Render a prompt with variables.
    """
    prompt_repo = BaseRepository(Prompt, db)
    prompt = await prompt_repo.get(prompt_id)

    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt with ID {prompt_id} not found"
        )

    # Check if user has permission to view this prompt
    if (
            prompt.owner_id != current_user.id
            and not prompt.is_public
            and current_user.role.value != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Render the prompt with variables
    rendered_content = prompt.content

    for key, value in variables.items():
        placeholder = f"{{{key}}}"
        rendered_content = rendered_content.replace(placeholder, str(value))

    return {
        "original": prompt.content,
        "rendered": rendered_content
    }