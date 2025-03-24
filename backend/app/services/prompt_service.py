# File: app/services/prompt_service.py
from typing import Dict, List, Optional, Union
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.repositories.prompt_repository import PromptRepository, PromptTemplateRepository
from backend.app.db.models.orm.models import Prompt, PromptTemplate, User
from backend.app.db.schema.prompt_schema import (
    PromptCreate, PromptUpdate, PromptTemplateCreate, PromptTemplateUpdate
)


class PromptService:
    """Service for prompt-related operations."""

    def __init__(self, db: AsyncSession):
        """Initialize with database session."""
        self.db = db
        self.prompt_repo = PromptRepository(db)
        self.template_repo = PromptTemplateRepository(db)

    async def create_prompt_template(
        self, template_data: PromptTemplateCreate, current_user: User
    ) -> PromptTemplate:
        """
        Create a new prompt template with permission checks.

        Args:
            template_data: Template data
            current_user: Current user

        Returns:
            PromptTemplate: Created template

        Raises:
            HTTPException: If user doesn't have permission
        """
        # Only admins can create public templates
        if template_data.is_public and current_user.role.value != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can create public templates"
            )

        # Create prompt template
        template_dict = template_data.model_dump()
        template = await self.template_repo.create(template_dict)
        return template

    async def get_prompt_templates(
        self, skip: int = 0, limit: int = 100, is_public: Optional[bool] = None
    ) -> List[PromptTemplate]:
        """
        List prompt templates with optional filtering.

        Args:
            skip: Number of records to skip
            limit: Number of records to return
            is_public: Filter by public status

        Returns:
            List[PromptTemplate]: List of templates
        """
        filters = {}
        if is_public is not None:
            filters["is_public"] = is_public

        templates = await self.template_repo.get_multi(
            skip=skip, limit=limit, filters=filters
        )
        return templates

    async def get_prompt_template(self, template_id: UUID) -> PromptTemplate:
        """
        Get a prompt template by ID.

        Args:
            template_id: Template ID

        Returns:
            PromptTemplate: Template

        Raises:
            HTTPException: If template not found
        """
        template = await self.template_repo.get(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt template with ID {template_id} not found"
            )
        return template

    async def update_prompt_template(
        self, template_id: UUID, template_data: PromptTemplateUpdate, current_user: User
    ) -> PromptTemplate:
        """
        Update a prompt template with permission checks.

        Args:
            template_id: Template ID
            template_data: Update data
            current_user: Current user

        Returns:
            PromptTemplate: Updated template

        Raises:
            HTTPException: If template not found or user doesn't have permission
        """
        template = await self.template_repo.get(template_id)
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

        updated_template = await self.template_repo.update(template_id, update_data)
        return updated_template

    async def delete_prompt_template(self, template_id: UUID) -> bool:
        """
        Delete a prompt template by ID.

        Args:
            template_id: Template ID

        Returns:
            bool: True if deleted, False otherwise

        Raises:
            HTTPException: If template not found or delete failed
        """
        template = await self.template_repo.get(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt template with ID {template_id} not found"
            )

        success = await self.template_repo.delete(template_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete prompt template"
            )
        return True

    async def create_prompt(
        self, prompt_data: PromptCreate, current_user: User
    ) -> Prompt:
        """
        Create a new prompt with permission checks.

        Args:
            prompt_data: Prompt data
            current_user: Current user

        Returns:
            Prompt: Created prompt

        Raises:
            HTTPException: If user doesn't have permission or template not found
        """
        # Only admins can create public prompts
        if prompt_data.is_public and current_user.role.value != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can create public prompts"
            )

        # Check if template exists if provided
        if prompt_data.template_id:
            template = await self.template_repo.get(prompt_data.template_id)
            if not template:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Prompt template with ID {prompt_data.template_id} not found"
                )

        # Create prompt
        prompt_dict = prompt_data.model_dump()
        prompt_dict["owner_id"] = current_user.id

        prompt = await self.prompt_repo.create(prompt_dict)
        return prompt

    async def get_prompts(
        self,
        skip: int = 0,
        limit: int = 100,
        is_public: Optional[bool] = None,
        template_id: Optional[UUID] = None,
        current_user: User = None
    ) -> List[Prompt]:
        """
        List prompts with optional filtering and permission checks.

        Args:
            skip: Number of records to skip
            limit: Number of records to return
            is_public: Filter by public status
            template_id: Filter by template ID
            current_user: Current user

        Returns:
            List[Prompt]: List of prompts
        """
        filters = {}

        # Add filters if provided
        if is_public is not None:
            filters["is_public"] = is_public
        if template_id:
            filters["template_id"] = template_id

        if current_user.role.value == "admin":
            # Admins can see all prompts
            prompts = await self.prompt_repo.get_multi(
                skip=skip, limit=limit, filters=filters
            )
        else:
            # Regular users can see their own prompts and public prompts
            prompts_owned = await self.prompt_repo.get_multi(
                skip=0, limit=None, filters={"owner_id": current_user.id, **filters}
            )

            # If is_public filter is explicitly set to False, don't fetch public prompts
            if is_public is False:
                return prompts_owned

            # Get public prompts not owned by the user
            public_filters = {"is_public": True, **filters}
            prompts_public = await self.prompt_repo.get_multi(
                skip=0, limit=None, filters=public_filters
            )

            # Combine and paginate manually
            all_prompts = prompts_owned + [
                p for p in prompts_public if p.owner_id != current_user.id
            ]

            # Apply pagination
            prompts = all_prompts[skip:skip + limit]

        return prompts

    async def get_prompt(self, prompt_id: UUID, current_user: User) -> Prompt:
        """
        Get a prompt by ID with permission checks.

        Args:
            prompt_id: Prompt ID
            current_user: Current user

        Returns:
            Prompt: Prompt

        Raises:
            HTTPException: If prompt not found or user doesn't have permission
        """
        prompt = await self.prompt_repo.get(prompt_id)

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

    async def update_prompt(
        self, prompt_id: UUID, prompt_data: PromptUpdate, current_user: User
    ) -> Prompt:
        """
        Update a prompt with permission checks.

        Args:
            prompt_id: Prompt ID
            prompt_data: Update data
            current_user: Current user

        Returns:
            Prompt: Updated prompt

        Raises:
            HTTPException: If prompt not found, user doesn't have permission,
                          or template not found
        """
        prompt = await self.prompt_repo.get(prompt_id)

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
            template = await self.template_repo.get(prompt_data.template_id)
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

        updated_prompt = await self.prompt_repo.update(prompt_id, update_data)
        return updated_prompt

    async def delete_prompt(self, prompt_id: UUID, current_user: User) -> bool:
        """
        Delete a prompt with permission checks.

        Args:
            prompt_id: Prompt ID
            current_user: Current user

        Returns:
            bool: True if deleted, False otherwise

        Raises:
            HTTPException: If prompt not found, user doesn't have permission,
                          or delete failed
        """
        prompt = await self.prompt_repo.get(prompt_id)

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
        success = await self.prompt_repo.delete(prompt_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete prompt"
            )
        return True

    async def render_prompt(
        self, prompt_id: UUID, variables: Dict[str, str], current_user: User
    ) -> Dict[str, str]:
        """
        Render a prompt with variables.

        Args:
            prompt_id: Prompt ID
            variables: Variables to render
            current_user: Current user

        Returns:
            Dict[str, str]: Original and rendered content

        Raises:
            HTTPException: If prompt not found or user doesn't have permission
        """
        prompt = await self.prompt_repo.get(prompt_id)

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