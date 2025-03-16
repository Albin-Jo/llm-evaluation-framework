# File: app/api/v1/agents.py
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import BaseRepository
from app.db.session import get_db
from app.models.orm.models import MicroAgent, User
from app.schema.microagent_schema import (
    MicroAgentCreate, MicroAgentResponse, MicroAgentUpdate
)
from app.services.auth import get_current_active_user, get_current_admin_user

router = APIRouter()


@router.post("/", response_model=MicroAgentResponse)
async def create_microagent(
        microagent_data: MicroAgentCreate,
        current_user: User = Depends(get_current_admin_user),  # Only admins can create
        db: AsyncSession = Depends(get_db)
):
    """
    Create a new MicroAgent. Admin only.
    """
    microagent_repo = BaseRepository(MicroAgent, db)

    # Create MicroAgent
    microagent_dict = microagent_data.model_dump()

    microagent = await microagent_repo.create(microagent_dict)
    return microagent


@router.get("/", response_model=List[MicroAgentResponse])
async def list_microagents(
        skip: int = 0,
        limit: int = 100,
        domain: Optional[str] = None,
        is_active: Optional[bool] = None,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    List MicroAgents with optional filtering.
    """
    filters = {}

    # Add filters if provided
    if domain:
        filters["domain"] = domain
    if is_active is not None:
        filters["is_active"] = is_active

    # Get MicroAgents
    microagent_repo = BaseRepository(MicroAgent, db)
    microagents = await microagent_repo.get_multi(skip=skip, limit=limit, filters=filters)

    return microagents


@router.get("/{microagent_id}", response_model=MicroAgentResponse)
async def get_microagent(
        microagent_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Get MicroAgent by ID.
    """
    microagent_repo = BaseRepository(MicroAgent, db)
    microagent = await microagent_repo.get(microagent_id)

    if not microagent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MicroAgent with ID {microagent_id} not found"
        )

    return microagent


@router.put("/{microagent_id}", response_model=MicroAgentResponse)
async def update_microagent(
        microagent_id: UUID,
        microagent_data: MicroAgentUpdate,
        current_user: User = Depends(get_current_admin_user),  # Only admins can update
        db: AsyncSession = Depends(get_db)
):
    """
    Update MicroAgent by ID. Admin only.
    """
    microagent_repo = BaseRepository(MicroAgent, db)
    microagent = await microagent_repo.get(microagent_id)

    if not microagent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MicroAgent with ID {microagent_id} not found"
        )

    # Update the MicroAgent
    update_data = {
        k: v for k, v in microagent_data.model_dump().items() if v is not None
    }

    if not update_data:
        return microagent

    updated_microagent = await microagent_repo.update(microagent_id, update_data)
    return updated_microagent


@router.delete("/{microagent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_microagent(
        microagent_id: UUID,
        current_user: User = Depends(get_current_admin_user),  # Only admins can delete
        db: AsyncSession = Depends(get_db)
):
    """
    Delete MicroAgent by ID. Admin only.
    """
    microagent_repo = BaseRepository(MicroAgent, db)
    microagent = await microagent_repo.get(microagent_id)

    if not microagent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MicroAgent with ID {microagent_id} not found"
        )

    # Delete the MicroAgent
    success = await microagent_repo.delete(microagent_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete MicroAgent"
        )


@router.post("/{microagent_id}/test", response_model=Dict)
async def test_microagent(
        microagent_id: UUID,
        test_input: Dict,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Test a MicroAgent with sample input.
    """
    import httpx

    microagent_repo = BaseRepository(MicroAgent, db)
    microagent = await microagent_repo.get(microagent_id)

    if not microagent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MicroAgent with ID {microagent_id} not found"
        )

    if not microagent.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MicroAgent is not active"
        )

    try:
        # Call the MicroAgent API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                microagent.api_endpoint,
                json=test_input,
                timeout=30.0
            )

            response.raise_for_status()
            return response.json()

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error from MicroAgent API: {e.response.status_code} - {e.response.text}"
        )

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error connecting to MicroAgent API: {str(e)}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )