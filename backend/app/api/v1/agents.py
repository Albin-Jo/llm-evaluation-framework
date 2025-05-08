import logging
from typing import Dict, Optional, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.exceptions import NotFoundException, DuplicateResourceException
from backend.app.db.repositories.agent_repository import AgentRepository
from backend.app.db.schema.agent_schema import (
    AgentCreate, AgentResponse, AgentUpdate
)
from backend.app.db.session import get_db
from backend.app.services.agent_service import test_agent_service
from backend.app.utils.response_utils import create_paginated_response

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
        agent_data: AgentCreate,
        db: AsyncSession = Depends(get_db)
):
    """
    Create a new Agent. Admin only.

    Args:
        agent_data: The agent data to create
        db: Database session

    Returns:
        The created agent

    Raises:
        HTTPException: If validation fails
    """
    logger.info(f"Creating new agent: {agent_data.name}")

    agent_repo = AgentRepository(db)

    # Check if an agent with this name already exists
    existing_agent = await agent_repo.get_by_name(agent_data.name)
    if existing_agent:
        raise DuplicateResourceException(
            resource="Agent",
            field="name",
            value=agent_data.name
        )

    # Create agent
    agent_dict = agent_data.model_dump()

    agent = await agent_repo.create(agent_dict)
    logger.info(f"Agent created successfully: {agent.id}")

    return agent


@router.get("/", response_model=Dict[str, Any])
async def list_agents(
        skip: int = 0,
        limit: int = 100,
        domain: Optional[str] = Query(None, description="Filter agents by domain"),
        is_active: Optional[bool] = Query(None, description="Filter agents by active status"),
        name: Optional[str] = Query(None, description="Filter agents by name (partial match)"),
        db: AsyncSession = Depends(get_db)
):
    """
    List Agents with optional filtering and pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        domain: Optional domain filter
        is_active: Optional active status filter
        name: Optional name filter (partial match)
        db: Database session

    Returns:
        Dict containing list of agents and pagination info
    """
    logger.debug(f"Listing agents with filters: domain={domain}, is_active={is_active}, name={name}")

    filters = {}
    if domain:
        filters["domain"] = domain
    if is_active is not None:
        filters["is_active"] = is_active

    agent_repo = AgentRepository(db)

    # Get total count first
    if name:
        # Count with name search
        total_count = await agent_repo.count_with_search(name, filters)
    else:
        total_count = await agent_repo.count(filters)

    # Get agents
    if name:
        agents = await agent_repo.search_by_name(name, skip=skip, limit=limit, additional_filters=filters)
    else:
        agents = await agent_repo.get_multi(skip=skip, limit=limit, filters=filters)

    agents_schema_list = [AgentResponse.from_orm(agent) for agent in agents]
    return create_paginated_response(agents_schema_list, total_count, skip, limit)


@router.post("/search", response_model=Dict[str, Any])
async def search_agents(
        query: Optional[str] = Body(None, description="Search query for name, description, or domain"),
        filters: Optional[Dict[str, Any]] = Body(None, description="Additional filters"),
        skip: int = Body(0, ge=0, description="Number of records to skip"),
        limit: int = Body(100, ge=1, le=1000, description="Maximum number of records to return"),
        db: AsyncSession = Depends(get_db)
):
    """
    Advanced search for agents across multiple fields.

    Supports text search across name, description, and domain fields,
    as well as additional filters for exact matches.

    Args:
        query: Search query text
        filters: Additional filters (exact match)
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session

    Returns:
        Dict containing search results and pagination info
    """
    logger.info(f"Advanced search for agents with query: '{query}' and filters: {filters}")

    agent_repo = AgentRepository(db)

    # Get total count
    total_count = await agent_repo.count_advanced_search(query, filters)

    # Get agents
    agents = await agent_repo.advanced_search(
        query_text=query,
        filters=filters,
        skip=skip,
        limit=limit
    )
    agents_schema_list = [AgentResponse.from_orm(agent) for agent in agents]
    return create_paginated_response(agents_schema_list, total_count, skip, limit)


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
        agent_id: UUID,
        db: AsyncSession = Depends(get_db)
):
    """
    Get Agent by ID.

    Args:
        agent_id: The ID of the agent to retrieve
        db: Database session

    Returns:
        The requested agent

    Raises:
        HTTPException: If agent not found
    """
    logger.debug(f"Getting agent with ID: {agent_id}")

    agent_repo = AgentRepository(db)
    agent = await agent_repo.get(agent_id)

    if not agent:
        raise NotFoundException(resource="Agent", resource_id=str(agent_id))

    return agent


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
        agent_id: UUID,
        agent_data: AgentUpdate,
        db: AsyncSession = Depends(get_db)
):
    """
    Update Agent by ID. Admin only.

    Args:
        agent_id: The ID of the agent to update
        agent_data: The updated agent data
        db: Database session

    Returns:
        The updated agent

    Raises:
        HTTPException: If agent not found or validation fails
    """
    logger.info(f"Updating agent with ID: {agent_id}")

    agent_repo = AgentRepository(db)
    agent = await agent_repo.get(agent_id)

    if not agent:
        raise NotFoundException(resource="Agent", resource_id=str(agent_id))

    # If name is being updated, check if it conflicts with another agent
    if agent_data.name and agent_data.name != agent.name:
        existing_agent = await agent_repo.get_by_name(agent_data.name)
        if existing_agent and existing_agent.id != agent_id:
            raise DuplicateResourceException(
                resource="Agent",
                field="name",
                value=agent_data.name
            )

    # Update the Agent
    update_data = {
        k: v for k, v in agent_data.model_dump().items() if v is not None
    }

    if not update_data:
        return agent

    updated_agent = await agent_repo.update(agent_id, update_data)
    logger.info(f"Agent updated successfully: {agent_id}")

    return updated_agent


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
        agent_id: UUID,
        db: AsyncSession = Depends(get_db)
):
    """
    Delete Agent by ID. Admin only.

    Args:
        agent_id: The ID of the agent to delete
        db: Database session

    Raises:
        HTTPException: If agent not found or deletion fails
    """
    logger.info(f"Deleting agent with ID: {agent_id}")

    agent_repo = AgentRepository(db)
    agent = await agent_repo.get(agent_id)

    if not agent:
        raise NotFoundException(resource="Agent", resource_id=str(agent_id))

    # Check if agent is used in any evaluations
    has_evaluations = await agent_repo.has_related_evaluations(agent_id)
    if has_evaluations:
        logger.warning(f"Cannot delete agent {agent_id} - has related evaluations")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete agent that has associated evaluations"
        )

    # Delete the Agent
    success = await agent_repo.delete(agent_id)
    if not success:
        logger.error(f"Failed to delete agent: {agent_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete Agent"
        )

    logger.info(f"Agent deleted successfully: {agent_id}")


@router.post("/{agent_id}/test", response_model=Dict)
async def test_agent(
        agent_id: UUID,
        test_input: Dict,
        db: AsyncSession = Depends(get_db)
):
    """
    Test an Agent with sample input.

    Args:
        agent_id: The ID of the agent to test
        test_input: The input data to test with
        db: Database session

    Returns:
        The response from the agent

    Raises:
        HTTPException: If agent not found, is inactive, or the test fails
    """
    logger.info(f"Testing agent with ID: {agent_id}")

    agent_repo = AgentRepository(db)
    agent = await agent_repo.get(agent_id)

    if not agent:
        raise NotFoundException(resource="Agent", resource_id=str(agent_id))

    if not agent.is_active:
        logger.warning(f"Cannot test inactive agent: {agent_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent is not active"
        )

    try:
        # Call the agent service to perform the test
        response = await test_agent_service(agent, test_input)
        logger.info(f"Agent test successful: {agent_id}")
        return response

    except Exception as e:
        logger.error(f"Error testing agent {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing agent: {str(e)}"
        )
