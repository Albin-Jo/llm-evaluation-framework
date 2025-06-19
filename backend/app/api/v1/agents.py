"""
API endpoints for managing agents.

This module provides API endpoints for creating, retrieving, updating, and deleting agents,
as well as specialized endpoints for testing agents and listing MCP tools.
"""
import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.dependencies.auth import get_required_current_user, get_jwt_token
from backend.app.api.middleware.jwt_validator import UserContext
from backend.app.core.exceptions import NotFoundException, DuplicateResourceException
from backend.app.db.models.orm import AuthType, IntegrationType
from backend.app.db.repositories.agent_repository import AgentRepository
from backend.app.db.schema.agent_schema import (
    AgentCreate, AgentResponse, AgentUpdate
)
from backend.app.db.session import get_db
from backend.app.services.agent_clients.factory import AgentClientFactory
from backend.app.services.agent_service import test_agent_service
from backend.app.utils.credential_utils import mask_credentials
from backend.app.utils.response_utils import create_paginated_response

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
        agent_data: AgentCreate,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Create a new Agent.

    Args:
        agent_data: The agent data to create
        db: Database session
        current_user: Current authenticated user

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

    # Validate integration type specific requirements
    if agent_data.integration_type == IntegrationType.MCP:
        # For MCP, we can use the user's JWT token, so no need for stored credentials
        # Just ensure auth_type is set correctly
        if agent_data.auth_type != AuthType.BEARER_TOKEN:
            agent_data.auth_type = AuthType.BEARER_TOKEN
            logger.info(f"Setting auth_type to BEARER_TOKEN for MCP agent {agent_data.name}")

        # Ensure tool_name is specified in config
        if not agent_data.config or "tool_name" not in agent_data.config:
            if not agent_data.config:
                agent_data.config = {}
            agent_data.config["tool_name"] = "McpAskPolicyBot"  # Default tool
            logger.info(f"Set default tool_name 'McpAskPolicyBot' for MCP agent {agent_data.name}")
    else:
        # For non-MCP agents, validate credentials based on auth type
        if agent_data.auth_type == AuthType.BEARER_TOKEN:
            if not agent_data.auth_credentials or "token" not in agent_data.auth_credentials:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Bearer token authentication requires a 'token' field in auth_credentials"
                )
        elif agent_data.auth_type == AuthType.API_KEY:
            if not agent_data.auth_credentials or "api_key" not in agent_data.auth_credentials:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="API key authentication requires an 'api_key' field in auth_credentials"
                )

    # Create agent with encrypted credentials and associate with current user
    agent_dict = agent_data.model_dump()

    # Set the created_by_id if the user exists in database
    if current_user.db_user:
        agent_dict["created_by_id"] = current_user.db_user.id

    # Log masked credentials for debugging
    if agent_data.auth_credentials:
        masked_creds = mask_credentials(agent_data.auth_credentials)
        logger.debug(f"Creating agent with credentials: {masked_creds}")

    # Use special repository method to handle credential encryption
    agent = await agent_repo.create_with_encrypted_credentials(agent_dict)
    logger.info(f"Agent created successfully: {agent.id}")

    return agent


@router.get("/", response_model=Dict[str, Any])
async def list_agents(
        skip: int = 0,
        limit: int = 100,
        domain: Optional[str] = Query(None, description="Filter agents by domain"),
        is_active: Optional[bool] = Query(None, description="Filter agents by active status"),
        name: Optional[str] = Query(None, description="Filter agents by name (partial match)"),
        integration_type: Optional[IntegrationType] = Query(None, description="Filter agents by integration type"),
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    List Agents with optional filtering and pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        domain: Optional domain filter
        is_active: Optional active status filter
        name: Optional name filter (partial match)
        integration_type: Optional integration type filter
        db: Database session
        current_user: Current authenticated user

    Returns:
        Dict containing list of agents and pagination info
    """
    logger.debug(
        f"Listing agents with filters: domain={domain}, is_active={is_active}, name={name}, integration_type={integration_type}")

    filters = {}
    if domain:
        filters["domain"] = domain
    if is_active is not None:
        filters["is_active"] = is_active
    if integration_type:
        filters["integration_type"] = integration_type

    # Add name filter if provided (handled by base repository)
    if name:
        filters["name"] = name

    # Filter by user ID (show only user's own agents)
    user_id = current_user.db_user.id if current_user.db_user else None

    agent_repo = AgentRepository(db)

    # Get total count and agents
    total_count = await agent_repo.count(filters, user_id=user_id)
    agents = await agent_repo.get_multi(skip=skip, limit=limit, filters=filters, user_id=user_id)

    # Convert to response schema and add MCP-specific fields
    agents_schema_list = []
    for agent in agents:
        agent_response = AgentResponse.model_validate(agent)
        
        # Add tools and capabilities for MCP agents only
        if agent.integration_type == IntegrationType.MCP:
            # These will be populated by separate API calls if needed
            agent_response.tools = []
            agent_response.capabilities = []
        else:
            # Explicitly set to None for non-MCP agents
            agent_response.tools = None
            agent_response.capabilities = None
            
        agents_schema_list.append(agent_response)

    return create_paginated_response(agents_schema_list, total_count, skip, limit)


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
        agent_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Get Agent by ID.

    Args:
        agent_id: The ID of the agent to retrieve
        db: Database session
        current_user: Current authenticated user

    Returns:
        The requested agent

    Raises:
        HTTPException: If agent not found
    """
    logger.debug(f"Getting agent with ID: {agent_id}")

    agent_repo = AgentRepository(db)
    
    # Use user-owned method to ensure access control
    user_id = current_user.db_user.id if current_user.db_user else None
    agent = await agent_repo.get_user_owned(agent_id, user_id) if user_id else await agent_repo.get(agent_id)

    if not agent:
        raise NotFoundException(resource="Agent", resource_id=str(agent_id))

    # Convert to response schema and add MCP-specific fields
    agent_response = AgentResponse.model_validate(agent)
    
    # Add tools and capabilities for MCP agents only
    if agent.integration_type == IntegrationType.MCP:
        # These will be populated by separate API calls if needed
        agent_response.tools = []
        agent_response.capabilities = []
    else:
        # Explicitly set to None for non-MCP agents
        agent_response.tools = None
        agent_response.capabilities = None

    return agent_response


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
        agent_id: UUID,
        agent_data: AgentUpdate,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Update Agent by ID.

    Args:
        agent_id: The ID of the agent to update
        agent_data: The updated agent data
        db: Database session
        current_user: Current authenticated user

    Returns:
        The updated agent

    Raises:
        HTTPException: If agent not found or validation fails
    """
    logger.info(f"Updating agent with ID: {agent_id}")

    agent_repo = AgentRepository(db)
    
    # Use user-owned method to ensure access control
    user_id = current_user.db_user.id if current_user.db_user else None
    agent = await agent_repo.get_user_owned(agent_id, user_id) if user_id else await agent_repo.get(agent_id)

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

    # Check if integration type is being updated
    integration_type = agent_data.integration_type if agent_data.integration_type is not None else agent.integration_type

    # If integration type is MCP, make sure auth_type is BEARER_TOKEN
    if integration_type == IntegrationType.MCP:
        if agent_data.auth_type is not None and agent_data.auth_type != AuthType.BEARER_TOKEN:
            agent_data.auth_type = AuthType.BEARER_TOKEN
            logger.info(f"Setting auth_type to BEARER_TOKEN for MCP agent {agent_id}")

        # Ensure tool_name is specified for MCP
        if agent_data.config is not None:
            config = agent_data.config
            if "tool_name" not in config:
                config["tool_name"] = "McpAskPolicyBot"  # Default tool
                logger.info(f"Set default tool_name 'McpAskPolicyBot' for MCP agent {agent_id}")
    else:
        # For non-MCP agents, validate credentials based on auth type
        if agent_data.auth_type is not None:
            # Get effective credentials (updated or current)
            credentials = agent_data.auth_credentials if agent_data.auth_credentials is not None else agent.auth_credentials

            if agent_data.auth_type == AuthType.BEARER_TOKEN:
                if not credentials or "token" not in credentials:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Bearer token authentication requires a 'token' field in auth_credentials"
                    )
            elif agent_data.auth_type == AuthType.API_KEY:
                if not credentials or "api_key" not in credentials:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="API key authentication requires an 'api_key' field in auth_credentials"
                    )

    # Update the Agent with encrypted credentials
    update_data = {
        k: v for k, v in agent_data.model_dump(exclude_unset=True).items() if v is not None
    }

    if not update_data:
        # Convert to response schema and add MCP-specific fields
        agent_response = AgentResponse.model_validate(agent)
        
        # Add tools and capabilities for MCP agents only
        if agent.integration_type == IntegrationType.MCP:
            agent_response.tools = []
            agent_response.capabilities = []
        else:
            agent_response.tools = None
            agent_response.capabilities = None
            
        return agent_response

    # Log masked credentials if updating
    if agent_data.auth_credentials:
        masked_creds = mask_credentials(agent_data.auth_credentials)
        logger.debug(f"Updating agent credentials: {masked_creds}")

    # Use special repository method to handle credential encryption
    updated_agent = await agent_repo.update_with_encrypted_credentials(agent_id, update_data)
    logger.info(f"Agent updated successfully: {agent_id}")

    # Convert to response schema and add MCP-specific fields
    agent_response = AgentResponse.model_validate(updated_agent)
    
    # Add tools and capabilities for MCP agents only
    if updated_agent.integration_type == IntegrationType.MCP:
        agent_response.tools = []
        agent_response.capabilities = []
    else:
        agent_response.tools = None
        agent_response.capabilities = None

    return agent_response


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
        agent_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Delete Agent by ID.

    Args:
        agent_id: The ID of the agent to delete
        db: Database session
        current_user: Current authenticated user

    Raises:
        HTTPException: If agent not found or deletion fails
    """
    logger.info(f"Deleting agent with ID: {agent_id}")

    agent_repo = AgentRepository(db)
    
    # Use user-owned method to ensure access control
    user_id = current_user.db_user.id if current_user.db_user else None
    agent = await agent_repo.get_user_owned(agent_id, user_id) if user_id else await agent_repo.get(agent_id)

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

    # Delete the Agent using user ownership check
    success = await agent_repo.delete(agent_id, user_id=user_id)
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
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Test an Agent with sample input.

    Args:
        agent_id: The ID of the agent to test
        test_input: The input data to test with
        db: Database session
        current_user: Current authenticated user

    Returns:
        The response from the agent

    Raises:
        HTTPException: If agent not found, is inactive, or the test fails
    """
    logger.info(f"Testing agent with ID: {agent_id}")

    agent_repo = AgentRepository(db)
    
    # Use user-owned method to ensure access control
    user_id = current_user.db_user.id if current_user.db_user else None
    agent = await agent_repo.get_user_owned_with_decrypted_credentials(agent_id, user_id) if user_id else await agent_repo.get_with_decrypted_credentials(agent_id)

    if not agent:
        raise NotFoundException(resource="Agent", resource_id=str(agent_id))

    if not agent.is_active:
        logger.warning(f"Cannot test inactive agent: {agent_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent is not active"
        )

    try:
        # Get user token for MCP agents if available (safely)
        user_token = None
        if agent.integration_type == IntegrationType.MCP and hasattr(current_user, "token"):
            user_token = getattr(current_user, "token", None)

        # Call the agent service to perform the test
        response = await test_agent_service(agent, test_input, user_token)
        logger.info(f"Agent test successful: {agent_id}")
        return response

    except Exception as e:
        logger.error(f"Error testing agent {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing agent: {str(e)}"
        )


@router.post("/{agent_id}/test-mcp", response_model=Dict)
async def test_mcp_agent(
        agent_id: UUID,
        test_message: str = Body(..., embed=True, description="Message to test with"),
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user),
        jwt_token: Optional[str] = Depends(get_jwt_token)
):
    """
    Test an MCP agent with a simple message.

    Args:
        agent_id: The ID of the agent to test
        test_message: Message to test with
        db: Database session
        current_user: Current authenticated user
        jwt_token: JWT token from request

    Returns:
        The response from the agent

    Raises:
        HTTPException: If agent not found, is inactive, or the test fails
    """
    logger.info(f"Testing MCP agent with ID: {agent_id}")

    agent_repo = AgentRepository(db)
    
    # Use user-owned method to ensure access control
    user_id = current_user.db_user.id if current_user.db_user else None
    agent = await agent_repo.get_user_owned_with_decrypted_credentials(agent_id, user_id) if user_id else await agent_repo.get_with_decrypted_credentials(agent_id)

    if not agent:
        raise NotFoundException(resource="Agent", resource_id=str(agent_id))

    if not agent.is_active:
        logger.warning(f"Cannot test inactive agent: {agent_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent is not active"
        )

    # Verify agent is MCP type
    if agent.integration_type != IntegrationType.MCP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This endpoint is only for MCP agents"
        )

    try:
        # Create MCP client with JWT token
        client = await AgentClientFactory.create_client(agent, jwt_token)

        # Test with the client
        response = await client.process_query(test_message)

        logger.info(f"MCP agent test successful: {agent_id}")
        return response

    except Exception as e:
        logger.error(f"Error testing MCP agent {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing agent: {str(e)}"
        )


@router.get("/{agent_id}/tools", response_model=List[Dict])
async def list_agent_tools(
        agent_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user),
        jwt_token: Optional[str] = Depends(get_jwt_token)
):
    """
    List available tools for an agent. Returns empty array for non-MCP agents.

    Args:
        agent_id: The ID of the agent
        db: Database session
        current_user: Current authenticated user
        jwt_token: JWT token from request

    Returns:
        List of available tools (empty for non-MCP agents)

    Raises:
        HTTPException: If agent not found or is inactive
    """
    logger.info(f"Listing tools for agent with ID: {agent_id}")

    agent_repo = AgentRepository(db)
    
    # Use user-owned method to ensure access control
    user_id = current_user.db_user.id if current_user.db_user else None
    agent = await agent_repo.get_user_owned_with_decrypted_credentials(agent_id, user_id) if user_id else await agent_repo.get_with_decrypted_credentials(agent_id)

    if not agent:
        raise NotFoundException(resource="Agent", resource_id=str(agent_id))

    if not agent.is_active:
        logger.warning(f"Cannot list tools for inactive agent: {agent_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent is not active"
        )

    # Return empty array for non-MCP agents
    if agent.integration_type != IntegrationType.MCP:
        logger.info(f"Agent {agent_id} is not MCP type, returning empty tools list")
        return []

    try:
        # Create MCP client with JWT token
        client = await AgentClientFactory.create_client(agent, jwt_token)

        # List tools
        tools = await client.list_tools()

        # Convert to list of dictionaries for response
        tool_list = []
        for tool in tools:
            tool_dict = {
                "name": tool.name,
                "description": getattr(tool, "description", ""),
                "parameters": getattr(tool, "parameters", {}),
                "returns": getattr(tool, "returns", {})
            }
            tool_list.append(tool_dict)

        logger.info(f"Retrieved {len(tool_list)} tools for MCP agent {agent_id}")
        return tool_list

    except Exception as e:
        logger.error(f"Error listing tools for agent {agent_id}: {str(e)}")
        # Return empty array on error rather than raising exception
        logger.warning(f"Returning empty tools list due to error: {str(e)}")
        return []


@router.post("/{agent_id}/health", response_model=Dict[str, Any])
async def check_agent_health(
        agent_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Check if an agent is healthy and available.

    Args:
        agent_id: The ID of the agent to check
        db: Database session
        current_user: Current authenticated user

    Returns:
        Dict with health status

    Raises:
        HTTPException: If agent not found or is inactive
    """
    logger.info(f"Checking health for agent with ID: {agent_id}")

    agent_repo = AgentRepository(db)
    
    # Use user-owned method to ensure access control
    user_id = current_user.db_user.id if current_user.db_user else None
    agent = await agent_repo.get_user_owned_with_decrypted_credentials(agent_id, user_id) if user_id else await agent_repo.get_with_decrypted_credentials(agent_id)

    if not agent:
        raise NotFoundException(resource="Agent", resource_id=str(agent_id))

    if not agent.is_active:
        logger.warning(f"Cannot check health for inactive agent: {agent_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent is not active"
        )

    try:
        # Create client based on integration type
        client = await AgentClientFactory.create_client(agent)

        # Check health
        is_healthy = await client.health_check()

        return {
            "healthy": is_healthy,
            "agent_id": str(agent_id),
            "name": agent.name
        }
    except Exception as e:
        logger.error(f"Error checking agent health {agent_id}: {str(e)}")
        return {
            "healthy": False,
            "agent_id": str(agent_id),
            "name": agent.name,
            "error": str(e)
        }