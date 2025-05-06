from typing import List, Any, Dict, TypeVar

from backend.app.db.schema.response_schema import PaginatedResponse

T = TypeVar('T')


def create_paginated_response(
        items: List[Any],
        total: int,
        skip: int,
        limit: int
) -> Dict[str, Any]:
    """
    Create a standardized paginated response.

    Returns a dictionary that matches the PaginatedResponse schema.
    This allows FastAPI to properly serialize the response.
    """
    page = (skip // limit) + 1
    has_next = (skip + limit) < total
    has_previous = skip > 0

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": limit,
        "has_next": has_next,
        "has_previous": has_previous
    }