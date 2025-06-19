import logging
from typing import Dict, Any

import httpx
from fastapi import HTTPException, status
from jose import jwt

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class ImpersonationService:
    """Service for handling user impersonation via external API."""

    def __init__(self):
        # Configure from settings or environment
        self.impersonate_api_url = getattr(settings, 'IMPERSONATE_API_URL', 'https://api.example.com/impersonate')

    def _decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decode JWT token to extract user information.

        Args:
            token: JWT token to decode

        Returns:
            Dict containing decoded token payload
        """
        try:
            # Decode without verification since we trust the impersonation API
            decoded = jwt.get_unverified_claims(token)
            logger.debug(f"Decoded token claims: {decoded}")
            return decoded
        except Exception as e:
            logger.warning(f"Failed to decode token: {str(e)}")
            return {}

    async def impersonate_user(self, employee_id: str, auth_token: str) -> Dict[str, Any]:
        """
        Call external impersonation API to get token for employee.

        Args:
            employee_id: Employee ID to impersonate
            auth_token: Bearer token for API authentication

        Returns:
            Dict containing token and user info

        Raises:
            HTTPException: If impersonation fails
        """

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                auth_token_ = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJjRHZOOENlWlFLM2pwT2MyNV8zeUpXcGU3R3pBMzd3SXktV2pxLVY4dzVFIn0.eyJleHAiOjE3NTAzOTAxNTIsImlhdCI6MTc1MDA2NjE1MiwianRpIjoiOGYyY2ViMmMtYWUyYi00OTU2LTgxYWYtZmYyMjk5ZGMzMWFkIiwiaXNzIjoiaHR0cHM6Ly9wZW9wbGV4c3NvLXVhdC5xYXRhcmFpcndheXMuY29tLnFhL2F1dGgvcmVhbG1zL0VtcGxveWVlLUV4cGVyaWVuY2UiLCJhdWQiOlsicmVhbG0tbWFuYWdlbWVudCIsImFjY291bnQiXSwic3ViIjoiY2U4NWM3NzMtNDRlMC00YmU5LWE2NzUtOWQzMGJjMTMyYzNkIiwidHlwIjoiQmVhcmVyIiwiYXpwIjoiRVgtUmVzb3VyY2VBcHAiLCJzZXNzaW9uX3N0YXRlIjoiM2Q1MTZkMGYtYWIyMi00ZGU2LWI0ODgtOTJmODAyNTUwMjI1IiwiYWNyIjoiMSIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLWVtcGxveWVlLWV4cGVyaWVuY2UiLCJBdXRvbWF0aW9uX1Rlc3RfVXNlcl9Sb2xlIiwib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7InJlYWxtLW1hbmFnZW1lbnQiOnsicm9sZXMiOlsiaW1wZXJzb25hdGlvbiJdfSwiRVgtUmVzb3VyY2VBcHAiOnsicm9sZXMiOlsicGVvcGxleDpwaXhpLXRyYXZlbGJvdCJdfSwiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJyb2xlcyBlbWFpbCBwcm9maWxlIiwic2lkIjoiM2Q1MTZkMGYtYWIyMi00ZGU2LWI0ODgtOTJmODAyNTUwMjI1IiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJ0c3R1c3IwMiJ9.UN3x3TJ122wGStZuJhF7uWYzQEu7HtVqZnIMkG-SE5OxljQo9gn84nPHX8V1uS6E3axmsg0GqisuK-kcq9lMqyAtmfWvS5uWUkt0YtA1U81ofwFBtDf7UpPvb2nbuaIlSr0qaURT0l154SSERfmLCO9BbKBkkY6eVTr33N4faWmSLUbeBw14IGlSozzjL-jgmps8XArSxqa_4zjfYwUUI-CrSLQtNDKMOzRVy2ujnCvilYc5TAEt2lBSFLCnYerYmpMkYmSaKEMoVtEAAsPg99G2e0rs_IKuPQSUJ2-kYUuAjNyorHIdx_B4AF0JQuR8Hp4XOgZKzt-Z_7mJh_MD4w"
                headers = {
                    "Authorization": f"Bearer {auth_token_}",
                    "Content-Type": "application/json"
                }

                # Build URL with employee ID as path parameter
                impersonate_url = f"{self.impersonate_api_url}/{employee_id}"

                logger.info(f"Requesting impersonation token for employee {employee_id}")

                response = await client.get(
                    impersonate_url,
                    headers=headers
                )

                response.raise_for_status()
                data = response.json()

                if "accessToken" not in data:
                    error_msg = data.get("errorDescription") or data.get("error") or "Missing accessToken"
                    raise ValueError(f"Invalid response: {error_msg}")

                access_token = data["accessToken"]

                # Decode token to get user info
                decoded_token = self._decode_token(access_token)

                # Extract user info from decoded token
                user_info = {
                    "employee_id": employee_id,
                    "name": decoded_token.get("name", ""),
                    "preferred_username": decoded_token.get("preferred_username", ""),
                    "email": decoded_token.get("email", ""),
                    "expires_in": data.get("expiresIn"),
                    "token_type": data.get("tokenType", "Bearer"),
                    "scope": data.get("scope", ""),
                    "session_state": data.get("sessionState", "")
                }

                logger.info(f"Successfully impersonated employee {employee_id} - {user_info.get('name')}")

                return {
                    "token": access_token,
                    "refresh_token": data.get("refreshToken"),
                    "user_info": user_info
                }

        except httpx.HTTPStatusError as e:
            logger.error(f"Impersonation API error {e.response.status_code}: {e.response.text}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to impersonate user: {e.response.text}"
            )
        except Exception as e:
            logger.error(f"Impersonation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Impersonation failed: {str(e)}"
            )
