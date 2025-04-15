# File: tests/api/test_reports.py

import pytest
from httpx import AsyncClient
from uuid import uuid4

from backend.app.db.models.orm import ReportStatus, ReportFormat
from backend.app.main import app


@pytest.mark.asyncio
async def test_create_report(client: AsyncClient, test_evaluation):
    """Test creating a report."""
    # Create report data
    report_data = {
        "name": "Test Report",
        "description": "Test report description",
        "evaluation_id": str(test_evaluation["id"]),
        "format": ReportFormat.PDF.value,
        "include_executive_summary": True,
        "include_evaluation_details": True,
        "include_metrics_overview": True,
        "include_detailed_results": True,
        "include_agent_responses": True,
        "is_public": False
    }

    # Send request
    response = await client.post("/api/v1/reports/", json=report_data)

    # Check response
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == report_data["name"]
    assert data["description"] == report_data["description"]
    assert data["evaluation_id"] == report_data["evaluation_id"]
    assert data["format"] == report_data["format"]
    assert data["status"] == ReportStatus.DRAFT.value


@pytest.mark.asyncio
async def test_list_reports(client: AsyncClient, test_report):
    """Test listing reports."""
    # Send request
    response = await client.get("/api/v1/reports/")

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1

    # Check if test report is in the list
    report_ids = [r["id"] for r in data]
    assert str(test_report["id"]) in report_ids


@pytest.mark.asyncio
async def test_get_report(client: AsyncClient, test_report):
    """Test getting a report."""
    # Send request
    response = await client.get(f"/api/v1/reports/{test_report['id']}")

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(test_report["id"])
    assert data["name"] == test_report["name"]


@pytest.mark.asyncio
async def test_update_report(client: AsyncClient, test_report):
    """Test updating a report."""
    # Update data
    update_data = {
        "name": "Updated Report Name",
        "description": "Updated report description"
    }

    # Send request
    response = await client.put(f"/api/v1/reports/{test_report['id']}", json=update_data)

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(test_report["id"])
    assert data["name"] == update_data["name"]
    assert data["description"] == update_data["description"]


@pytest.mark.asyncio
async def test_delete_report(client: AsyncClient, test_report):
    """Test deleting a report."""
    # Send request
    response = await client.delete(f"/api/v1/reports/{test_report['id']}")

    # Check response
    assert response.status_code == 204

    # Verify report is deleted
    get_response = await client.get(f"/api/v1/reports/{test_report['id']}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_generate_report(client: AsyncClient, test_report):
    """Test generating a report."""
    # Send request
    response = await client.post(
        f"/api/v1/reports/{test_report['id']}/generate",
        json={"force_regenerate": True}
    )

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(test_report["id"])
    assert data["status"] == ReportStatus.GENERATED.value
    assert data["file_path"] is not None


@pytest.mark.asyncio
async def test_download_report(client: AsyncClient, test_generated_report):
    """Test downloading a report."""
    # Send request
    response = await client.get(f"/api/v1/reports/{test_generated_report['id']}/download")

    # Check response
    assert response.status_code == 200
    assert response.headers["content-type"] in [
        "application/pdf", "text/html", "application/json"
    ]
    assert "content-disposition" in response.headers


@pytest.mark.asyncio
async def test_send_report(client: AsyncClient, test_generated_report):
    """Test sending a report."""
    # Send data
    send_data = {
        "recipients": [
            {"email": "test@example.com", "name": "Test User"}
        ],
        "subject": "Test Report",
        "message": "Here is your test report",
        "include_pdf": True
    }

    # Send request
    response = await client.post(
        f"/api/v1/reports/{test_generated_report['id']}/send",
        json=send_data
    )

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["recipients_count"] == 1


@pytest.mark.asyncio
async def test_preview_report(client: AsyncClient, test_report):
    """Test previewing a report."""
    # Send request
    response = await client.get(f"/api/v1/reports/{test_report['id']}/preview")

    # Check response
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html"
    assert "<!DOCTYPE html>" in response.text


@pytest.mark.asyncio
async def test_report_not_found(client: AsyncClient):
    """Test error handling for non-existent report."""
    random_id = uuid4()

    # Send request
    response = await client.get(f"/api/v1/reports/{random_id}")

    # Check response
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()