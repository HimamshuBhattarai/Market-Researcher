from fastapi.testclient import TestClient

from main import app


def test_api_root():
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_api_health():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_includes_research_route():
    client = TestClient(app)
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/research" in response.json()["paths"]


def test_api_rejects_empty_query():
    client = TestClient(app)
    response = client.post("/api/research", json={"query": "   "})

    assert response.status_code == 422
