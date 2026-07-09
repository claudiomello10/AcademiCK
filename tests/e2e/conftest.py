"""Full-stack end-to-end suite: real services, real LLM, real embeddings.

Runs against the compose stack through nginx (default http://localhost) and
skips cleanly when the stack is not up. Local only — never runs in CI.
"""

import asyncio
import os
import time

import httpx
import pytest

BASE_URL = os.getenv("E2E_BASE_URL", "http://localhost")
PROCESS_TIMEOUT = float(os.getenv("E2E_PROCESS_TIMEOUT", "300"))

FIXTURE_PDF = os.path.join(os.path.dirname(__file__), "fixtures", "academick-e2e-fixture.pdf")
BOOK_NAME = "academick-e2e-fixture"


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="session")
async def client():
    c = httpx.AsyncClient(base_url=BASE_URL, timeout=180.0)
    try:
        r = await c.get("/health")
    except httpx.HTTPError:
        await c.aclose()
        pytest.skip(f"stack not reachable at {BASE_URL} — start it with docker compose up")
    if r.status_code != 200:
        await c.aclose()
        pytest.skip(f"stack unhealthy at {BASE_URL}: {r.status_code}")
    yield c
    await c.aclose()


@pytest.fixture(scope="session")
async def admin_token(client):
    password = os.getenv("ADMIN_PASSWORD")
    if not password:
        pytest.skip("ADMIN_PASSWORD not set — run via scripts/run-tests.sh --e2e")
    r = await client.post(
        "/api/v1/login", json={"username": "admin", "password": password}
    )
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


@pytest.fixture(scope="session")
async def guest_token(client):
    password = os.getenv("GUEST_PASSWORD")
    if not password:
        pytest.skip("GUEST_PASSWORD not set — run via scripts/run-tests.sh --e2e")
    r = await client.post(
        "/api/v1/login", json={"username": "guest", "password": password}
    )
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


async def upload_and_wait(client, admin_token) -> dict:
    """Upload the fixture PDF and poll its job until completion."""
    with open(FIXTURE_PDF, "rb") as f:
        r = await client.post(
            "/api/v1/admin/upload-pdfs",
            files={"files": (f"{BOOK_NAME}.pdf", f.read(), "application/pdf")},
            headers=auth(admin_token),
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["jobs"], f"upload accepted no files: {body}"
    job_id = body["jobs"][0]["job_id"]

    deadline = time.monotonic() + PROCESS_TIMEOUT
    last = {}
    while time.monotonic() < deadline:
        r = await client.get(
            f"/api/v1/admin/pdf-job/{job_id}", headers=auth(admin_token)
        )
        assert r.status_code == 200, r.text
        last = r.json()
        if last.get("status") == "completed":
            return last
        if last.get("status") in ("failed", "cancelled"):
            raise AssertionError(f"processing did not complete: {last}")
        await asyncio.sleep(5)
    raise AssertionError(f"processing timed out after {PROCESS_TIMEOUT}s: {last}")


async def book_stats(client, admin_token, name: str) -> dict | None:
    r = await client.get("/api/v1/admin/book-list", headers=auth(admin_token))
    assert r.status_code == 200, r.text
    return next((b for b in r.json() if b["name"] == name), None)


@pytest.fixture(scope="session")
async def processed_book(client, admin_token):
    """The fixture book, uploaded and fully processed; deleted afterwards."""
    await upload_and_wait(client, admin_token)
    yield BOOK_NAME
    await client.delete(
        f"/api/v1/admin/books/{BOOK_NAME}", headers=auth(admin_token)
    )
