"""Snapshot management through the real containers.

Unlike the service-tier snapshot tests (which run the gateway in-process on
the host), this exercises the gateway container's real filesystem — a
permissions regression on the metadata volume fails here.
"""

import pytest


async def test_snapshot_create_list_delete_in_container(client, admin_token, auth):
    r = await client.post("/api/v1/admin/snapshots/create", headers=auth(admin_token))
    if r.status_code == 403:
        pytest.skip("snapshot management disabled in this deployment")
    assert r.status_code == 200, r.text
    name = r.json()["snapshot_name"]

    try:
        r = await client.get("/api/v1/admin/snapshots", headers=auth(admin_token))
        assert r.status_code == 200, r.text
        snapshot = next(s for s in r.json()["snapshots"] if s["name"] == name)
        assert snapshot["has_metadata"] is True
    finally:
        r = await client.delete(
            f"/api/v1/admin/snapshots/{name}", headers=auth(admin_token)
        )
        assert r.status_code == 200, r.text

    r = await client.get("/api/v1/admin/snapshots", headers=auth(admin_token))
    assert name not in {s["name"] for s in r.json()["snapshots"]}


async def test_guest_cannot_manage_snapshots(client, guest_token, auth):
    r = await client.post("/api/v1/admin/snapshots/create", headers=auth(guest_token))
    assert r.status_code == 403
