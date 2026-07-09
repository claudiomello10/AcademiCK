"""Upload → process → query → re-upload, through the whole product."""


async def test_processed_book_is_listed_with_content(processed_book, book_stats):
    stats = await book_stats(processed_book)
    assert stats is not None
    assert stats["processing_status"] == "completed"
    assert stats["total_chunks"] > 0
    assert stats["total_chapters"] > 0


async def test_query_answers_from_uploaded_book(
    client, guest_token, auth, processed_book
):
    r = await client.post(
        "/api/v1/chat/single",
        json={"query": "What is the Zorbite consolidation algorithm and what does it do?"},
        headers=auth(guest_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["response"]
    assert processed_book in {s["book"] for s in body["sources"]}


async def test_reupload_replaces_instead_of_duplicating(
    processed_book, book_stats, upload_and_wait
):
    before = await book_stats(processed_book)
    await upload_and_wait()
    after = await book_stats(processed_book)

    assert after["total_chunks"] == before["total_chunks"]
    assert after["total_chapters"] == before["total_chapters"]


async def test_guest_cannot_upload(client, guest_token, auth):
    r = await client.post(
        "/api/v1/admin/upload-pdfs",
        files={"files": ("forbidden.pdf", b"%PDF-1.4 fake", "application/pdf")},
        headers=auth(guest_token),
    )
    assert r.status_code == 403
