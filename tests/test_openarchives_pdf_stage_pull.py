from __future__ import annotations

from pathlib import Path

from glossapi.scripts.openarchives_pdf_stage_pull import (
    TransferItem,
    TransferState,
    canonicalize_pdf_name,
    load_priority_filenames,
    read_manifest,
)


def _write_manifest(path: Path) -> None:
    path.write_text(
        "\t".join(["canonical_filename", "remote_path", "remote_size_bytes", "remote_name"])
        + "\n"
        + "\t".join(["AAA_456.pdf", "/remote/AAA_456.pdf", "10", "AAA_456.pdf"])
        + "\n"
        + "\t".join(["VFK_368.pdf", "/remote/VFK_368.pdf.Ac6Dc3BA", "20", "VFK_368.pdf.Ac6Dc3BA"])
        + "\n",
        encoding="utf-8",
    )


def test_read_manifest_parses_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(manifest)

    items = read_manifest(manifest)

    assert items == [
        TransferItem("AAA_456.pdf", "/remote/AAA_456.pdf", 10, "AAA_456.pdf"),
        TransferItem("VFK_368.pdf", "/remote/VFK_368.pdf.Ac6Dc3BA", 20, "VFK_368.pdf.Ac6Dc3BA"),
    ]


def test_transfer_state_resets_stale_and_marks_completed(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    downloads = tmp_path / "downloads"
    partials = tmp_path / "partials"
    downloads.mkdir()
    partials.mkdir()
    state = TransferState(db_path)
    state.sync_manifest(
        [
            TransferItem("AAA_456.pdf", "/remote/AAA_456.pdf", 10, "AAA_456.pdf"),
            TransferItem("BBB_001.pdf", "/remote/BBB_001.pdf", 12, "BBB_001.pdf"),
        ]
    )

    state.mark_in_progress("AAA_456.pdf", 5)
    (downloads / "BBB_001.pdf").write_bytes(b"x" * 12)

    state.reset_stale_in_progress()
    state.mark_completed_if_present(downloads, partials)

    cur = state.conn.execute(
        "SELECT canonical_filename, status, last_seen_size_bytes, last_error FROM transfer_items ORDER BY canonical_filename"
    )
    rows = cur.fetchall()
    assert rows[0][0] == "AAA_456.pdf"
    assert rows[0][1] == "pending"
    assert "Recovered from interrupted transfer" in rows[0][3]
    assert rows[1][0] == "BBB_001.pdf"
    assert rows[1][1] == "completed"
    assert rows[1][2] == 12

    counts = state.counts()
    assert counts["pending"] == 1
    assert counts["completed"] == 1
    state.close()


def test_transfer_state_next_item_respects_attempt_limit(tmp_path: Path) -> None:
    state = TransferState(tmp_path / "state.sqlite3")
    state.sync_manifest(
        [
            TransferItem("AAA_456.pdf", "/remote/AAA_456.pdf", 10, "AAA_456.pdf"),
            TransferItem("BBB_001.pdf", "/remote/BBB_001.pdf", 12, "BBB_001.pdf"),
        ]
    )
    state.conn.execute(
        "UPDATE transfer_items SET status='failed', attempts=25 WHERE canonical_filename='AAA_456.pdf'"
    )
    state.conn.execute(
        "UPDATE transfer_items SET status='failed', attempts=2 WHERE canonical_filename='BBB_001.pdf'"
    )
    state.conn.commit()

    row = state.next_item(max_attempts=20)

    assert row is not None
    assert row["canonical_filename"] == "BBB_001.pdf"
    state.close()


def test_load_priority_filenames_supports_lists_and_suffix_forms(tmp_path: Path) -> None:
    priority_dir = tmp_path / "priority"
    priority_dir.mkdir()
    (priority_dir / "manual.txt").write_text(
        "AAA_456.pdf\n"
        "/tmp/VFK_368.pdf.Ac6Dc3BA\n"
        "ignore me\n",
        encoding="utf-8",
    )
    (priority_dir / "BBB_001.pdf").write_text("", encoding="utf-8")

    names = load_priority_filenames(priority_dir)

    assert names == {"AAA_456.pdf", "VFK_368.pdf", "BBB_001.pdf"}
    assert canonicalize_pdf_name("VFK_368.pdf.Ac6Dc3BA") == "VFK_368.pdf"


def test_transfer_state_priorities_are_selected_first(tmp_path: Path) -> None:
    state = TransferState(tmp_path / "state.sqlite3")
    state.sync_manifest(
        [
            TransferItem("AAA_456.pdf", "/remote/AAA_456.pdf", 10, "AAA_456.pdf"),
            TransferItem("BBB_001.pdf", "/remote/BBB_001.pdf", 12, "BBB_001.pdf"),
            TransferItem("CCC_002.pdf", "/remote/CCC_002.pdf", 14, "CCC_002.pdf"),
        ]
    )
    state.set_priorities({"CCC_002.pdf"})

    row = state.next_item(max_attempts=20)

    assert row is not None
    assert row["canonical_filename"] == "CCC_002.pdf"
    counts = state.priority_counts()
    assert counts["priority_total"] == 1
    assert counts["priority_pending"] == 1
    state.close()
