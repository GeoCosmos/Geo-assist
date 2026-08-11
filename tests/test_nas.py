"""Tests for the NAS share walker — filesystem policy only, no ingestion."""
from pathlib import Path

import pytest

import config
import nas


@pytest.fixture
def fake_nas(tmp_path, monkeypatch):
    """A NAS tree with real documents, QNAP junk, and unsupported files."""
    root = tmp_path / "documents"
    (root / "Manuals" / "Avionics").mkdir(parents=True)
    (root / "@eaDir").mkdir()
    (root / "@Recycle").mkdir()
    (root / "Manuals" / "@eaDir").mkdir()
    (root / ".hidden").mkdir()

    (root / "top.pdf").write_bytes(b"top")
    (root / "Manuals" / "guide.docx").write_bytes(b"guide")
    (root / "Manuals" / "Avionics" / "spec.pdf").write_bytes(b"spec")
    (root / "Manuals" / "sheet.xlsx").write_bytes(b"nope")
    (root / "@eaDir" / "thumb.pdf").write_bytes(b"junk")
    (root / "Manuals" / "@eaDir" / "thumb2.pdf").write_bytes(b"junk")
    (root / "@Recycle" / "deleted.pdf").write_bytes(b"junk")
    (root / "Thumbs.db").write_bytes(b"junk")
    (root / ".DS_Store").write_bytes(b"junk")
    (root / "~$draft.docx").write_bytes(b"lock")
    (root / ".hidden" / "secret.pdf").write_bytes(b"junk")

    monkeypatch.setattr(config, "NAS_ROOT", str(root))
    return root


def test_scan_finds_only_supported_documents(fake_nas):
    files, _counts = nas.scan(fake_nas)
    assert sorted(f.relpath for f in files) == [
        "Manuals/Avionics/spec.pdf",
        "Manuals/guide.docx",
        "top.pdf",
    ]


def test_scan_counts_unsupported_without_raising(fake_nas):
    _files, counts = nas.scan(fake_nas)
    assert counts["unsupported"] == 1  # sheet.xlsx; junk is excluded, not counted


def test_scan_skips_oversized_files(fake_nas, monkeypatch):
    monkeypatch.setattr(nas, "MAX_FILE_BYTES", 3)
    files, counts = nas.scan(fake_nas)
    # "guide" and "spec" are 5 and 4 bytes; "top" is 3 and survives.
    assert [f.relpath for f in files] == ["top.pdf"]
    assert counts["oversized"] == 2


def test_folder_derives_from_directory(fake_nas):
    files, _counts = nas.scan(fake_nas)
    by_path = {f.relpath: f.folder for f in files}
    assert by_path["top.pdf"] == "General"
    assert by_path["Manuals/guide.docx"] == "Manuals"
    assert by_path["Manuals/Avionics/spec.pdf"] == "Manuals/Avionics"


def test_scan_records_size_and_mtime(fake_nas):
    files, _counts = nas.scan(fake_nas)
    top = next(f for f in files if f.relpath == "top.pdf")
    assert top.size == 3
    assert top.mtime > 0


def test_resolve_subpath_accepts_empty_and_subdir(fake_nas):
    assert nas.resolve_subpath("") == Path(fake_nas).resolve()
    assert nas.resolve_subpath("Manuals") == (Path(fake_nas) / "Manuals").resolve()


@pytest.mark.parametrize("bad", ["..", "../etc", "Manuals/../..", "/etc", "/etc/passwd"])
def test_resolve_subpath_rejects_escapes(fake_nas, bad):
    with pytest.raises(ValueError):
        nas.resolve_subpath(bad)


def test_resolve_subpath_rejects_symlink_escape(fake_nas, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (fake_nas / "escape").symlink_to(outside)
    with pytest.raises(ValueError):
        nas.resolve_subpath("escape")


def test_scan_does_not_follow_symlinked_dirs(fake_nas, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leaked.pdf").write_bytes(b"leaked")
    (fake_nas / "link").symlink_to(outside)
    files, _counts = nas.scan(fake_nas)
    assert not any("leaked" in f.relpath for f in files)


def test_scan_missing_root_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        nas.scan(tmp_path / "nope")


def test_scan_excludes_qnap_snapshot_directories(tmp_path, monkeypatch):
    """@Recently-Snapshot holds a complete point-in-time copy of the whole share.

    Content hashing would stop duplicates reaching the store, but the scan would
    still read and hash every file twice — and a NAS retaining daily snapshots
    multiplies the corpus by the number of snapshots retained. Structure below is
    taken verbatim from a real QNAP share.
    """
    root = tmp_path / "share"
    (root / "Armsat_1").mkdir(parents=True)
    (root / "Armsat_1" / "report.docx").write_bytes(b"the real document")
    (root / "@Recycle").mkdir()
    (root / "@Recycle" / "desktop.ini").write_bytes(b"junk")
    snap = root / "@Recently-Snapshot" / "GMT+04_2026-08-11_0000" / "Armsat_1"
    snap.mkdir(parents=True)
    (snap / "report.docx").write_bytes(b"the real document")

    monkeypatch.setattr(config, "NAS_ROOT", str(root))
    files, _counts = nas.scan(root)
    assert [f.relpath for f in files] == ["Armsat_1/report.docx"]
