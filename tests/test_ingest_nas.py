"""Tests for the NAS scan driver: walk → manifest diff → batch ingest."""
import os

import pytest

import config
import ingest
import ingest_nas
import jobs


@pytest.fixture
def nas_tree(tmp_path, monkeypatch):
    root = tmp_path / "documents"
    (root / "Manuals").mkdir(parents=True)
    (root / "Manuals" / "alpha.txt").write_bytes(b"alpha document body")
    (root / "Manuals" / "beta.txt").write_bytes(b"beta document body")
    (root / "gamma.txt").write_bytes(b"gamma document body")
    (root / "sheet.xlsx").write_bytes(b"unsupported")
    monkeypatch.setattr(config, "NAS_ROOT", str(root))
    monkeypatch.setattr(config, "NAS_MANIFEST_PATH", str(tmp_path / "manifest.db"))
    ingest_nas._manifest_singleton = None
    yield root
    ingest_nas._manifest_singleton = None


@pytest.mark.asyncio
async def test_preview_counts_new_files_without_ingesting(nas_tree, mock_ollama):
    result = await ingest_nas.preview("")
    assert result["new"] == 3
    assert result["unchanged"] == 0
    assert result["unsupported"] == 1
    assert await ingest.list_documents() == []


@pytest.mark.asyncio
async def test_preview_scoped_to_subpath(nas_tree, mock_ollama):
    result = await ingest_nas.preview("Manuals")
    assert result["new"] == 2


@pytest.mark.asyncio
async def test_preview_rejects_escape(nas_tree, mock_ollama):
    with pytest.raises(ValueError):
        await ingest_nas.preview("../..")


@pytest.mark.asyncio
async def test_scan_ingests_all_files(nas_tree, mock_ollama):
    job = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job)
    assert job.status == "done"
    docs = await ingest.list_documents()
    assert sorted(d["filename"] for d in docs) == ["alpha.txt", "beta.txt", "gamma.txt"]


@pytest.mark.asyncio
async def test_scan_assigns_folder_from_directory(nas_tree, mock_ollama):
    job = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job)
    docs = {d["filename"]: d["folder"] for d in await ingest.list_documents()}
    assert docs["alpha.txt"] == "Manuals"
    assert docs["gamma.txt"] == "General"


@pytest.mark.asyncio
async def test_second_scan_skips_everything(nas_tree, mock_ollama):
    job1 = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job1)

    preview = await ingest_nas.preview("")
    assert preview["new"] == 0
    assert preview["unchanged"] == 3

    job2 = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job2)
    assert job2.total == 0
    assert len(await ingest.list_documents()) == 3


@pytest.mark.asyncio
async def test_modified_file_is_reingested(nas_tree, mock_ollama):
    job1 = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job1)

    path = nas_tree / "gamma.txt"
    path.write_bytes(b"gamma document body, revised")
    os.utime(path, (9_000_000, 9_000_000))

    preview = await ingest_nas.preview("")
    assert preview["new"] == 1


@pytest.mark.asyncio
async def test_existing_document_keeps_its_folder(nas_tree, mock_ollama):
    """A file uploaded by hand into one folder must not be reclassified when the
    identical bytes are found on the NAS."""
    await ingest.ingest(b"gamma document body", "gamma.txt", folder="HandFiled")
    job = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job)
    docs = {d["filename"]: d["folder"] for d in await ingest.list_documents()}
    assert docs["gamma.txt"] == "HandFiled"


@pytest.mark.asyncio
async def test_manifest_records_survive_for_resume(nas_tree, mock_ollama):
    job = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job)
    assert ingest_nas._manifest().count() == 3


@pytest.mark.asyncio
async def test_missing_root_fails_job_cleanly(tmp_path, monkeypatch, mock_ollama):
    monkeypatch.setattr(config, "NAS_ROOT", str(tmp_path / "gone"))
    monkeypatch.setattr(config, "NAS_MANIFEST_PATH", str(tmp_path / "m.db"))
    ingest_nas._manifest_singleton = None
    job = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job)
    assert job.status == "failed"
    assert any("not a directory" in e.lower() or "unreachable" in e.lower()
               for e in job.errors)


@pytest.mark.asyncio
async def test_mount_loss_aborts_the_job(nas_tree, mock_ollama, monkeypatch):
    """A mount that disappears mid-scan must fail the job, not report success
    after quietly logging one error per unreadable file."""
    monkeypatch.setattr(config, "NAS_IO_ERROR_LIMIT", 1)
    real_open = open

    def gone(path, *a, **kw):
        if str(path).endswith((".txt", ".pdf")):
            raise OSError(5, "Input/output error")
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", gone)
    job = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job)
    assert job.status == "failed"
    assert any("unreachable" in e.lower() for e in job.errors)


@pytest.mark.asyncio
async def test_errors_are_capped(nas_tree, mock_ollama, monkeypatch):
    monkeypatch.setattr(config, "NAS_MAX_ERRORS", 2)
    for i in range(5):
        (nas_tree / f"bad{i}.txt").write_bytes(f"body {i}".encode())

    async def boom(*a, **kw):
        raise RuntimeError("parse exploded")

    monkeypatch.setattr(ingest, "_prepare", boom)
    job = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job)
    # cap (2) + the "and N more" summary line; the unsupported-type notice for
    # sheet.xlsx is appended before the cap applies.
    assert len(job.errors) <= 4
    assert any("more error" in e for e in job.errors)
