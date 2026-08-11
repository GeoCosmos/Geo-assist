import asyncio
import glob
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import bm25_index
import config
import ingest
import ingest_nas
import jobs
import llm
import nas
import retriever
import store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    await llm.embed(["warmup"], prefix="search_query")
    # Load the keyword index once at boot rather than lazily on the first query,
    # so the first user of the day does not pay a cold-start rebuild.
    try:
        await bm25_index.load_or_rebuild()
    except Exception:  # the app is still usable on semantic search alone
        log.warning("BM25 index unavailable at startup", exc_info=True)
    yield
    # Ordered shutdown: stop accepting parse work, then close network clients.
    ingest.shutdown_pool()
    await store.aclose()
    await llm.aclose()


app = FastAPI(title="Geo-Assist", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    ok = await llm.reachable()
    db = await store.health()
    return {
        "status": "ok" if (ok and db["reachable"]) else "degraded",
        "ollama": ok,
        "qdrant": db["reachable"],
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
        "chunks_stored": db["chunks_stored"],
    }


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".csv"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


def _safe_filename(raw: str | None) -> str:
    """Strip directory components from an upload filename."""
    return Path(raw or "unknown").name or "unknown"


def _safe_folder(raw: str) -> str:
    """Strip whitespace, cap at 64 chars, default to General if blank."""
    cleaned = raw.strip()[:64]
    return cleaned or "General"


@app.post("/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    folder: str = Form("General"),
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 100 MB)")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    try:
        result = await ingest.ingest(data, _safe_filename(file.filename), folder=_safe_folder(folder))
    except Exception as e:
        raise HTTPException(500, str(e))
    return result


@app.post("/ingest/bulk")
async def ingest_files_bulk(
    files: list[UploadFile] = File(...),
    folder: str = Form("General"),
):
    if not files:
        raise HTTPException(400, "No files provided")
    items: list[tuple[bytes, str]] = []
    errors: list[str] = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"{f.filename}: unsupported type {ext}")
            continue
        data = await f.read()
        if not data:
            errors.append(f"{f.filename}: empty file")
            continue
        if len(data) > MAX_UPLOAD_BYTES:
            errors.append(f"{f.filename}: file too large (max 100 MB)")
            continue
        items.append((data, _safe_filename(f.filename)))
    if not items:
        raise HTTPException(400, "; ".join(errors) if errors else "No valid files")
    job = jobs.create(total=len(items), errors=errors)
    task = asyncio.create_task(ingest.ingest_many_tracked(items, job, folder=_safe_folder(folder)))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job.id, "total": job.total, "skipped": job.errors}


@app.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "total": job.total,
        "prepared": job.prepared,
        "results": job.results,
        "errors": job.errors,
    }


class NasScanRequest(BaseModel):
    subpath: str = ""
    reingest: bool = False


@app.get("/ingest/nas/health")
async def nas_health():
    """Liveness of the NAS mount, checked on every call rather than at startup.

    Separates "missing", "unreadable", and "empty" because they are
    indistinguishable from the outside and have completely different fixes: a
    missing bind mount, a UID mismatch against the CIFS mount, and a genuinely
    empty share. Also returns the immediate subdirectories, which populate the
    UI's subfolder picker.
    """
    root = config.NAS_ROOT
    if not root or not os.path.isdir(root):
        return {"available": False, "reason": "missing", "root": root, "subfolders": []}
    if not os.access(root, os.R_OK | os.X_OK):
        return {"available": False, "reason": "unreadable", "root": root, "subfolders": []}
    try:
        entries = await asyncio.get_running_loop().run_in_executor(
            None, lambda: sorted(
                e.name for e in os.scandir(root)
                if e.is_dir(follow_symlinks=False)
                and e.name not in nas.EXCLUDED_DIRS
                and not e.name.startswith(".")
            )
        )
    except OSError as exc:
        return {"available": False, "reason": "unreadable", "root": root,
                "subfolders": [], "detail": str(exc)}
    return {"available": True, "reason": "ok", "root": root, "subfolders": entries}


@app.post("/ingest/nas/preview")
async def nas_preview(req: NasScanRequest):
    try:
        return await ingest_nas.preview(req.subpath, reingest=req.reingest)
    except ValueError as exc:
        raise HTTPException(403, str(exc))
    except ingest_nas.NasUnreachable as exc:
        raise HTTPException(503, str(exc))


@app.post("/ingest/nas/scan")
async def nas_scan(req: NasScanRequest):
    # Validate containment before creating a job, so a rejected path surfaces as
    # an immediate 403 rather than as a failed background job the user has to go
    # and read the status of.
    try:
        nas.resolve_subpath(req.subpath)
    except ValueError as exc:
        raise HTTPException(403, str(exc))

    job = jobs.create(total=0)
    task = asyncio.create_task(
        ingest_nas.scan_and_ingest(req.subpath, job, reingest=req.reingest)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job.id, "total": job.total}


@app.get("/documents")
async def list_documents():
    return {"documents": await ingest.list_documents()}


@app.get("/folders")
async def list_folders():
    return {"folders": await ingest.list_folders()}


class MoveFolderRequest(BaseModel):
    folder: str


@app.patch("/documents/{doc_id}/folder")
async def move_document_folder(doc_id: str, req: MoveFolderRequest):
    moved = await ingest.move_document(doc_id, _safe_folder(req.folder))
    if moved == 0:
        raise HTTPException(404, "Document not found")
    return {"moved_chunks": moved}


_DOC_ID_RE = re.compile(r"^[0-9a-f]{16}$")


@app.get("/documents/{doc_id}/file")
async def get_document_file(doc_id: str):
    if not _DOC_ID_RE.match(doc_id):
        raise HTTPException(404, "Document not found")
    # Distinguish a storage-configuration fault from a per-document one. The
    # previous single message asserted "ingested before this feature was added"
    # even when the entire originals directory was missing — which is what
    # happens when the data directory is not persisted, and says nothing about
    # the document.
    if not os.path.isdir(config.ORIGINALS_DIR):
        raise HTTPException(
            404,
            "Source files are not available on this server — the data directory "
            "is not persisted.",
        )
    matches = glob.glob(os.path.join(config.ORIGINALS_DIR, f"{doc_id}.*"))
    if not matches:
        raise HTTPException(
            404,
            "No source file stored for this document. It was ingested before "
            "source files were kept, or the file was removed.",
        )
    return FileResponse(matches[0], headers={"Content-Disposition": "inline"})


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    async with ingest._lock:
        removed = await ingest.delete_document(doc_id)
    if removed == 0:
        raise HTTPException(404, "Document not found")
    return {"removed_chunks": removed}


@app.delete("/documents")
async def clear_all_documents():
    async with ingest._lock:
        removed = await ingest.clear_all_documents()
    return {"removed_chunks": removed}


_background_tasks: set = set()

_sessions: dict[str, list[dict]] = {}
_MAX_HISTORY_TURNS = 3           # last 3 user+assistant pairs
_MAX_HISTORY_MSGS = _MAX_HISTORY_TURNS * 2
_MAX_SESSIONS = 200              # evict oldest when exceeded

# Procedure mode: keyed by session_id, stores {doc_id, filename, steps, step_idx}
_proc_sessions: dict[str, dict] = {}


def _get_history(session_id: str | None) -> list[dict]:
    if not session_id:
        return []
    return _sessions.get(session_id, [])


def _append_history(session_id: str | None, question: str, answer: str) -> None:
    if not session_id:
        return
    hist = _sessions.setdefault(session_id, [])
    hist.append({"role": "user", "content": question})
    hist.append({"role": "assistant", "content": answer})
    if len(hist) > _MAX_HISTORY_MSGS:
        _sessions[session_id] = hist[-_MAX_HISTORY_MSGS:]
    _evict_sessions()


def _evict_sessions() -> None:
    # Dict preserves insertion order (Python 3.7+); pop the oldest entry.
    while len(_sessions) > _MAX_SESSIONS:
        _sessions.pop(next(iter(_sessions)))


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    folder_filter: str | None = None


class RestoreRequest(BaseModel):
    messages: list[dict]


class ProcedureStartRequest(BaseModel):
    doc_id: str


class ProcedureNavigateRequest(BaseModel):
    direction: str = "next"   # "next" or "prev"
    step: int | None = None   # 0-based index for direct jump


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question is empty")
    history = _get_history(req.session_id)
    proc = _proc_sessions.get(req.session_id) if req.session_id else None

    async def generate():
        tokens: list[str] = []
        async for chunk in retriever.answer_stream(req.question, history=history,
                                                   folder_filter=req.folder_filter,
                                                   procedure=proc):
            if "token" in chunk:
                tokens.append(chunk["token"])
            yield f"data: {json.dumps(chunk)}\n\n"
        _append_history(req.session_id, req.question, "".join(tokens))

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat/session/{session_id}/restore")
async def restore_session(session_id: str, req: RestoreRequest):
    _sessions[session_id] = req.messages[-_MAX_HISTORY_MSGS:]
    _evict_sessions()
    return {"restored": len(_sessions[session_id])}


@app.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"cleared": session_id}


# ── procedure mode ────────────────────────────────────────────────────────────

@app.post("/procedure/session/{session_id}/start")
async def procedure_start(session_id: str, req: ProcedureStartRequest):
    hits = await store.get_doc_chunks(req.doc_id)
    if not hits:
        raise HTTPException(404, "Document not found")
    filename = next(iter(hits.values()))[1]["filename"]
    # Sort by page then chunk_index; exclude image/summary chunks (chunk_index < 0).
    # This filter only became effective once chunk_index was actually persisted —
    # it previously defaulted to 0 for every chunk and excluded nothing.
    paired = sorted(hits.items(), key=lambda kv: (kv[1][1]["page"], kv[1][1].get("chunk_index", 0)))
    chunks = [
        text for _, (text, meta, _dist) in paired
        if meta.get("chunk_index", 0) >= 0 and meta.get("chunk_type", "") != "image"
    ]
    steps = await retriever._generate_procedure_steps(chunks)
    _proc_sessions[session_id] = {
        "doc_id": req.doc_id,
        "filename": filename,
        "steps": steps,
        "step_idx": 0,
    }
    return {
        "filename": filename,
        "step_num": 1,
        "total_steps": len(steps),
        "step_text": steps[0] if steps else "",
    }


@app.post("/procedure/session/{session_id}/navigate")
async def procedure_navigate(session_id: str, req: ProcedureNavigateRequest):
    proc = _proc_sessions.get(session_id)
    if not proc:
        raise HTTPException(404, "No active procedure for this session")
    n = len(proc["steps"])
    if req.step is not None:
        idx = max(0, min(req.step, n - 1))
    elif req.direction == "prev":
        idx = max(proc["step_idx"] - 1, 0)
    else:
        idx = min(proc["step_idx"] + 1, n - 1)
    proc["step_idx"] = idx
    return {
        "step_num": idx + 1,
        "total_steps": n,
        "step_text": proc["steps"][idx],
    }


@app.delete("/procedure/session/{session_id}")
async def procedure_end(session_id: str):
    _proc_sessions.pop(session_id, None)
    return {"ended": session_id}


if __name__ == "__main__":
    os.makedirs(config.DATA_DIR, exist_ok=True)
    uvicorn.run("main:app", host="127.0.0.1", port=config.API_PORT, reload=False)
