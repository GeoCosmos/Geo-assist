import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import ingest
import jobs
import llm
import retriever

if config.AUTH_ENABLED:
    import auth

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await llm.embed(["warmup"], prefix="search_query")
    yield


app = FastAPI(title="Geo-Assist", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    ok = await llm.reachable()
    col = ingest._db()
    return {
        "status": "ok" if ok else "degraded",
        "ollama": ok,
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
        "vision_model": config.VISION_MODEL or None,
        "chunks_stored": col.count(),
    }


# ── auth dependency ───────────────────────────────────────────────────────────

def get_current_user(authorization: str | None = Header(None)) -> dict | None:
    """Return current user dict or None. Raises 401 if auth is enabled and token is missing/invalid."""
    if not config.AUTH_ENABLED:
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authentication required")
    try:
        return auth.verify_token(authorization[7:])
    except ValueError as e:
        raise HTTPException(401, str(e))


def require_admin(current_user: dict | None = Depends(get_current_user)) -> dict:
    if not current_user or current_user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    return current_user


# ── auth endpoints ────────────────────────────────────────────────────────────

_login_failures: dict[str, list[float]] = {}  # username → timestamps of recent failures
_MAX_LOGIN_FAILURES = 10
_LOCKOUT_SECONDS = 60


def _check_login_rate(username: str) -> None:
    now = time.time()
    recent = [t for t in _login_failures.get(username, []) if now - t < _LOCKOUT_SECONDS]
    _login_failures[username] = recent
    if len(recent) >= _MAX_LOGIN_FAILURES:
        raise HTTPException(429, f"Too many failed attempts. Try again in {_LOCKOUT_SECONDS}s.")


def _record_login_failure(username: str) -> None:
    _login_failures.setdefault(username, []).append(time.time())


def _clear_login_failures(username: str) -> None:
    _login_failures.pop(username, None)


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


class UpdateRoleRequest(BaseModel):
    role: str


@app.post("/auth/setup")
async def auth_setup(req: LoginRequest):
    """Create the first admin user. Returns 409 once any user exists."""
    if not config.AUTH_ENABLED:
        raise HTTPException(404, "Auth is disabled")
    if auth.has_users():
        raise HTTPException(409, "Setup already complete — use /auth/users to add more users")
    user = auth.create_user(req.username, req.password, role="admin")
    token = auth.create_token(user["username"], user["role"])
    return {"token": token, "username": user["username"], "role": user["role"]}


@app.post("/auth/login")
async def auth_login(req: LoginRequest):
    if not config.AUTH_ENABLED:
        raise HTTPException(404, "Auth is disabled")
    _check_login_rate(req.username)
    user = auth.authenticate(req.username, req.password)
    if not user:
        _record_login_failure(req.username)
        raise HTTPException(401, "Invalid username or password")
    _clear_login_failures(req.username)
    token = auth.create_token(user["username"], user["role"])
    return {"token": token, "username": user["username"], "role": user["role"]}


@app.get("/auth/me")
async def auth_me(current_user: dict | None = Depends(get_current_user)):
    if not config.AUTH_ENABLED:
        return {"auth_enabled": False}
    return {"auth_enabled": True, **current_user}


@app.get("/auth/users")
async def auth_list_users(_admin: dict = Depends(require_admin)):
    return {"users": auth.list_users()}


@app.post("/auth/users")
async def auth_create_user(req: CreateUserRequest, _admin: dict = Depends(require_admin)):
    try:
        user = auth.create_user(req.username, req.password, req.role)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return user


@app.delete("/auth/users/{username}")
async def auth_delete_user(username: str, _admin: dict = Depends(require_admin)):
    if not auth.delete_user(username):
        raise HTTPException(404, "User not found")
    return {"deleted": username}


@app.patch("/auth/users/{username}/role")
async def auth_update_role(username: str, req: UpdateRoleRequest, _admin: dict = Depends(require_admin)):
    if not auth.update_role(username, req.role):
        raise HTTPException(404, "User not found")
    return {"username": username, "role": req.role}


ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".txt", ".csv",
    # code files
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php",
    ".swift", ".kt", ".scala", ".r",
    ".sh", ".ps1", ".bat",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".env",
    ".html", ".css", ".xml", ".sql", ".md",
    # audio / video — transcribed via faster-whisper (requires ffmpeg on PATH)
    ".mp3", ".mp4", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".mkv", ".mov", ".avi",
}
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
    access: str = Form("public"),
    current_user: dict | None = Depends(get_current_user),
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 100 MB)")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    owner = current_user["username"] if current_user else ""
    try:
        result = await ingest.ingest(data, _safe_filename(file.filename),
                                     folder=_safe_folder(folder), access=access, owner=owner)
    except RuntimeError as e:
        msg = str(e)
        raise HTTPException(501 if "faster-whisper" in msg else 500, msg)
    except Exception as e:
        raise HTTPException(500, str(e))
    return result


@app.post("/ingest/bulk")
async def ingest_files_bulk(
    files: list[UploadFile] = File(...),
    folder: str = Form("General"),
    access: str = Form("public"),
    current_user: dict | None = Depends(get_current_user),
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
    owner = current_user["username"] if current_user else ""
    job = jobs.create(total=len(items), errors=errors)
    task = asyncio.create_task(ingest.ingest_many_tracked(items, job, folder=_safe_folder(folder), access=access, owner=owner))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job.id, "total": job.total, "skipped": job.errors}


@app.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str, _: dict | None = Depends(get_current_user)):
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


@app.get("/documents")
async def list_documents(_: dict | None = Depends(get_current_user)):
    return {"documents": ingest.list_documents()}


@app.get("/folders")
async def list_folders(_: dict | None = Depends(get_current_user)):
    return {"folders": ingest.list_folders()}


class MoveFolderRequest(BaseModel):
    folder: str


@app.patch("/documents/{doc_id}/folder")
async def move_document_folder(doc_id: str, req: MoveFolderRequest,
                               _: dict | None = Depends(get_current_user)):
    moved = ingest.move_document(doc_id, _safe_folder(req.folder))
    if moved == 0:
        raise HTTPException(404, "Document not found")
    return {"moved_chunks": moved}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, _: dict | None = Depends(get_current_user)):
    removed = ingest.delete_document(doc_id)
    if removed == 0:
        raise HTTPException(404, "Document not found")
    return {"removed_chunks": removed}


@app.delete("/documents")
async def clear_all_documents(_: dict | None = Depends(get_current_user)):
    async with ingest._lock:
        removed = ingest.clear_all_documents()
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
async def chat_stream(req: ChatRequest, current_user: dict | None = Depends(get_current_user)):
    if not req.question.strip():
        raise HTTPException(400, "Question is empty")
    history = _get_history(req.session_id)
    proc = _proc_sessions.get(req.session_id) if req.session_id else None

    async def generate():
        tokens: list[str] = []
        async for chunk in retriever.answer_stream(req.question, history=history,
                                                   folder_filter=req.folder_filter,
                                                   current_user=current_user,
                                                   procedure=proc):
            if "token" in chunk:
                tokens.append(chunk["token"])
            yield f"data: {json.dumps(chunk)}\n\n"
        _append_history(req.session_id, req.question, "".join(tokens))

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat/session/{session_id}/restore")
async def restore_session(session_id: str, req: RestoreRequest,
                          _: dict | None = Depends(get_current_user)):
    _sessions[session_id] = req.messages[-_MAX_HISTORY_MSGS:]
    _evict_sessions()
    return {"restored": len(_sessions[session_id])}


@app.delete("/chat/session/{session_id}")
async def clear_session(session_id: str, _: dict | None = Depends(get_current_user)):
    _sessions.pop(session_id, None)
    return {"cleared": session_id}


# ── procedure mode ────────────────────────────────────────────────────────────

@app.post("/procedure/session/{session_id}/start")
async def procedure_start(session_id: str, req: ProcedureStartRequest,
                          _: dict | None = Depends(get_current_user)):
    col = ingest._db()
    result = col.get(where={"doc_id": req.doc_id}, include=["documents", "metadatas"])
    if not result["ids"]:
        raise HTTPException(404, "Document not found")
    filename = result["metadatas"][0]["filename"]
    # Sort by page then chunk_index; exclude image/summary chunks (chunk_index < 0)
    paired = sorted(
        zip(result["ids"], result["documents"], result["metadatas"]),
        key=lambda x: (x[2]["page"], x[2].get("chunk_index", 0)),
    )
    chunks = [
        doc for _, doc, meta in paired
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
async def procedure_navigate(session_id: str, req: ProcedureNavigateRequest,
                             _: dict | None = Depends(get_current_user)):
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
async def procedure_end(session_id: str, _: dict | None = Depends(get_current_user)):
    _proc_sessions.pop(session_id, None)
    return {"ended": session_id}


if __name__ == "__main__":
    os.makedirs(config.DATA_DIR, exist_ok=True)
    uvicorn.run("main:app", host="127.0.0.1", port=config.API_PORT, reload=False)
