"""
Local authentication module — no external dependencies.

Passwords: PBKDF2-HMAC-SHA256 (200k iterations), random 16-byte salt.
Tokens: HMAC-SHA256 signed payloads (same structure as HS256 JWT, no library needed).
         Carry an `iat` (issued-at) claim so revoked tokens can be rejected.
Users: stored in data/users.json alongside ChromaDB.
Secret: auto-generated on first start, stored in data/secret.key (chmod 600).
Revocations: stored in data/revocations.json — survives user deletion.
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path

import config

log = logging.getLogger(__name__)

_USERS_PATH = Path(config.DATA_DIR) / "users.json"
_SECRET_PATH = Path(config.DATA_DIR) / "secret.key"
_REVOCATIONS_PATH = Path(config.DATA_DIR) / "revocations.json"
_PBKDF2_ITERS = 200_000
_TOKEN_TTL_HOURS = 24 * 7   # 7 days


# ── signing secret ────────────────────────────────────────────────────────────

_secret_cache: bytes | None = None


def _secret() -> bytes:
    global _secret_cache
    if _secret_cache is None:
        if _SECRET_PATH.exists():
            _secret_cache = _SECRET_PATH.read_bytes()
        else:
            _SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
            _secret_cache = os.urandom(32)
            _SECRET_PATH.write_bytes(_secret_cache)
            _SECRET_PATH.chmod(0o600)
    return _secret_cache


# ── passwords ─────────────────────────────────────────────────────────────────

def hash_password(password: str) -> tuple[str, str]:
    """Return (hash_hex, salt_hex)."""
    salt = os.urandom(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERS)
    return h.hex(), salt.hex()


def verify_password(password: str, hash_hex: str, salt_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERS)
    return hmac.compare_digest(h.hex(), hash_hex)


# ── revocations ───────────────────────────────────────────────────────────────

def _load_revocations() -> dict:
    if not _REVOCATIONS_PATH.exists():
        return {}
    try:
        return json.loads(_REVOCATIONS_PATH.read_text())
    except Exception as exc:
        log.warning("revocations.json unreadable (%s); treating as empty — revoked tokens may be accepted", exc)
        return {}


def _save_revocations(revs: dict) -> None:
    _REVOCATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _REVOCATIONS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(revs, indent=2))
    os.replace(tmp, _REVOCATIONS_PATH)


def _revoke(username: str) -> None:
    """Record that all tokens for username issued at or before now are invalid."""
    revs = _load_revocations()
    revs[username] = time.time()
    _save_revocations(revs)


# ── tokens ────────────────────────────────────────────────────────────────────

def create_token(username: str, role: str) -> str:
    now = time.time()
    payload = json.dumps({"u": username, "r": role, "iat": now,
                          "exp": now + _TOKEN_TTL_HOURS * 3600})
    p64 = base64.urlsafe_b64encode(payload.encode()).decode()
    sig = hmac.new(_secret(), p64.encode(), hashlib.sha256).hexdigest()
    return f"{p64}.{sig}"


def verify_token(token: str) -> dict:
    """Return {username, role} or raise ValueError."""
    try:
        p64, sig = token.rsplit(".", 1)
    except ValueError:
        raise ValueError("Malformed token")
    expected = hmac.new(_secret(), p64.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise ValueError("Invalid token signature")
    try:
        payload = json.loads(base64.urlsafe_b64decode(p64 + "=="))
    except Exception:
        raise ValueError("Malformed token payload")
    if payload["exp"] < time.time():
        raise ValueError("Token expired")
    iat = payload.get("iat", 0)
    # Revocation check: covers role-update and deleted-not-recreated cases.
    revs = _load_revocations()
    if payload["u"] in revs and iat <= revs[payload["u"]]:
        raise ValueError("Token has been revoked")
    # created_at check: rejects pre-deletion tokens when a user is recreated
    # (revocation is cleared on recreation, so we fall back to created_at).
    users = _load()
    user = users.get(payload["u"])
    if user and iat < user.get("created_at", 0):
        raise ValueError("Token has been revoked")
    return {"username": payload["u"], "role": payload["r"]}


# ── user store ────────────────────────────────────────────────────────────────

def _load() -> dict:
    if not _USERS_PATH.exists():
        return {}
    try:
        return json.loads(_USERS_PATH.read_text())
    except Exception as exc:
        log.error("users.json is corrupt (%s) — authentication unavailable", exc)
        raise


def _save(users: dict) -> None:
    _USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _USERS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(users, indent=2))
    os.replace(tmp, _USERS_PATH)


def has_users() -> bool:
    return bool(_load())


def create_user(username: str, password: str, role: str = "user") -> dict:
    users = _load()
    if username in users:
        raise ValueError(f"User '{username}' already exists")
    # Clear any leftover revocation so the fresh account isn't blocked.
    revs = _load_revocations()
    if username in revs:
        del revs[username]
        _save_revocations(revs)
    h, s = hash_password(password)
    users[username] = {"hash": h, "salt": s, "role": role, "created_at": time.time()}
    _save(users)
    return {"username": username, "role": role}


def delete_user(username: str) -> bool:
    users = _load()
    if username not in users:
        return False
    _revoke(username)   # invalidate all existing tokens before removing the record
    del users[username]
    _save(users)
    return True


def update_role(username: str, role: str) -> bool:
    users = _load()
    if username not in users:
        return False
    users[username]["role"] = role
    _save(users)
    _revoke(username)   # old tokens carry the old role — force re-login
    return True


def list_users() -> list[dict]:
    return [{"username": u, "role": v["role"]} for u, v in _load().items()]


def authenticate(username: str, password: str) -> dict | None:
    """Return {username, role} if credentials valid, else None."""
    users = _load()
    user = users.get(username)
    if not user:
        return None
    if not verify_password(password, user["hash"], user["salt"]):
        return None
    return {"username": username, "role": user["role"]}
