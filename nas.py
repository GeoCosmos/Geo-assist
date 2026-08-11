"""
Walker for a mounted NAS document share.

Filesystem policy only — what counts as a document, what counts as junk, and what
counts as inside the share. Knows nothing about ingestion, HTTP, or the manifest,
so it can be tested against a plain tmp_path tree.
"""
import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path

import config

SUPPORTED_EXTS = {".pdf", ".docx", ".pptx", ".txt", ".csv"}
MAX_FILE_BYTES = 100 * 1024 * 1024  # mirrors main.MAX_UPLOAD_BYTES

# QNAP scatters @eaDir (thumbnail/metadata sidecars) through every share — one per
# directory holding media, each containing files named after the originals. Left
# in, they would roughly double the corpus with unreadable stubs. @Recycle is the
# NAS trash. The rest is Windows/macOS/Office debris.
EXCLUDED_DIRS = {
    "@eaDir", "@Recycle", "#recycle", "$RECYCLE.BIN", "System Volume Information",
}
EXCLUDED_FILE_PATTERNS = ("~$*", "Thumbs.db", "desktop.ini", ".DS_Store")


@dataclass(frozen=True)
class NasFile:
    abs_path: str
    relpath: str      # POSIX-style, relative to the scan root
    size: int
    mtime: float
    folder: str       # derived from relpath's parent; "General" at the root
    ext: str


def resolve_subpath(subpath: str) -> Path:
    """Resolve a *relative* subpath under NAS_ROOT.

    Absolute paths are rejected outright rather than silently reinterpreted: the
    mount point is fixed by the container, so the only legitimate input is a
    subfolder name. Resolution happens before the containment check, so symlinks
    pointing outside the share are caught too.
    """
    if os.path.isabs(subpath or ""):
        raise ValueError("absolute paths are not accepted")
    root = Path(config.NAS_ROOT).resolve()
    cleaned = (subpath or "").strip().lstrip("/")
    target = (root / cleaned).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes the NAS root: {subpath!r}")
    return target


def folder_for(relpath: str) -> str:
    """Map a file's directory onto the existing `folder` metadata field."""
    parent = str(Path(relpath).parent).replace(os.sep, "/")
    if parent in (".", "", "/"):
        return "General"
    return parent[:64]


def _excluded_file(name: str) -> bool:
    if name.startswith("."):
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDED_FILE_PATTERNS)


def scan(root: Path, base: Path | None = None) -> tuple[list[NasFile], dict[str, int]]:
    """Walk `root`, returning ingestable files and counts of what was passed over.

    `base` is what relpaths are measured against, defaulting to `root`. Callers
    scanning a subfolder must pass the share root, or the same file gets a
    different identity depending on scan scope: scanning "Manuals" would key
    Manuals/alpha.txt as "alpha.txt" and file it under folder "General", while a
    full scan keys it "Manuals/alpha.txt" under folder "Manuals". That breaks the
    manifest in both directions — every file looks new after switching scope, and
    a root-level file sharing a basename collides with the subfolder one.

    Blocking: `stat` over CIFS is a network round trip. Callers must run this in
    an executor, never on the event loop.
    """
    root = Path(root)
    base = Path(base) if base is not None else root
    if not root.is_dir():
        raise FileNotFoundError(f"NAS path is not a directory: {root}")

    files: list[NasFile] = []
    counts = {"unsupported": 0, "oversized": 0, "excluded": 0}

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune in place so os.walk never descends into junk directories.
        keep = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith(".")]
        counts["excluded"] += len(dirnames) - len(keep)
        dirnames[:] = keep

        for name in filenames:
            if _excluded_file(name):
                counts["excluded"] += 1
                continue
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTS:
                counts["unsupported"] += 1
                continue
            abs_path = os.path.join(dirpath, name)
            try:
                st = os.stat(abs_path)
            except OSError:
                # A file that vanished or is unreadable is not fatal to the walk.
                counts["excluded"] += 1
                continue
            if st.st_size > MAX_FILE_BYTES:
                counts["oversized"] += 1
                continue
            relpath = os.path.relpath(abs_path, base).replace(os.sep, "/")
            files.append(NasFile(
                abs_path=abs_path,
                relpath=relpath,
                size=st.st_size,
                mtime=st.st_mtime,
                folder=folder_for(relpath),
                ext=ext,
            ))

    return files, counts
