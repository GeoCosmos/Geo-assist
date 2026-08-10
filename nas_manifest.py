"""
Local record of which NAS files have already been ingested.

Purely a cache, never a source of truth. Deleting the database causes a full
re-scan that re-derives the same state — files already in the store are skipped
by content hash — it does not cause duplication.

Its job is to make a re-scan cheap. `doc_id` is a hash of file *content*, so
without this the only way to know whether a file has been seen is to read every
byte of it. Over CIFS, across thousands of files, that is the difference between
a scan measured in seconds and one measured in hours.
"""
import sqlite3
import time

# CIFS timestamps can jitter in the low-order digits between stats of an
# unmodified file. Comparing mtimes for exact equality would then mark the entire
# corpus as changed on every scan, whose only symptom is a very slow re-ingest.
MTIME_TOLERANCE = 1e-3


class Manifest:
    def __init__(self, path: str):
        self._path = path
        # check_same_thread=False: the walk and the ingest loop touch this from
        # different executor threads, serialised by the ingest lock upstream.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS seen (
              relpath    TEXT PRIMARY KEY,
              size       INTEGER NOT NULL,
              mtime      REAL    NOT NULL,
              doc_id     TEXT,
              status     TEXT    NOT NULL,
              scanned_at REAL    NOT NULL
            )
        """)
        self._conn.commit()

    def known(self, relpath: str, size: int, mtime: float) -> bool:
        """True if this exact file (path, size, mtime) has been processed before.

        Files recorded with status="error" count as known: a permanently corrupt
        PDF should not be re-attempted on every scan. Editing the file changes
        its mtime and makes it unknown again.
        """
        row = self._conn.execute(
            "SELECT size, mtime FROM seen WHERE relpath = ?", (relpath,)
        ).fetchone()
        if row is None:
            return False
        return row[0] == size and abs(row[1] - mtime) <= MTIME_TOLERANCE

    def record(self, relpath: str, size: int, mtime: float,
               doc_id: str | None, status: str) -> None:
        self.record_many([(relpath, size, mtime, doc_id, status)])

    def record_many(self, rows: list[tuple]) -> None:
        """Commit a batch of rows in one transaction.

        Called after each write flush, which is what makes an interrupted job
        resumable: re-running the scan skips whatever already committed.
        """
        now = time.time()
        self._conn.executemany(
            "INSERT OR REPLACE INTO seen "
            "(relpath, size, mtime, doc_id, status, scanned_at) VALUES (?,?,?,?,?,?)",
            [(r[0], r[1], r[2], r[3], r[4], now) for r in rows],
        )
        self._conn.commit()

    def forget(self, relpath: str) -> None:
        self._conn.execute("DELETE FROM seen WHERE relpath = ?", (relpath,))
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
