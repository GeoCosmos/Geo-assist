"""Tests for the incremental-scan manifest."""
import pytest

from nas_manifest import Manifest


@pytest.fixture
def manifest(tmp_path):
    m = Manifest(str(tmp_path / "manifest.db"))
    yield m
    m.close()


def test_unknown_file_is_not_known(manifest):
    assert manifest.known("a.pdf", 100, 1000.0) is False


def test_recorded_file_is_known(manifest):
    manifest.record("a.pdf", 100, 1000.0, "abc123", "ok")
    assert manifest.known("a.pdf", 100, 1000.0) is True


def test_changed_mtime_is_not_known(manifest):
    manifest.record("a.pdf", 100, 1000.0, "abc123", "ok")
    assert manifest.known("a.pdf", 100, 2000.0) is False


def test_changed_size_is_not_known(manifest):
    manifest.record("a.pdf", 100, 1000.0, "abc123", "ok")
    assert manifest.known("a.pdf", 200, 1000.0) is False


def test_record_replaces_previous_row(manifest):
    manifest.record("a.pdf", 100, 1000.0, "abc123", "ok")
    manifest.record("a.pdf", 200, 2000.0, "def456", "ok")
    assert manifest.count() == 1
    assert manifest.known("a.pdf", 200, 2000.0) is True


def test_errored_file_is_known_so_it_is_not_retried_forever(manifest):
    manifest.record("bad.pdf", 100, 1000.0, None, "error")
    assert manifest.known("bad.pdf", 100, 1000.0) is True


def test_forget_makes_file_unknown_again(manifest):
    manifest.record("a.pdf", 100, 1000.0, "abc123", "ok")
    manifest.forget("a.pdf")
    assert manifest.known("a.pdf", 100, 1000.0) is False


def test_record_many_is_atomic_batch(manifest):
    manifest.record_many([
        ("a.pdf", 1, 1.0, "id1", "ok"),
        ("b.pdf", 2, 2.0, "id2", "ok"),
    ])
    assert manifest.count() == 2


def test_manifest_persists_across_instances(tmp_path):
    path = str(tmp_path / "m.db")
    m1 = Manifest(path)
    m1.record("a.pdf", 100, 1000.0, "abc123", "ok")
    m1.close()
    m2 = Manifest(path)
    assert m2.known("a.pdf", 100, 1000.0) is True
    m2.close()


def test_mtime_comparison_tolerates_float_noise(manifest):
    """CIFS can return mtimes differing in the sub-microsecond digits between
    stats of an unmodified file. Exact float equality would re-ingest the whole
    corpus every scan, with no symptom except hours of wasted embedding."""
    manifest.record("a.pdf", 100, 1000.0000001, "abc123", "ok")
    assert manifest.known("a.pdf", 100, 1000.0000002) is True
