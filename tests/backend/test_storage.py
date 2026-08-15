"""Tests for the LocalStorage and EncryptedStorage implementations."""

from __future__ import annotations

from pathlib import Path

import pytest

from storage.base import KeyNotFound, Storage, StorageError
from storage.encrypted import EncryptedStorage
from storage.factory import get_storage, reset_storage
from storage.local import LocalStorage


@pytest.fixture()
def storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "datasets")


@pytest.fixture()
def enc_storage(tmp_path: Path) -> EncryptedStorage:
    base = LocalStorage(tmp_path / "datasets")
    return EncryptedStorage(base, "test-passphrase-123")


# ── LocalStorage basic operations ──


class TestLocalStorage:
    def test_put_and_get_bytes(self, storage: LocalStorage) -> None:
        storage.put_bytes("report1/file", b"hello world")
        assert storage.get_bytes("report1/file") == b"hello world"

    def test_put_and_get_text(self, storage: LocalStorage) -> None:
        storage.put_text("report1/output_log", "line1\nline2\n")
        assert storage.get_text("report1/output_log") == "line1\nline2\n"

    def test_get_missing_key_raises(self, storage: LocalStorage) -> None:
        with pytest.raises(KeyNotFound):
            storage.get_bytes("nonexistent/key")

    def test_exists(self, storage: LocalStorage) -> None:
        assert not storage.exists("r/file")
        storage.put_bytes("r/file", b"x")
        assert storage.exists("r/file")

    def test_delete(self, storage: LocalStorage) -> None:
        storage.put_bytes("r/file", b"x")
        storage.delete("r/file")
        assert not storage.exists("r/file")

    def test_delete_missing_is_noop(self, storage: LocalStorage) -> None:
        storage.delete("r/nonexistent")

    def test_delete_prefix(self, storage: LocalStorage) -> None:
        storage.put_bytes("r/file", b"a")
        storage.put_bytes("r/viz/ROC_model", b"b")
        storage.put_bytes("r/report.csv", b"c")
        count = storage.delete_prefix("r")
        assert count == 3
        assert not storage.exists("r/file")

    def test_delete_prefix_nonexistent(self, storage: LocalStorage) -> None:
        assert storage.delete_prefix("nonexistent") == 0

    def test_copy(self, storage: LocalStorage) -> None:
        storage.put_bytes("r/file", b"original")
        storage.copy("r/file", "r/copy")
        assert storage.get_bytes("r/copy") == b"original"
        assert storage.get_bytes("r/file") == b"original"

    def test_copy_missing_source(self, storage: LocalStorage) -> None:
        with pytest.raises(KeyNotFound):
            storage.copy("nonexistent", "dst")

    def test_list_with_prefix(self, storage: LocalStorage) -> None:
        storage.put_bytes("r1/file", b"a")
        storage.put_bytes("r1/viz/ROC_rf", b"b")
        storage.put_bytes("r1/viz/SHAP_rf", b"c")
        storage.put_bytes("r2/file", b"d")
        keys = storage.list("r1")
        assert sorted(keys) == ["r1/file", "r1/viz/ROC_rf", "r1/viz/SHAP_rf"]

    def test_list_empty_prefix(self, storage: LocalStorage) -> None:
        assert storage.list("nonexistent") == []

    def test_list_single_file(self, storage: LocalStorage) -> None:
        storage.put_bytes("r/file", b"a")
        assert storage.list("r/file") == ["r/file"]

    def test_path_traversal_rejected(self, storage: LocalStorage) -> None:
        with pytest.raises(StorageError):
            storage.put_bytes("../etc/passwd", b"x")

    def test_empty_key_rejected(self, storage: LocalStorage) -> None:
        with pytest.raises(StorageError):
            storage.put_bytes("", b"x")

    def test_overwrite(self, storage: LocalStorage) -> None:
        storage.put_bytes("r/file", b"v1")
        storage.put_bytes("r/file", b"v2")
        assert storage.get_bytes("r/file") == b"v2"

    def test_open_read(self, storage: LocalStorage) -> None:
        storage.put_bytes("r/file", b"data")
        with storage.open_read("r/file") as f:
            assert f.read() == b"data"

    def test_open_write_atomic(self, storage: LocalStorage) -> None:
        with storage.open_write("r/file") as f:
            f.write(b"written")
        assert storage.get_bytes("r/file") == b"written"

    def test_open_write_rollback_on_error(self, storage: LocalStorage) -> None:
        try:
            with storage.open_write("r/file") as f:
                f.write(b"partial")
                raise RuntimeError("simulated error")
        except RuntimeError:
            pass
        assert not storage.exists("r/file")

    def test_creates_nested_dirs(self, storage: LocalStorage) -> None:
        storage.put_bytes("r/viz/deeply/nested/ROC_model", b"png")
        assert storage.exists("r/viz/deeply/nested/ROC_model")

    def test_empty_dir_cleanup_on_delete(self, storage: LocalStorage) -> None:
        storage.put_bytes("r/viz/ROC_rf", b"x")
        storage.delete("r/viz/ROC_rf")
        # Parent dirs should be cleaned up (but not the root)
        assert not (storage.root / "r" / "viz").exists()


class TestStorageCSV:
    def test_write_and_read_csv(self, storage: LocalStorage) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        storage.write_csv("r/file", df, index=False)
        result = storage.read_csv("r/file")
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3


class TestEncryptedStorage:
    def test_put_and_get_bytes(self, enc_storage: EncryptedStorage) -> None:
        enc_storage.put_bytes("r/file", b"secret data")
        assert enc_storage.get_bytes("r/file") == b"secret data"

    def test_encrypted_on_disk(self, enc_storage: EncryptedStorage) -> None:
        """Verify the plaintext is NOT stored on disk."""
        enc_storage.put_bytes("r/file", b"plaintext secret")
        raw = enc_storage._base.get_bytes("r/file")
        assert b"plaintext secret" not in raw

    def test_put_and_get_text(self, enc_storage: EncryptedStorage) -> None:
        enc_storage.put_text("r/log", "log line")
        assert enc_storage.get_text("r/log") == "log line"

    def test_exists(self, enc_storage: EncryptedStorage) -> None:
        assert not enc_storage.exists("r/file")
        enc_storage.put_bytes("r/file", b"x")
        assert enc_storage.exists("r/file")

    def test_delete(self, enc_storage: EncryptedStorage) -> None:
        enc_storage.put_bytes("r/file", b"x")
        enc_storage.delete("r/file")
        assert not enc_storage.exists("r/file")

    def test_copy(self, enc_storage: EncryptedStorage) -> None:
        enc_storage.put_bytes("r/src", b"data")
        enc_storage.copy("r/src", "r/dst")
        assert enc_storage.get_bytes("r/dst") == b"data"

    def test_list(self, enc_storage: EncryptedStorage) -> None:
        enc_storage.put_bytes("r/file", b"a")
        enc_storage.put_bytes("r/viz/x", b"b")
        keys = enc_storage.list("r")
        assert sorted(keys) == ["r/file", "r/viz/x"]

    def test_wrong_passphrase_fails(self, tmp_path: Path) -> None:
        base = LocalStorage(tmp_path / "ds")
        enc1 = EncryptedStorage(base, "correct-pass")
        enc1.put_bytes("r/file", b"secret")
        enc2 = EncryptedStorage(base, "wrong-pass")
        from cryptography.exceptions import InvalidTag

        with pytest.raises((InvalidTag, ValueError)):
            enc2.get_bytes("r/file")

    def test_csv_roundtrip(self, enc_storage: EncryptedStorage) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"col": [1, 2]})
        enc_storage.write_csv("r/file", df, index=False)
        result = enc_storage.read_csv("r/file")
        assert list(result["col"]) == [1, 2]


class TestStorageFactory:
    def test_get_storage_local(self, tmp_data_dir: None, settings: object) -> None:  # noqa: ARG002
        reset_storage()
        storage = get_storage()
        assert isinstance(storage, LocalStorage)

    def test_get_storage_caches(self, tmp_data_dir: None, settings: object) -> None:  # noqa: ARG002
        reset_storage()
        s1 = get_storage()
        s2 = get_storage()
        assert s1 is s2

    def test_storage_protocol_compliance(self, storage: LocalStorage) -> None:
        assert isinstance(storage, Storage)
