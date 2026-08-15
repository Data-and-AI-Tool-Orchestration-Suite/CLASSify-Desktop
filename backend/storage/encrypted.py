"""Optional AES-GCM encryption wrapper for LocalStorage.

When ``settings.encryption_enabled`` is true, all file bodies are
encrypted at rest with AES-256-GCM.  The passphrase is stored in the
OS keyring (set via the first-run wizard) and never persisted to disk.

NOTE: Full hardening (Argon2id KDF, per-report keys, memory zeroization)
is a post-v1 roadmap item (R5 in ROADMAP.md).  This implementation uses
PBKDF2-HMAC-SHA256 for key derivation, which is adequate for v1.
"""

from __future__ import annotations

import hashlib
import io
import os
from typing import IO, Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from storage.local import LocalStorage

# 12-byte nonce for AES-GCM (NIST recommendation)
_NONCE_SIZE = 12
# 600k iterations for PBKDF2 (OWASP 2023 recommendation for SHA-256)
_PBKDF2_ITERATIONS = 600_000
_KEY_SIZE = 32  # AES-256


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a 256-bit key from a passphrase + salt using PBKDF2."""
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, _PBKDF2_ITERATIONS, _KEY_SIZE)


class EncryptedStorage:
    """Wraps a LocalStorage, encrypting/decrypting all bytes transparently.

    The underlying LocalStorage still manages file paths and listing;
    this layer only transforms the byte content.
    """

    def __init__(self, base: LocalStorage, passphrase: str) -> None:
        self._base = base
        self._passphrase = passphrase

    @property
    def root(self) -> Any:
        return self._base.root

    def _enc(self, data: bytes) -> bytes:
        """Encrypt data with AES-GCM.  Returns: salt(16) + nonce(12) + ciphertext."""
        salt = os.urandom(16)
        key = _derive_key(self._passphrase, salt)
        nonce = os.urandom(_NONCE_SIZE)
        ct = AESGCM(key).encrypt(nonce, data, None)
        return salt + nonce + ct

    def _dec(self, blob: bytes) -> bytes:
        """Decrypt an AES-GCM blob.  Expects: salt(16) + nonce(12) + ciphertext."""
        if len(blob) < 16 + _NONCE_SIZE:
            raise ValueError("Encrypted blob too short")
        salt = blob[:16]
        nonce = blob[16 : 16 + _NONCE_SIZE]
        ct = blob[16 + _NONCE_SIZE :]
        key = _derive_key(self._passphrase, salt)
        return AESGCM(key).decrypt(nonce, ct, None)

    def get_bytes(self, key: str) -> bytes:
        return self._dec(self._base.get_bytes(key))

    def put_bytes(self, key: str, data: bytes) -> None:
        self._base.put_bytes(key, self._enc(data))

    def get_text(self, key: str, encoding: str = "utf-8") -> str:
        return self.get_bytes(key).decode(encoding)

    def put_text(self, key: str, text: str, encoding: str = "utf-8") -> None:
        self.put_bytes(key, text.encode(encoding))

    def read_csv(self, key: str, **kwargs: Any) -> Any:
        import pandas as pd

        data = self.get_bytes(key)
        return pd.read_csv(io.BytesIO(data), **kwargs)

    def write_csv(self, key: str, df: Any, **kwargs: Any) -> None:
        buf = io.StringIO()
        df.to_csv(buf, **kwargs)
        self.put_text(key, buf.getvalue())

    def list(self, prefix: str) -> list[str]:
        return self._base.list(prefix)

    def exists(self, key: str) -> bool:
        return self._base.exists(key)

    def delete(self, key: str) -> None:
        self._base.delete(key)

    def delete_prefix(self, prefix: str) -> int:
        return self._base.delete_prefix(prefix)

    def copy(self, src: str, dst: str) -> None:
        self.put_bytes(dst, self.get_bytes(src))

    def open_read(self, key: str) -> IO[bytes]:
        data = self.get_bytes(key)
        return io.BytesIO(data)

    def open_write(self, key: str) -> IO[bytes]:
        return _EncryptedWriter(self, key)  # type: ignore[return-value]


class _EncryptedWriter:
    """Buffered writer that encrypts on close (small files only — fine for our use)."""

    def __init__(self, storage: EncryptedStorage, key: str) -> None:
        self._storage = storage
        self._key = key
        self._buf = bytearray()
        self._closed = False

    def write(self, data: bytes) -> int:
        self._buf.extend(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._storage.put_bytes(self._key, bytes(self._buf))
        self._buf.clear()

    def __enter__(self) -> _EncryptedWriter:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self._closed = True
