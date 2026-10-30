"""One storage layer for a local folder and a GCS bucket.

fsspec picks the filesystem from the URI: "./data" is the local disk,
"gs://bucket/prefix" is Google Cloud Storage (through gcsfs). The pipeline
code only uses relative paths like "clean/BALRD/observations.parquet", so the
same code runs on a laptop, in tests (temp folder) and in GitHub Actions.
"""
from __future__ import annotations

import io
import json

import fsspec
import pandas as pd
from fsspec.implementations.local import LocalFileSystem


class Storage:
    def __init__(self, uri: str):
        self.uri = uri
        self.fs, self.root = fsspec.core.url_to_fs(uri)
        self.is_local = isinstance(self.fs, LocalFileSystem)

    def _full(self, rel: str) -> str:
        return f"{self.root.rstrip('/')}/{rel}"

    def exists(self, rel: str) -> bool:
        return self.fs.exists(self._full(rel))

    def list(self, prefix: str) -> list[str]:
        """Relative paths of all files below `prefix` (empty list if none)."""
        full = self._full(prefix)
        if not self.fs.exists(full):
            return []
        root = self.root.rstrip("/") + "/"
        return sorted(p.split(root, 1)[-1] for p in self.fs.find(full))

    # ------------------------------------------------------------------ bytes
    def read_bytes(self, rel: str) -> bytes:
        with self.fs.open(self._full(rel), "rb") as f:
            return f.read()

    def write_bytes(self, rel: str, data: bytes) -> None:
        """Write atomically: readers never see a half-written file.

        Locally we write a temp file and rename it (rename is atomic on one
        disk). On GCS a single object upload is already all-or-nothing.
        """
        full = self._full(rel)
        if self.is_local:
            self.fs.makedirs(full.rsplit("/", 1)[0], exist_ok=True)
            tmp = full + ".tmp"
            with self.fs.open(tmp, "wb") as f:
                f.write(data)
            self.fs.mv(tmp, full)
        else:
            with self.fs.open(full, "wb") as f:
                f.write(data)

    # ------------------------------------------------------------------ helpers
    def read_json(self, rel: str):
        return json.loads(self.read_bytes(rel))

    def write_json(self, rel: str, obj) -> None:
        self.write_bytes(rel, json.dumps(obj, indent=2, default=str).encode())

    def read_parquet(self, rel: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self.read_bytes(rel)))

    def write_parquet(self, rel: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf)
        self.write_bytes(rel, buf.getvalue())

    def append_line(self, rel: str, line: str) -> None:
        """Append one line to a text file. GCS objects cannot be appended to,
        so we read, add and rewrite (fine for a small log file)."""
        old = self.read_bytes(rel) if self.exists(rel) else b""
        self.write_bytes(rel, old + line.rstrip("\n").encode() + b"\n")
