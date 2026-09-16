from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(paths: Iterable[str | Path]) -> dict[str, str]:
    resolved = sorted((Path(path) for path in paths), key=lambda item: item.as_posix())
    return {path.as_posix(): sha256_file(path) for path in resolved}


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def require_matching_checksum(path: str | Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f'checksum mismatch for {path}: expected {expected}, got {actual}')
