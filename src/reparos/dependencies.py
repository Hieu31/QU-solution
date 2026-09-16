from __future__ import annotations


def require(module: str, extra: str):
    try:
        return __import__(module)
    except ImportError as error:
        raise RuntimeError(
            f'{module} is required for this command; install with: uv sync --extra {extra}'
        ) from error
