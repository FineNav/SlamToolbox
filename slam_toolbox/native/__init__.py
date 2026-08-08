"""Backend selection for optional C++ algorithm implementations."""

from .backend import NativeBackendUnavailable, native_api_version, native_available, resolve_backend

__all__ = [
    "NativeBackendUnavailable",
    "native_api_version",
    "native_available",
    "resolve_backend",
]
