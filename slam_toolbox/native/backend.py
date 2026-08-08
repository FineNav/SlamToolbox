from __future__ import annotations

from typing import Literal

BackendName = Literal["auto", "native", "python"]

try:
    from slam_toolbox import _native
except ImportError as exc:
    _native = None
    _NATIVE_IMPORT_ERROR: ImportError | None = exc
else:
    _NATIVE_IMPORT_ERROR = None


class NativeBackendUnavailable(RuntimeError):
    pass


def native_available() -> bool:
    return _native is not None


def native_api_version() -> int | None:
    return int(_native.api_version) if _native is not None else None


def resolve_backend(requested: BackendName) -> Literal["native", "python"]:
    if requested not in ("auto", "native", "python"):
        raise ValueError(f"unsupported compute backend: {requested}")
    if requested == "python":
        return "python"
    if native_available():
        return "native"
    if requested == "native":
        detail = f": {_NATIVE_IMPORT_ERROR}" if _NATIVE_IMPORT_ERROR is not None else ""
        raise NativeBackendUnavailable(f"slam_toolbox native extension is unavailable{detail}")
    return "python"


def native_module():
    if _native is None:
        raise NativeBackendUnavailable("slam_toolbox native extension is unavailable")
    return _native
