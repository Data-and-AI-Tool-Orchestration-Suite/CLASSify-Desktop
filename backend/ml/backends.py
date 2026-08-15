"""Addon registry — guards imports of optional dependencies (TabPFN, SDV).

Both TabPFN and SDV require torch, which is NOT in the base installer.
The engine uses ``is_available()`` to check before referencing these
modules, and ``require()`` to produce a clean error message when a user
tries to use a feature that needs an addon that isn't installed.
"""

from __future__ import annotations

import importlib
from typing import Any


class AddonMissingError(Exception):
    """Raised when a torch-gated addon is needed but not installed."""

    def __init__(self, name: str) -> None:
        self.addon_name = name
        super().__init__(
            f"The '{name}' addon is not installed. "
            f"Install it via Settings → Add-ons, or run: "
            f"pip install classify-desktop[{name}]"
        )


_REGISTRY: dict[str, str] = {
    "tabpfn": "tabpfn",
    "sdv": "sdv",
    "torch": "torch",
}

_CACHE: dict[str, bool] = {}


def is_available(name: str) -> bool:
    """Return True if the given addon module can be imported."""
    if name not in _REGISTRY:
        return False
    if name in _CACHE:
        return _CACHE[name]
    try:
        importlib.import_module(_REGISTRY[name])
        _CACHE[name] = True
    except ImportError:
        _CACHE[name] = False
    return _CACHE[name]


def require(name: str) -> Any:
    """Import and return the addon module, or raise AddonMissingError."""
    if not is_available(name):
        raise AddonMissingError(name)
    return importlib.import_module(_REGISTRY[name])


def refresh_cache() -> None:
    """Clear the availability cache (call after installing an addon)."""
    _CACHE.clear()


def list_addons() -> list[dict[str, str | bool]]:
    """Return status info for all known addons."""
    return [
        {"name": name, "module": mod, "installed": is_available(name)}
        for name, mod in _REGISTRY.items()
    ]


def tabpfn_available() -> bool:
    """Convenience: is TabPFN available (requires torch + tabpfn)?"""
    return is_available("torch") and is_available("tabpfn")


def sdv_available() -> bool:
    """Convenience: is SDV available (requires torch + sdv)?"""
    return is_available("torch") and is_available("sdv")
