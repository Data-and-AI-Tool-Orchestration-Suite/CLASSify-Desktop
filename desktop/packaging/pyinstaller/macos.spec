# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for CLASSify Desktop on macOS.

Build separately on arm64 and x86_64, then lipo-merge for Universal2.
Produces a --onedir bundle with the main app + jobworker.
"""

block_cipher = None

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

repo_root = Path(SPECPATH).parents[2]
frontend_dist = str(repo_root / "frontend" / "dist")
migrations_dir = str(repo_root / "backend" / "migrations")

a_binaries = collect_dynamic_libs("xgboost")

a_datas = [
    (frontend_dist, "frontend/dist"),
    (migrations_dir, "migrations"),
] + collect_data_files("xgboost") + collect_data_files("pip", include_py_files=True)


def _pip_stdlib_hidden():
    """Stdlib modules pip imports at runtime (pip ships as disk data)."""
    import ast as _ast

    import sys as _sys

    import pip as _pip

    root = Path(_pip.__path__[0])
    names = set()
    for py_file in root.rglob("*.py"):
        try:
            tree = _ast.parse(py_file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                for alias in node.names:
                    names.add(alias.name)
            elif isinstance(node, _ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module)
    hidden = set()
    for name in names:
        if name.split(".")[0] in _sys.stdlib_module_names:
            hidden.add(name)
            while "." in name:
                name = name.rsplit(".", 1)[0]
                hidden.add(name)
    return sorted(hidden)


a_hidden_imports = [
    # stdlib modules pip needs at runtime (pip is a disk package,
    # so its imports are not analyzed — compute them from its source)
    *_pip_stdlib_hidden(),
    # App modules referenced by import string (invisible to static analysis)
    "classify_api",
    "classify_api.main",
    "runner.jobworker",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "sklearn._loss",
    "sklearn._loss.link",
    "sklearn._loss.loss",
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.utils._vector_sentinel",
    "sklearn.metrics._pairwise_distances_reduction._datasets_pair",
    "sklearn.metrics._pairwise_distances_reduction._middle_term_computer",
    "sklearn.metrics._pairwise_distances_reduction._base",
    "scipy.special._cdflib",
    "scipy.linalg.cython_blas",
    "scipy.linalg.cython_lapack",
    "webview.platforms.cocoa",
    "sqlalchemy.dialects.sqlite",
    "charset_normalizer",
    "joblib",
    "optuna",
    "xgboost",
    "shap",
]

a_excludes = [
    "pip",
    "torch", "tabpfn", "sdv", "ctgan", "copulas", "rdt", "deepecho",
    "clearml", "boto3", "s3transfer", "flask", "pytest", "mypy", "ruff",
    "setuptools", "_pytest", "IPython", "jupyter", "notebook",
    "tkinter",
]

a = Analysis(
    [str(repo_root / "desktop" / "classify_desktop" / "__main__.py")],
    pathex=[str(repo_root / "backend"), str(repo_root / "desktop")],
    binaries=a_binaries,
    datas=a_datas,
    hiddenimports=a_hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=a_excludes,
    cipher=block_cipher,
    noarchive=False,
)

jobworker_a = Analysis(
    [str(repo_root / "backend" / "runner" / "jobworker.py")],
    pathex=[str(repo_root / "backend"), str(repo_root / "desktop")],
    binaries=a_binaries,
    datas=[],
    hiddenimports=a_hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=a_excludes,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
jobworker_pyz = PYZ(jobworker_a.pure, jobworker_a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CLASSify",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

jobworker_exe = EXE(
    jobworker_pyz,
    jobworker_a.scripts,
    [],
    exclude_binaries=True,
    name="classify-jobworker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    jobworker_exe,
    jobworker_a.binaries,
    jobworker_a.zipfiles,
    jobworker_a.datas,
    strip=False,
    upx=False,
    name="CLASSify",
)
