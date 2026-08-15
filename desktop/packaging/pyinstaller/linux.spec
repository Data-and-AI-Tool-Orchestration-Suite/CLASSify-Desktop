# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for CLASSify Desktop on Linux.

Produces a --onedir bundle.  Packaged as AppImage, .deb, and .tar.gz.
"""

block_cipher = None

from pathlib import Path

repo_root = Path(SPECPATH).parents[3]
frontend_dist = str(repo_root / "frontend" / "dist")
migrations_dir = str(repo_root / "backend" / "migrations")

a_datas = [
    (frontend_dist, "frontend/dist"),
    (migrations_dir, "migrations"),
]

a_hidden_imports = [
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
    "webview.platforms.gtk",
    "sqlalchemy.dialects.sqlite",
    "charset_normalizer",
    "joblib",
    "optuna",
    "xgboost",
    "shap",
]

a_excludes = [
    "torch", "tabpfn", "sdv", "ctgan", "copulas", "rdt", "deepecho",
    "clearml", "boto3", "s3transfer", "flask", "pytest", "mypy", "ruff",
    "pip", "setuptools", "_pytest", "IPython", "jupyter", "notebook",
    "tkinter",
]

a = Analysis(
    [str(repo_root / "desktop" / "classify_desktop" / "__main__.py")],
    pathex=[str(repo_root / "backend"), str(repo_root / "desktop")],
    binaries=[],
    datas=a_data,
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
    binaries=[],
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
    strip=True,
    upx=True,
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
    strip=True,
    upx=True,
    console=True,
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
    upx=True,
    name="CLASSify",
)
