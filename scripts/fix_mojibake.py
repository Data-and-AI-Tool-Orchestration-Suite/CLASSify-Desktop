"""One-off: iteratively repair cp1252/UTF-8 double-encoding in text files.

Safe for files whose characters are all in Latin-1 range; files containing
higher codepoints are reported for manual review.
"""
from pathlib import Path

ROOTS = ["backend", "desktop", "frontend/src", "pyproject.toml"]
SUFFIXES = {".py", ".ts", ".svelte", ".css", ".html", ".toml", ".md", ".spec", ".iss", ".json"}

changed = []
needs_manual = []
for root in ROOTS:
    root_path = Path(root)
    files = [root_path] if root_path.is_file() else list(root_path.rglob("*"))
    for f in files:
        if not f.is_file() or f.suffix not in SUFFIXES:
            continue
        try:
            text = f.read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError):
            continue
        original = text
        for _ in range(4):
            if any(ord(c) > 0xFF for c in text):
                break  # real unicode present — cannot round-trip safely
            if "Ã" not in text and "â€" not in text:
                break
            try:
                text = text.encode("cp1252").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                break
        if text != original:
            f.write_text(text, encoding="utf-8")
            changed.append(str(f))
        if "Ã" in text or "â€" in text:
            needs_manual.append(str(f))

print("files changed:", changed or "none")
print("needs manual review:", needs_manual or "none")
