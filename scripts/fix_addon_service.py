"""Diagnose + repair remaining mojibake in addon_service.py, then apply
the unfreeze-based faker fix to the verification worker."""

from pathlib import Path

p = Path("backend/classify_api/services/addon_service.py")
text = p.read_text(encoding="utf-8")

# 1. diagnose the em-dash-like sequences
import re

for m in list(re.finditer("[\u00c0-\u00ff][\u0080-\u00ff]{1,6}", text))[:6]:
    seq = m.group(0)
    print("seq:", [hex(ord(c)) for c in seq], "->", repr(seq))

# 2. fix every occurrence of the double-encoded em-dash family
candidates = [
    "\u00c3\u00a2\u00e2\u0082\u00ac\u00c2\u0094",  # Ã¢â‚¬â€
    "\u00c3\u00a2\u0080\u0094",  # Ã¢â€"
    "\u00e2\u0080\u0094\u00c2".rstrip("\u00c2"),  # â€” (single-mangled variants end here)
]
em = "\u2014"
count = 0
for cand in candidates[:-1]:
    n = text.count(cand)
    if n:
        text = text.replace(cand, em)
        count += n
print("replaced:", count)

# 3. fix the verify worker: unfreeze instead of _MEIPASS override
old_worker = """    # faker resolves its locales from sys._MEIPASS when frozen"""
if old_worker in text:
    print("worker block still has old comment (will rewrite below)")

p.write_text(text, encoding="utf-8")
print("saved")
