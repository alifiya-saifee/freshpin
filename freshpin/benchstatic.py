"""A read-only snapshot of /bench for static hosting (GitHub Pages or any web server): the same page, reading
exported files instead of the live API. Only the committed samples (bench/samples) go in, never local results."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .server import SAMPLES, WEB, bench_file, bench_files

REPO = "https://github.com/alifiya-saifee/freshpin"
BANNER = ('<div style="background:#2a2410;border-bottom:1px solid #6b5a1e;color:#f2d38a;padding:8px 16px;'
          'font-size:14px">Read-only snapshot of the published results. Nothing runs here: '
          f'<a href="{REPO}#run-it" style="color:#f2b544">run FreshPin yourself</a> to benchmark your own agent.</div>')


def build(out_dir: Path) -> dict[str, int]:
    """Write index.html, data/list.json and data/<sample>.json into out_dir (created; must be empty or new)."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"{out_dir} is not empty; pick a new or empty folder")
    data = out_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    files = bench_files(include_results=False)
    for f in files:
        src = bench_file(f["name"])
        if src is None or src.parent != SAMPLES:          # samples only, never bench/results
            raise ValueError(f"not a committed sample: {f['name']}")
        shutil.copyfile(src, data / f["name"])
    (data / "list.json").write_text(json.dumps({"files": files}, ensure_ascii=False), encoding="utf-8")
    html = (WEB / "bench.html").read_text(encoding="utf-8")
    for old, new in (("</head>", "<script>window.FRESHPIN_STATIC = true;</script>\n</head>"),
                     ('<a href="/">← stale-answer detector</a>', f'<a href="{REPO}">← FreshPin on GitHub</a>'),
                     ("<body>", "<body>\n" + BANNER)):
        if html.count(old) != 1:
            raise ValueError(f"bench.html changed: expected one {old!r}")
        html = html.replace(old, new)
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")   # GitHub Pages: serve files as they are
    return {"results": len(files)}
