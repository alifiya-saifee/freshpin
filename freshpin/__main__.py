"""Command line:  python -m freshpin serve | mcp | check "answer text" | sample <id>"""
from __future__ import annotations

import json
import sys

from .app import FreshPin
from .config import settings
from .server import serve

USAGE = """usage:
  python -m freshpin serve                 start the web app (default http://127.0.0.1:8787)
  python -m freshpin mcp                   run as an MCP server (stdio) for AI assistants
  python -m freshpin check "answer text"   check an answer and print the JSON report
  python -m freshpin sample <id>           load a sample ledger, check its answer, print a summary
"""


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    cmd = argv[0]
    if cmd == "serve":
        serve()
        return 0
    if cmd == "mcp":
        from .mcp import run
        run()
        return 0
    app = FreshPin(settings())
    if cmd == "check" and len(argv) > 1:
        print(json.dumps(app.check(" ".join(argv[1:])), indent=2, ensure_ascii=False))
        return 0
    if cmd == "sample" and len(argv) > 1:
        app.load_sample(argv[1])
        rep = app.check(app.samples()[argv[1]]["answer"])
        for c in rep["claims"]:
            f = c["freshness"]
            print(f"[{f['band'].upper():5}] {f['score']:3}  {c['text']}")
            if c["verdict"] in ("stale", "contradicted"):
                print(f"         now: {c['current_statement']}")
        print("reopened:", rep["summary"]["decisions_reopened"] or "none")
        g = rep["gateway"]
        print(f"gateway: {g['calls']} calls, {g['failed_attempts']} failed, {g['fallbacks_used']} fallbacks, {g['tokens']} tokens")
        return 0
    print(USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
