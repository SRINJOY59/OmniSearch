"""Command line: build the demo, run queries, serve the UI.

    python -m omnisearch.cli demo [--profile classic|clip]
    python -m omnisearch.cli query "SEARCH demo WHERE MATCH(text, 'no parking') LIMIT 5"
    python -m omnisearch.cli explain "SEARCH demo WHERE ..." [--analyze]
    python -m omnisearch.cli stats demo
    python -m omnisearch.cli calibrate demo
    python -m omnisearch.cli serve [--port 8000]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from omnisearch.engine import Engine


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omnisearch")
    ap.add_argument("--root", default="collections", help="directory holding collections")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="generate synthetic demo data and build the demo collection")
    d.add_argument("--profile", default="classic", choices=["classic", "clip"])
    d.add_argument("--data", default="data/demo")
    q = sub.add_parser("query", help="run an OmniQL query")
    q.add_argument("query")
    q.add_argument("--json", action="store_true")
    e = sub.add_parser("explain", help="show the physical plan")
    e.add_argument("query")
    e.add_argument("--analyze", action="store_true")
    s = sub.add_parser("stats")
    s.add_argument("collection")
    c = sub.add_parser("calibrate", help="measure cost-model constants on this machine")
    c.add_argument("collection")
    sv = sub.add_parser("serve", help="start the API + web UI")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        from omnisearch.demo import build_demo

        if not (Path(args.data) / "records.jsonl").exists():
            subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "scripts" / "make_demo_data.py"),
                            "--out", args.data], check=True)
        coll = build_demo(args.data, args.root, args.profile)
        print(json.dumps(coll.stats(), indent=2))
        return
    if args.cmd == "serve":
        import os

        import uvicorn

        os.environ["OMNISEARCH_ROOT"] = args.root
        uvicorn.run("omnisearch.api.server:app", host=args.host, port=args.port)
        return
    engine = Engine(args.root)
    if args.cmd == "query":
        res = engine.query(args.query)
        if args.json:
            print(json.dumps(res.to_dict(), indent=2, default=str))
            return
        for i, h in enumerate(res.hits, 1):
            moment = f"  @ {h.moment:.1f}s" if h.moment is not None else ""
            print(f"{i:>3}. {h.id:<20} {h.score:.4f}  {h.record.get('title', '')}{moment}")
        print(f"({len(res.hits)} hits, {res.timings['total_ms']:.1f} ms)")
    elif args.cmd == "explain":
        from omnisearch.query import parse

        coll = engine.collection(parse(args.query).collection)
        print(coll.explain(args.query, analyze=args.analyze))
    elif args.cmd == "stats":
        print(json.dumps(engine.collection(args.collection).stats(), indent=2))
    elif args.cmd == "calibrate":
        cm = engine.collection(args.collection).calibrate()
        print(json.dumps(cm.__dict__, indent=2))


if __name__ == "__main__":
    main()
