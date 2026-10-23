"""LaTeX tables for the report, generated from results/*.json.

    python report/make_tables.py      # -> report/tables/*.tex

Each file holds a booktabs tabular only; captions and labels live in the report,
so re-running the benchmarks and this script refreshes every number in the PDF.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "tables"


def load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def tex(s) -> str:
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def write(name, spec, header, rows, best=None):
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(r"\midrule" if r == "MID" else " & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / name}.tex")


def pct(x, digits=2):
    return f"{100 * x:.{digits}f}\\%"


NAMES = {"hnsw": "HNSW (ours)", "ivf": "IVF (ours)", "flat": "Exact scan (ours)", "faiss-hnsw": "FAISS HNSW",
         "faiss-ivf": "FAISS IVF", "faiss-ivfpq": "FAISS IVF-PQ"}


def ann():
    d = load("ann")
    rows = []
    for name, r in d["runs"].items():
        ok = [p for p in r["points"] if p["recall"] >= 0.95]
        label = NAMES.get(name, f"IVF-PQ{name.split('pq')[-1]} (ours)" if "pq" in name else name)
        rows.append([label, f"{r['build_s']:.2f}", f"{max(p['recall'] for p in r['points']):.3f}",
                     f"{min(p['ms'] for p in ok):.3f}" if ok else "--"])
    write("ann", "lrrr", ["Index", "Build (s)", "Best recall@10", r"ms at recall $\geq$ 0.95"], rows)


def optimizer():
    d = load("optimizer")
    rows = []
    for r in d["sweep"]:
        rows.append([pct(r["actual"]), f"{r['exact']['ms']:.2f}", f"{r['index']['ms']:.2f}",
                     f"{r['postfilter']['ms']:.2f} ({r['postfilter']['recall']:.2f})",
                     rf"\textbf{{{r['auto']['ms']:.2f}}} ({r['auto']['recall']:.2f})", r["auto"]["chosen"]])
    write("optimizer", "rrrrrl", ["Selectivity", "Exact", "In-graph", "Post-filter (rec.)", "Optimizer (rec.)", "Chosen"], rows)
    rows = []
    for name, row in d["correlation"].items():
        rows.append([name] + [f"{row[k]['ms']:.2f} ({row[k]['recall']:.2f})" for k in ["exact", "index", "postfilter", "auto"]])
    write("correlation", "lrrrr", ["Filter", "Exact", "In-graph", "Post-filter", "Optimizer"], rows)


def topk():
    d = load("topk")
    rows, last = [], None
    for r in d["rows"]:
        if last and r["setting"] != last:
            rows.append("MID")
        last = r["setting"]
        rows.append([r["setting"], str(r["sources"]), str(r["k"]), f"{r['exhaustive']['ms']:.2f}",
                     f"{r['full']['ms']:.2f} ({r['full']['overlap'] * 100:.0f}\\%)",
                     f"{r['ta']['ms']:.2f} ({r['ta']['overlap'] * 100:.0f}\\%)",
                     f"{r['ta']['sorted']:.0f}+{r['ta']['random']:.0f}",
                     f"{r['nra']['ms']:.2f} ({r['nra']['overlap'] * 100:.0f}\\%)"])
    write("topk", "lrrrrrrr", ["Lists", "$m$", "$k$", "Exhaustive", "Merge@100", "TA", "TA accesses", "NRA"], rows)


def text():
    d = load("text")
    st = d["index"]
    ex, wand = d["algos"]["exhaustive"], d["algos"]["wand"]
    rows = [["Documents / distinct terms", f"{d['docs']:,} / {st['terms']:,}"],
            ["Postings", f"{st['postings']:,}"],
            ["Index size, compressed vs raw", f"{st['compressed_bytes'] / 1e6:.2f} MB vs {st['uncompressed_bytes'] / 1e6:.2f} MB"],
            ["Bits per posting", f"{d['bits_per_posting']:.1f}"],
            ["Vectorised VByte decode", f"{d['decode_million_ints_per_s']:.0f} M integers/s"],
            ["BM25 p50 latency: exhaustive / WAND", f"{ex['latency']['p50']:.2f} ms / {wand['latency']['p50']:.2f} ms"],
            ["Documents fully scored: exhaustive / WAND", f"{ex['docs_scored']:,.0f} / {wand['docs_scored']:,.0f}"],
            ["SQLite FTS5 p50 latency", f"{d['fts5']['latency']['p50']:.2f} ms"],
            ["Top-10 overlap with FTS5", f"{d['fts5']['top10_overlap_with_ours'] * 100:.0f}\\%"]]
    write("text", "lr", ["Metric", "Value"], rows)


def filters():
    d = load("filters")
    rows = [[pct(r["selectivity"], 3), f"{r['tree_ms']:.3f}", f"{r['scan_ms']:.3f}",
             "B+-tree" if r["optimizer_choice"] == "index" else "column scan"] for r in d["access_paths"]]
    write("filters", "rrrl", ["Selectivity", "B+-tree (ms)", "Column scan (ms)", "Optimizer choice"], rows)
    q = d["estimation_qerror"]
    rows = [[n, f"{q[k]['median']:.2f}", f"{q[k]['p90']:.2f}", f"{q[k]['max']:.1f}"]
            for n, k in [("Equi-depth + MCV (ours)", "equi_depth_mcv"), ("Equi-width", "equi_width")]]
    write("qerror", "lrrr", ["Statistics", "Median", "p90", "Max"], rows)


def quantization():
    d = load("quantization")
    rows = [[r["quantization"].upper() if r["quantization"] != "float32" else "float32", str(r["refine"] or "--"),
             f"{r['bytes_per_vector']:.0f}", f"{r['recall']:.3f}", f"{r['ms']:.2f}"] for r in d["rows"]]
    write("quantization", "lrrrr", ["Codes", "Re-rank", "Bytes/vector", "Recall@10", "ms/query"], rows)


def scale():
    d = load("scale")
    rows = []
    for r in d["rows"]:
        lat = r["latency"]
        rows.append([f"{r['n']:,}", f"{r['ingest_s']:.1f}", f"{r['hnsw_build_s']:.1f}", f"{r['faiss_hnsw_build_s'] or 0:.2f}",
                     f"{r['disk_bytes'] / 1e6:.1f}"] + [f"{lat[k]['p50']:.2f}" for k in ["keyword", "vector", "hybrid", "hybrid+filter"]])
    write("scale", "rrrrrrrrr", ["Docs", "Ingest (s)", "HNSW (s)", "FAISS (s)", "Disk (MB)", "Keyword", "Vector", "Hybrid", "+Filter"], rows)


MODES = [("keyword", "Keyword (BM25)"), ("vector", "Vector"), ("hybrid-rrf", "Hybrid RRF"),
         ("hybrid-weighted", "Hybrid weighted"), ("hybrid-auto", "Hybrid AUTO")]


def quality_demo():
    d = load("quality_demo")
    if not d:
        return
    rows, first = [], True
    for set_name, s in d["sets"].items():
        if not first:
            rows.append("MID")
        first = False
        best = max(s["results"][m]["nDCG@10"] for m, _ in MODES)
        for i, (m, label) in enumerate(MODES):
            r = s["results"][m]
            ndcg = f"{r['nDCG@10']:.3f}"
            if abs(r["nDCG@10"] - best) < 1e-9:
                ndcg = rf"\textbf{{{ndcg}}}"
            head = rf"\multirow{{5}}{{*}}{{\shortstack[l]{{{tex(set_name)}\\ \footnotesize $n={s['queries']}$}}}}" if i == 0 else ""
            rows.append([head, label, f"{r['R@1']:.3f}", f"{r['R@10']:.3f}", f"{r['MRR@10']:.3f}", ndcg, f"{r['p50_ms']:.2f}"])
    write("quality_demo", "llrrrrr", ["Query set", "Mode", "R@1", "R@10", "MRR@10", "nDCG@10", "p50 ms"], rows)
    q = d["query_by_example"]
    rows = [["Same colour", f"{q['P@10_same_colour']:.3f}", f"{q['chance_colour']:.3f}"],
            ["Same category", f"{q['P@10_same_category']:.3f}", f"{q['chance_category']:.3f}"]]
    write("qbe", "lrr", ["Top-10 neighbour shares the query's\\ldots", "Precision@10", "Chance"], rows)


def environment():
    rows = []
    for name, label in [("ann", "Systems benchmarks (ANN, optimizer, top-$k$, text, filters, compression, scale)"),
                        ("quality_demo", "Demo retrieval quality")]:
        d = load(name)
        if d:
            e = d["env"]
            rows.append([label, tex(f"Python {e['python']}, numpy {e['numpy']}, {e['cpu_count']} CPUs ({e['machine']}), {e['date']}")])
    write("environment", "p{0.42\\linewidth}p{0.5\\linewidth}", ["Experiments", "Environment"], rows)


if __name__ == "__main__":
    for fn in (ann, optimizer, topk, text, filters, quantization, scale, quality_demo, environment):
        fn()
