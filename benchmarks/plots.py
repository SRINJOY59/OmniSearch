"""Figures and a markdown summary from results/*.json.

    python -m benchmarks.plots     # -> results/figures/*.png, results/SUMMARY.md

Palette: validated categorical slots (blue, orange, aqua) in fixed order;
the cost-based optimizer is drawn in primary ink (dashed) because it is not a
separate entity but a choice among the others. Every line is direct-labelled
and every figure has a numeric table in SUMMARY.md.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks.common import RESULTS  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
FIG = RESULTS / "figures"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 6, "lines.solid_capstyle": "round",
    })


def load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def label_end(ax, x, y, text, color, dx=4, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=INK2, fontsize=9,
                va="center", ha="left")


def note(fig, text):
    fig.text(0.01, -0.02, text, color=MUTED, fontsize=8, ha="left", va="top")


def plain_log(ax, axis="y"):
    """Log scale with plain tick labels (0.1, 1, 10) instead of 10^n."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    target = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    target.set_major_formatter(fmt)
    target.set_minor_formatter(NullFormatter())


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}.png"


# ---------------------------------------------------------------------- figures
def fig_ann(d):
    pairs = [("HNSW", "hnsw", "faiss-hnsw"), ("IVF", "ivf", "faiss-ivf"),
             ("IVF-PQ16 + rerank", next(k for k in d["runs"] if k.startswith("ivf-pq")), "faiss-ivfpq")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (title, ours, theirs) in zip(axes, pairs):
        for key, color, name in [(ours, S1, "OmniSearch (Python)"), (theirs, S2, "FAISS (C++)")]:
            if key not in d["runs"]:
                continue
            pts = d["runs"][key]["points"]
            xs, ys = [p["recall"] for p in pts], [p["ms"] for p in pts]
            ax.plot(xs, ys, color=color, marker="o", label=name)
            label_end(ax, xs[-1], ys[-1], "ours" if key == ours else "FAISS", color)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + (hi - lo) * 0.18)
        plain_log(ax, "y")
        ax.set_title(title)
        ax.set_xlabel("recall@10")
    axes[0].set_ylabel("latency per query (ms, log)")
    axes[0].legend(loc="upper left")
    note(fig, f"{d['n']:,} synthetic clustered vectors, d={d['dim']}, {d['queries']} queries, single thread")
    return save(fig, "ann_recall_latency")


def fig_optimizer(d):
    rows = d["sweep"]
    sel = [r["actual"] * 100 for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    names = [("exact", S1, "exact pre-filter scan"), ("index", S2, "HNSW in-graph filter"),
             ("postfilter", S3, "HNSW + post-filter")]
    for key, color, name in names:
        a1.plot(sel, [r[key]["ms"] for r in rows], color=color, marker="o", label=name)
        a2.plot(sel, [r[key]["recall"] for r in rows], color=color, marker="o", label=name)
    a1.plot(sel, [r["auto"]["ms"] for r in rows], color=INK, linestyle=(0, (4, 3)), marker="s",
            markersize=5, linewidth=1.6, label="cost-based optimizer (auto)")
    a2.plot(sel, [r["auto"]["recall"] for r in rows], color=INK, linestyle=(0, (4, 3)), marker="s",
            markersize=5, linewidth=1.6, label="cost-based optimizer (auto)")
    for r, x in zip(rows, sel):
        a1.annotate(r["auto"]["chosen"], (x, r["auto"]["ms"]), xytext=(0, -13), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK2)
    for ax in (a1, a2):
        plain_log(ax, "x")
        ax.set_xlabel("filter selectivity (% of docs passing, log)")
    plain_log(a1, "y")
    a1.set_ylabel("filter + search time (ms, log)")
    a1.set_title("Latency: the optimizer follows the cheapest access path")
    a2.set_ylabel("recall@10 vs exact filtered answer")
    a2.set_ylim(0, 1.05)
    a2.set_title("Accuracy")
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=9)
    note(fig, f"{d['n']:,} vectors (d={d['dim']}), {d['queries']} queries per point; labels under the dashed line = plan chosen")
    fig.texts[-1].set_y(-0.09)
    return save(fig, "optimizer_selectivity")


def fig_topk(d):
    rows = [r for r in d["rows"] if r["sources"] == 3]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    algos = [("exhaustive", S1, "score every doc"), ("full", S2, "materialise depth 100"),
             ("ta", S3, "Threshold Algorithm")]
    for ax, setting in zip(axes, ["correlated", "independent"]):
        sub = [r for r in rows if r["setting"] == setting]
        ks = [r["k"] for r in sub]
        width = 0.26
        for i, (a, color, name) in enumerate(algos):
            xs = [j + (i - 1) * (width + 0.02) for j in range(len(ks))]
            ax.bar(xs, [r[a]["ms"] for r in sub], width=width, color=color, label=name, zorder=2)
            for x, r in zip(xs, sub):
                if r[a]["overlap"] < 0.999:
                    ax.annotate(f"{r[a]['overlap']:.0%}", (x, r[a]["ms"]), xytext=(0, 3),
                                textcoords="offset points", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(range(len(ks)), [f"k={k}" for k in ks])
        ax.set_title(f"{setting} sources (3 lists)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("median search time (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=9)
    nra = {(r["setting"], r["k"]): r["nra"]["ms"] for r in rows}
    note(fig, "Labels: share of results that are truly top-k (others are 100%). NRA omitted for scale - " +
         ", ".join(f"{s} k={k}: {v:.0f} ms" for (s, k), v in nra.items()))
    fig.texts[-1].set_y(-0.09)
    return save(fig, "topk_algorithms")


def fig_text(d):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    names = ["exhaustive\n(term-at-a-time)", "WAND\n(doc-at-a-time)", "SQLite FTS5\n(C, baseline)"]
    lat = [d["algos"]["exhaustive"]["latency"]["p50"], d["algos"]["wand"]["latency"]["p50"], d["fts5"]["latency"]["p50"]]
    a1.bar(names, lat, color=[S1, S2, S3], width=0.55, zorder=2)
    for i, v in enumerate(lat):
        a1.annotate(f"{v:.2f} ms", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
    a1.set_title("p50 query latency")
    a1.set_ylabel("ms")
    scored = [d["algos"]["exhaustive"]["docs_scored"], d["algos"]["wand"]["docs_scored"]]
    a2.bar(names[:2], scored, color=[S1, S2], width=0.45, zorder=2)
    for i, v in enumerate(scored):
        a2.annotate(f"{v:,.0f}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
    a2.set_title("documents fully scored per query")
    for ax in (a1, a2):
        ax.grid(axis="x", visible=False)
    note(fig, f"{d['docs']:,} docs, {d['queries']} queries; postings compressed to {d['bits_per_posting']:.1f} bits "
              f"(vs 64 raw); FTS5 top-10 overlap {d['fts5']['top10_overlap_with_ours']:.0%}")
    return save(fig, "keyword_index")


def fig_filters(d):
    rows = d["access_paths"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    sel = [r["selectivity"] * 100 for r in rows]
    a1.plot(sel, [r["tree_ms"] for r in rows], color=S1, marker="o")
    a1.plot(sel, [r["scan_ms"] for r in rows], color=S2, marker="o")
    label_end(a1, sel[-1], rows[-1]["tree_ms"], "B+-tree index scan", S1)
    label_end(a1, sel[-1], rows[-1]["scan_ms"], "column scan", S2)
    switch = next((r["selectivity"] * 100 for r in rows if r["optimizer_choice"] == "scan"), None)
    if switch:
        a1.axvline(switch, color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
        a1.annotate("optimizer switches\nto column scan", (switch, a1.get_ylim()[1] * 0.5), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=INK2)
    plain_log(a1, "x")
    plain_log(a1, "y")
    a1.set_xlabel("range selectivity (%, log)")
    a1.set_ylabel("ms (log)")
    a1.set_title(f"Range predicate on {d['n']:,} rows")
    a1.set_xlim(right=max(sel) * 6)
    q = d["estimation_qerror"]
    labels = ["median", "p90"]
    x = range(len(labels))
    a2.bar([i - 0.2 for i in x], [q["equi_depth_mcv"][l] for l in labels], width=0.38, color=S1,
           label="equi-depth + MCV (ours)", zorder=2)
    a2.bar([i + 0.2 for i in x], [q["equi_width"][l] for l in labels], width=0.38, color=S2,
           label="equi-width", zorder=2)
    a2.set_xticks(list(x), labels)
    a2.set_title("Selectivity estimate q-error (1 = perfect)")
    a2.set_ylim(bottom=1)
    a2.legend(fontsize=8.5)
    a2.grid(axis="x", visible=False)
    note(fig, f"max q-error: equi-depth {q['equi_depth_mcv']['max']:.1f}, equi-width {q['equi_width']['max']:.1f}; "
              "skewed log-normal values with an 8% heavy hitter")
    return save(fig, "filters")


def fig_quant(d):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for refine, color, name in [(0, S1, "codes only"), (4, S2, "codes + exact re-rank of 4k")]:
        pts = [r for r in d["rows"] if r["refine"] == refine or (refine == 4 and r["quantization"] == "float32")]
        pts.sort(key=lambda r: r["bytes_per_vector"])
        xs, ys = [r["bytes_per_vector"] for r in pts], [r["recall"] for r in pts]
        ax.plot(xs, ys, color=color, marker="o", label=name)
        for r in pts:
            if refine == 0 or r["quantization"] == "float32":
                ax.annotate(r["quantization"], (r["bytes_per_vector"], r["recall"]), xytext=(5, -10),
                            textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("bytes per vector in memory (log2)")
    ax.set_ylabel("recall@10")
    ax.set_title("Compression vs accuracy")
    ax.legend(loc="lower right")
    note(fig, f"{d['n']:,} vectors, d={d['dim']}; re-ranking reads the float32 vectors for 40 candidates")
    return save(fig, "quantization")


def fig_scale(d):
    rows = d["rows"]
    ns = [r["n"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.plot(ns, [r["hnsw_build_s"] for r in rows], color=S1, marker="o")
    label_end(a1, ns[-1], rows[-1]["hnsw_build_s"], "OmniSearch HNSW", S1)
    if all(r["faiss_hnsw_build_s"] for r in rows):
        a1.plot(ns, [r["faiss_hnsw_build_s"] for r in rows], color=S2, marker="o")
        label_end(a1, ns[-1], rows[-1]["faiss_hnsw_build_s"], "FAISS HNSW", S2)
    plain_log(a1, "y")
    a1.set_ylabel("build time (s, log)")
    a1.set_xlabel("documents")
    a1.set_title("Index build")
    a1.set_xlim(right=ns[-1] * 1.35)
    for key, color in [("keyword", S1), ("vector", S2), ("hybrid", S3)]:
        ys = [r["latency"][key]["p50"] for r in rows]
        a2.plot(ns, ys, color=color, marker="o")
        label_end(a2, ns[-1], ys[-1], key, color)
    ys = [r["latency"]["hybrid+filter"]["p50"] for r in rows]
    a2.plot(ns, ys, color=INK, marker="s", markersize=5, linewidth=1.6, linestyle=(0, (4, 3)))
    label_end(a2, ns[-1], ys[-1], "hybrid + filter", INK)
    a2.set_ylim(bottom=0)
    a2.set_xlim(right=ns[-1] * 1.35)
    a2.set_ylabel("p50 latency (ms)")
    a2.set_xlabel("documents")
    a2.set_title("Query latency through the full engine")
    note(fig, "single thread; latency = plan + filter + search + fuse + fetch")
    return save(fig, "scale")


# ---------------------------------------------------------------------- summary
def summary_md(figs) -> str:
    out = ["# Benchmark results", "",
           "Generated by `python -m benchmarks.plots` from `results/*.json`. All systems benchmarks use",
           "synthetic data with controlled structure (no model downloads); retrieval-quality results on real",
           "datasets are produced separately by `benchmarks/retrieval_quality.py`.", ""]
    env = None
    if (d := load("ann")):
        env = d["env"]
        out += ["## ANN indexes (recall@10 vs latency)", "", f"![ann]({figs['ann']})", "",
                "| index | build (s) | best recall | latency at recall >= 0.95 (ms) |", "|---|---:|---:|---:|"]
        for name, r in d["runs"].items():
            ok = [p for p in r["points"] if p["recall"] >= 0.95]
            lat = f"{min(p['ms'] for p in ok):.3f}" if ok else "-"
            out.append(f"| {name} | {r['build_s']:.2f} | {max(p['recall'] for p in r['points']):.3f} | {lat} |")
        out.append("")
    if (d := load("optimizer")):
        out += ["## Cost-based optimizer vs fixed access paths", "", f"![optimizer]({figs['optimizer']})", "",
                "| selectivity | exact ms | in-graph ms | post-filter ms (recall) | auto ms (recall) | auto chose |",
                "|---:|---:|---:|---:|---:|---|"]
        for r in d["sweep"]:
            out.append(f"| {r['actual']:.2%} | {r['exact']['ms']:.2f} | {r['index']['ms']:.2f} | "
                       f"{r['postfilter']['ms']:.2f} ({r['postfilter']['recall']:.2f}) | "
                       f"{r['auto']['ms']:.2f} ({r['auto']['recall']:.2f}) | {r['auto']['chosen']} |")
        out += ["", "Filter correlated with the query vs anti-correlated (same selectivity):", "",
                "| filter | " + " | ".join(f"{k} ms (recall)" for k in ["exact", "index", "postfilter", "auto"]) + " |",
                "|---|---:|---:|---:|---:|"]
        for name, row in d["correlation"].items():
            out.append(f"| {name} | " + " | ".join(f"{row[k]['ms']:.2f} ({row[k]['recall']:.2f})"
                                                for k in ["exact", "index", "postfilter", "auto"]) + " |")
        out.append("")
    if (d := load("topk")):
        out += ["## Top-k fusion algorithms (WEIGHTED fusion)", "", f"![topk]({figs['topk']})", "",
                "| setting | lists | k | exhaustive ms | full ms (correct) | TA ms (correct) | TA sorted+random accesses | NRA ms (correct) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in d["rows"]:
            out.append(f"| {r['setting']} | {r['sources']} | {r['k']} | {r['exhaustive']['ms']:.2f} | "
                       f"{r['full']['ms']:.2f} ({r['full']['overlap']:.0%}) | {r['ta']['ms']:.2f} ({r['ta']['overlap']:.0%}) | "
                       f"{r['ta']['sorted']:.0f}+{r['ta']['random']:.0f} | {r['nra']['ms']:.2f} ({r['nra']['overlap']:.0%}) |")
        out.append("")
    if (d := load("text")):
        st = d["index"]
        out += ["## Keyword index", "", f"![text]({figs['text']})", "",
                f"- {d['docs']:,} docs, {st['postings']:,} postings: {st['compressed_bytes']/1e6:.2f} MB compressed vs "
                f"{st['uncompressed_bytes']/1e6:.2f} MB raw ({d['bits_per_posting']:.1f} bits per posting)",
                f"- vectorised decode: {d['decode_million_ints_per_s']:.0f} M integers/s",
                f"- BM25 p50: exhaustive {d['algos']['exhaustive']['latency']['p50']:.2f} ms, WAND "
                f"{d['algos']['wand']['latency']['p50']:.2f} ms (scores {d['algos']['wand']['docs_scored']:.0f} vs "
                f"{d['algos']['exhaustive']['docs_scored']:.0f} docs), SQLite FTS5 {d['fts5']['latency']['p50']:.2f} ms; "
                f"top-10 overlap with FTS5 {d['fts5']['top10_overlap_with_ours']:.0%}", ""]
    if (d := load("filters")):
        out += ["## Attribute filters", "", f"![filters]({figs['filters']})", "",
                "| selectivity | B+-tree ms | column scan ms | optimizer |", "|---:|---:|---:|---|"]
        for r in d["access_paths"]:
            out.append(f"| {r['selectivity']:.3%} | {r['tree_ms']:.3f} | {r['scan_ms']:.3f} | {r['optimizer_choice']} |")
        q = d["estimation_qerror"]
        out += ["", f"Selectivity q-error - equi-depth + MCV: median {q['equi_depth_mcv']['median']:.2f}, p90 "
                f"{q['equi_depth_mcv']['p90']:.2f}, max {q['equi_depth_mcv']['max']:.1f}; equi-width: median "
                f"{q['equi_width']['median']:.2f}, p90 {q['equi_width']['p90']:.2f}, max {q['equi_width']['max']:.1f}", ""]
    if (d := load("quantization")):
        out += ["## Vector compression", "", f"![quantization]({figs['quantization']})", "",
                "| codes | re-rank | bytes/vector | recall@10 | ms |", "|---|---:|---:|---:|---:|"]
        for r in d["rows"]:
            out.append(f"| {r['quantization']} | {r['refine'] or '-'} | {r['bytes_per_vector']:.0f} | {r['recall']:.3f} | {r['ms']:.2f} |")
        out.append("")
    if (d := load("scale")):
        out += ["## Scale", "", f"![scale]({figs['scale']})", "",
                "| docs | ingest s | HNSW build s | FAISS build s | disk MB | keyword | vector | hybrid | hybrid+filter (p50 ms) |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in d["rows"]:
            lat = r["latency"]
            out.append(f"| {r['n']:,} | {r['ingest_s']:.1f} | {r['hnsw_build_s']:.1f} | {r['faiss_hnsw_build_s'] or 0:.2f} | "
                       f"{r['disk_bytes']/1e6:.1f} | {lat['keyword']['p50']:.2f} | {lat['vector']['p50']:.2f} | "
                       f"{lat['hybrid']['p50']:.2f} | {lat['hybrid+filter']['p50']:.2f} |")
        out.append("")
    for name in sorted(p.stem for p in RESULTS.glob("quality_*.json")):
        d = load(name)
        out += [f"## Retrieval quality: {d['dataset']}", "", "| mode | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 |",
                "|---|---:|---:|---:|---:|---:|"]
        for mode, r in d["results"].items():
            out.append(f"| {mode} | {r['R@1']:.3f} | {r['R@5']:.3f} | {r['R@10']:.3f} | {r['MRR@10']:.3f} | {r['nDCG@10']:.3f} |")
        out.append("")
    if env:
        out += [f"_Environment: Python {env['python']}, numpy {env['numpy']}, {env['cpu_count']} CPUs, {env['date']}._", ""]
    return "\n".join(out)


def main(argv=None):
    style()
    figs = {}
    for key, name, fn in [("ann", "ann", fig_ann), ("optimizer", "optimizer", fig_optimizer), ("topk", "topk", fig_topk),
                          ("text", "text", fig_text), ("filters", "filters", fig_filters),
                          ("quantization", "quantization", fig_quant), ("scale", "scale", fig_scale)]:
        d = load(name)
        if d:
            figs[key] = fn(d)
            print(f"figure: results/{figs[key]}")
    (RESULTS / "SUMMARY.md").write_text(summary_md(figs), encoding="utf-8")
    print(f"wrote {RESULTS / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
