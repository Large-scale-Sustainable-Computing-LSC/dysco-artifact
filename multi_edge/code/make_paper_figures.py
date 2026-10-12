"""Draw the results figures from the extracted numbers.

    python3 make_paper_figures.py pod128            # writes into ../figures
    python3 make_paper_figures.py pod128 /tmp/check # write elsewhere, to compare

Annotations that state a finding are drawn only while the numbers support them.
"""

from __future__ import annotations

from typing import Any
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY = sys.argv[1] if len(sys.argv) > 1 else "pod128"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.normpath(os.path.join(HERE, "..", "figures"))
os.makedirs(OUTDIR, exist_ok=True)


def _numbers_path(family: str) -> str:
    """Find the extracted numbers beside this script, then in ``results/``."""

    local = os.path.join(HERE, f"analyzed_{family}.json")
    return local if os.path.exists(local) else os.path.join(HERE, "..", "results", f"analyzed_{family}.json")


NUM = json.load(open(_numbers_path(FAMILY)))

NOISE_FLOOR_CV = 6.0      # % - the k=1 control shows six-trial points cannot resolve less
C = {"fifo": "#2a78d6", "batch": "#eb6834", "inter": "#eda100", "dsb": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7.2, "axes.labelsize": 7.6,
    "axes.titlesize": 7.8, "legend.fontsize": 7.0, "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True,
})


def style(ax: Any) -> None:
    """Drop the top and right spines and the vertical grid."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=False)


def save(fig: Any, stem: str) -> None:
    """Write one figure as both PDF and PNG."""

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{stem}.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {os.path.join(OUTDIR, stem)}.{{pdf,png}}")


def span(series_list: list[tuple[list[float], list[float]]], floor_to: int = 10) -> tuple[float, float]:
    """Return a y-range covering every series and its error bars, snapped outward."""

    lo = min(mean - err for means, errs in series_list for mean, err in zip(means, errs))
    hi = max(mean + err for means, errs in series_list for mean, err in zip(means, errs))
    return math.floor(lo / floor_to) * floor_to, math.ceil(hi / floor_to) * floor_to


# ---------------------------------------------------------------- Figure: mechanism
if "tail" in NUM and "diversity" in NUM:
    rows = sorted(NUM["tail"]["rows"], key=lambda r: r["tail"])
    tail = [r["tail"] for r in rows]
    t_series = [([r[c][0] for r in rows], [r[c][1] for r in rows]) for c in ("fifo", "batch", "dsb")]
    ks = sorted(int(k) for k in NUM["diversity"]["gains"])
    arr = NUM["diversity"]["arrays"]
    d_series = [([p[0] for p in arr[c]], [p[1] for p in arr[c]]) for c in ("fifo", "batch", "progressive-sc")]

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.3))
    style_rows = [("FIFO", C["fifo"], "o", "--"), ("Exact-match", C["batch"], "s", "-."), ("DSB", C["dsb"], "D", "-")]
    for ax, xs, series, xlabel in [(axes[0], tail, t_series, "shared tail depth (layers)"),
                                   (axes[1], ks, d_series, "distinct cut layers among 8 sessions")]:
        for (lab, col, mk, ls), (ys, es) in zip(style_rows, series):
            ax.errorbar(xs, ys, yerr=es, label=lab, color=col, marker=mk, linestyle=ls,
                        linewidth=1.6, markersize=4.2, capsize=2, elinewidth=0.7,
                        markeredgecolor="white", markeredgewidth=0.5)
        lo, hi = span(series)
        ax.set_ylim(lo, hi + (0.22 * (hi - lo) if ax is axes[0] else 0.04 * (hi - lo)))  # legend headroom on top panel
        ax.set_xticks(xs); ax.set_xlabel(xlabel); ax.set_ylabel("decode throughput (tok/s)"); style(ax)
    axes[0].legend(frameon=False, loc="upper left", ncol=3, handlelength=1.8,
                   columnspacing=0.9, borderaxespad=0.2)

    cv = NUM["tail"]["dsb_cv"]
    lo0, hi0 = axes[0].get_ylim()
    dsb_t = t_series[2][0]
    if cv < NOISE_FLOOR_CV:
        axes[0].annotate(f"DSB near-constant\n(CV {cv:.1f}%)", xy=(tail[-2], dsb_t[-2]),
                         xytext=(tail[len(tail) // 2] + 1.4, hi0 - 0.20 * (hi0 - lo0)),
                         color=INK2, fontsize=6.8, ha="left",
                         arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    else:
        print(f"NOTE: DSB CV across tail depths is {cv:.1f}% (>= {NOISE_FLOOR_CV}%) - "
              f"'near-constant' annotation withheld; revisit the text that claims it")

    bat_d = d_series[1][0]
    decay = 100.0 * (1 - bat_d[-1] / bat_d[0])
    lo1, hi1 = axes[1].get_ylim()
    if decay > 15:
        i7 = ks.index(7) if 7 in ks else len(ks) - 2
        axes[1].annotate("exact-match decays\nwith heterogeneity", xy=(ks[i7], bat_d[i7]),
                         xytext=(ks[0] + 1.9, lo1 + 0.27 * (hi1 - lo1)), color=INK2, fontsize=6.8,
                         arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    else:
        print(f"NOTE: exact-match falls only {decay:.1f}% from 1 to {ks[-1]} distinct cuts - "
              f"'decays' annotation withheld; revisit the text that claims it")
    fig.tight_layout(pad=0.4)
    save(fig, "fig_dsb_mechanism")
    plt.close(fig)
else:
    print("SKIP fig_dsb_mechanism: tail-depth or diversity results not available yet")


# ---------------------------------------------------------------- Figure: Poisson
if "poisson" in NUM:
    table = NUM["poisson"]["table"]
    Ls = sorted(table, key=float)
    labels = [f"L={float(L):g}" for L in Ls]
    order = [("FIFO", "fifo", C["fifo"], ""), ("Exact-match", "batch", C["batch"], "///"),
             ("Interleave", "interleave", C["inter"], "xxx"), ("DSB", "progressive-sc", C["dsb"], "...")]
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.0))
    w, xs = 0.2, list(range(len(Ls)))
    thr_top = max(table[L][c]["tok"] + table[L][c]["tok_std"] for L in Ls for _, c, _, _ in order)
    lat_top = max(table[L][c]["lat"] + table[L][c]["lat_std"] for L in Ls for _, c, _, _ in order)
    thr_lim, lat_lim = thr_top * 1.24, lat_top * 1.12      # room for legend and DSB value labels
    for i, (lab, c, col, hatch) in enumerate(order):
        off = (i - 1.5) * w
        axes[0].bar([x + off for x in xs], [table[L][c]["tok"] for L in Ls], w * 0.92,
                    yerr=[table[L][c]["tok_std"] for L in Ls], label=lab, color=col, hatch=hatch,
                    edgecolor="white", linewidth=0.8, capsize=2, error_kw=dict(elinewidth=0.7, ecolor=INK2))
        axes[1].bar([x + off for x in xs], [table[L][c]["lat"] for L in Ls], w * 0.92,
                    yerr=[table[L][c]["lat_std"] for L in Ls], color=col, hatch=hatch,
                    edgecolor="white", linewidth=0.8, capsize=2, error_kw=dict(elinewidth=0.7, ecolor=INK2))
    for ax, ylab, top in [(axes[0], "throughput (tok/s)", thr_lim), (axes[1], "mean latency (s)", lat_lim)]:
        ax.set_xticks(xs); ax.set_xticklabels(labels); ax.set_ylabel(ylab); ax.set_ylim(0, top); style(ax)
    for i, L in enumerate(Ls):                                   # direct labels on the series the figure is about
        v, e = table[L]["progressive-sc"]["tok"], table[L]["progressive-sc"]["tok_std"]
        axes[0].text(i + 1.5 * w, v + e + 0.03 * thr_lim, f"{v:.0f}", ha="center", va="bottom",
                     fontsize=6.6, color=INK)
    axes[0].legend(frameon=False, ncol=2, loc="upper left", handlelength=1.5, columnspacing=1.0)
    axes[1].set_xlabel("target average concurrency (Little's Law)")
    fig.tight_layout(pad=0.4)
    save(fig, "fig_dsb_poisson")
    plt.close(fig)
else:
    print("SKIP fig_dsb_poisson: Poisson results not available yet")
