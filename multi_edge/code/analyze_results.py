"""

    python3 analyze_results.py pod128

Writes results/analyzed_<family>.json, which the figure script reads.
Missing inputs are reported rather than fatal. Gains are depth-synchronized
batching over the named baseline; t is Welch's t.
"""

from __future__ import annotations

from typing import Any, Callable
import json
import math
import os
import re
import statistics as st
import sys

FAMILY = sys.argv[1] if len(sys.argv) > 1 else "pod128"
HERE = os.path.dirname(os.path.abspath(__file__))
L_LLAMA, L_QWEN = 28, 36

FILES = {
    "pod128": dict(
        poisson="stats_poisson_hetero_pod128.json",
        tail="stats_tail_depth_sweep_pod128.json",
        div="stats_diversity_sweep_pod128.json",
        sched=["stats_hetero_sched_pod128_b1.json", "stats_hetero_sched_pod128_b2.json"],
        matrix="stats_fullshape_matrix_pod128.json",
        qwen="stats_tail_depth_sweep_qwen_pod128.json",
        gradient=dict(homog="stats_gradient_homog_pod128.json",
                      twotier="stats_gradient_twotier_pod128.json",
                      fullyhetero="stats_gradient_fullyhetero_pod128.json"),
        mem=dict(off="log_mem_llama_homog_off_pod128.log", on="log_mem_llama_homog_on_pod128.log"),
        context="stats_context_length_sweep_pod128.json",
        profile=dict(short="profile_wildgpt4_pod128_server.log", long="profile_longreason_pod128_server.log"),
        gen_tokens=128,
    ),
}[FAMILY]

# Distinct cut layers per fleet in the shape matrix (see fullshape_matrix_v2).
DISTINCT = {"s01_homog_d2": 1, "s01_homog_d8": 1, "s01_homog_d14": 1, "s05_maj6_out10_22": 3,
            "s06_twoclust_2_10": 2, "s07_twoclust_2_18": 2, "s08_threeclust": 3,
            "s09_spread_shallow": 8, "s11_spread_even": 8}
for outlier in (2, 4, 7, 13, 16, 19, 22, 26):
    DISTINCT[f"s03_majmid10_out{outlier}"] = 2
for outlier in (10, 19):
    DISTINCT[f"s04_maj6_pair{outlier}"] = 2


def path(name: str) -> str:
    """Resolve a result file beside this script, then in ``results/``."""

    local = os.path.join(HERE, name)
    return local if os.path.exists(local) else os.path.join(HERE, "..", "results", name)


def load(name: str) -> Any:
    """Load one result file as JSON."""

    return json.load(open(path(name)))


def pct(a: float, b: float) -> float:
    """Return ``a`` relative to ``b`` as a percentage change."""

    return 100.0 * (a / b - 1.0)


def welch(x: list[float], y: list[float]) -> float:
    """Welch's t statistic for two samples of per-trial values."""

    return (st.mean(x) - st.mean(y)) / math.sqrt(st.variance(x) / len(x) + st.variance(y) / len(y))


def welch_summary(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Welch's t from stored means and standard deviations."""

    na = len(a.get("trials") or []) or 6
    nb = len(b.get("trials") or []) or 6
    return (a["mean"] - b["mean"]) / math.sqrt(a["std"] ** 2 / na + b["std"] ** 2 / nb)


def fit(xs: list[float], ys: list[float]) -> dict[str, float]:
    """Least-squares fit of ``ys`` against ``xs``."""

    mean_x, mean_y = st.mean(xs), st.mean(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    spread_x = sum((x - mean_x) ** 2 for x in xs)
    spread_y = sum((y - mean_y) ** 2 for y in ys)
    slope = covariance / spread_x
    return dict(slope=slope, r=covariance / math.sqrt(spread_x * spread_y), intercept=mean_y - slope * mean_x)


def cv(values: list[float]) -> float:
    """Coefficient of variation, in percent."""

    return 100.0 * st.stdev(values) / st.mean(values)


def pooled(block_files: list[str], conds: list[str]) -> dict[str, dict[str, list[float]]]:
    """Pool per-trial throughput across measurement blocks, per condition and N."""

    blocks = [load(name) for name in block_files]
    return {c: {N: [t[0] for b in blocks for t in b["raw"][c][N]] for N in ("2", "4", "6", "8")}
            for c in conds}


def tail_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn a tail-depth sweep into rows sorted by shared tail depth."""

    rows = []
    for key, value in results.items():
        match = re.match(r"tail(\d+)_deep(\d+)", key)
        rows.append(dict(tail=int(match.group(1)), maxcut=int(match.group(2)), fifo=value["fifo"],
                         batch=value["batch"], dsb=value["progressive-sc"],
                         gain=pct(value["progressive-sc"]["mean"], value["batch"]["mean"])))
    return sorted(rows, key=lambda row: row["tail"])


OUT: dict[str, Any] = {"family": FAMILY, "gen_tokens": FILES["gen_tokens"]}
MISSING: list[str] = []


def section(name: str) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    """Run one extraction function and record its result, or why it was skipped."""

    def decorator(fn: Callable[[], Any]) -> Callable[[], Any]:
        try:
            OUT[name] = fn()
        except FileNotFoundError as exc:
            MISSING.append(f"{name}: {os.path.basename(exc.filename)}")
        except KeyError as exc:
            # A derived section whose input section has not been produced yet.
            MISSING.append(f"{name}: waits on {exc.args[0]!r}")
        except (AttributeError, ValueError, IndexError, StopIteration) as exc:
            # An input that exists but is still being written, for example a log
            # whose final summary line has not appeared yet.
            MISSING.append(f"{name}: input incomplete ({type(exc).__name__})")
        return fn

    return decorator


# ---------------------------------------------------------------- sections
@section("gradient")
def _():
    r, tt = {}, {}
    for shape, f in FILES["gradient"].items():
        d = load(f); s = d["summaries"]
        r[shape] = pct(s["batch"]["8"]["clean_tok_s_mean"], s["fifo"]["8"]["clean_tok_s_mean"])
        # significance of exact-match over FIFO at N=8, so "still improves" is tested, not assumed
        tt[shape] = welch([t[0] for t in d["raw"]["batch"]["8"]], [t[0] for t in d["raw"]["fifo"]["8"]])
    r["t"] = tt
    return r

@section("poisson")
def _():
    s = load(FILES["poisson"])["summaries"]
    Ls = sorted(s["fifo"], key=float)
    conds = ["fifo", "batch", "interleave", "progressive-sc"]
    table = {L: {c: dict(tok=s[c][L]["tok_s_mean"], tok_std=s[c][L]["tok_s_std"],
                         lat=s[c][L]["mean_latency_s_mean"], lat_std=s[c][L]["mean_latency_s_std"],
                         vs_fifo=pct(s[c][L]["tok_s_mean"], s["fifo"][L]["tok_s_mean"]))
                 for c in conds} for L in Ls}
    # Significance per load: throughput must beat both batching baselines; latency is
    # tested against the closest one.
    raw = load(FILES["poisson"])["raw"]
    sig = {}
    for L in Ls:
        dsb_tok = [t[0] for t in raw["progressive-sc"][L]]
        dsb_lat = [t[2] for t in raw["progressive-sc"][L]]
        lats = {c: st.mean(t[2] for t in raw[c][L]) for c in ("fifo", "batch", "interleave")}
        nearest = min(lats, key=lats.get)
        sig[L] = dict(t_vs_batch=welch(dsb_tok, [t[0] for t in raw["batch"][L]]),
                      t_vs_interleave=welch(dsb_tok, [t[0] for t in raw["interleave"][L]]),
                      latency_nearest=nearest,
                      t_latency_vs_nearest=welch([t[2] for t in raw[nearest][L]], dsb_lat))
    top = Ls[-1]
    return dict(table=table, significance=sig,
                throughput_lead_significant_at=[L for L in Ls if min(sig[L]["t_vs_batch"], sig[L]["t_vs_interleave"]) > 2.5],
                latency_lowest_significant_at=[L for L in Ls if sig[L]["t_latency_vs_nearest"] > 2.5],
                dsb_vs_batch_top=pct(s["progressive-sc"][top]["tok_s_mean"], s["batch"][top]["tok_s_mean"]),
                dsb_vs_interleave_top=pct(s["progressive-sc"][top]["tok_s_mean"], s["interleave"][top]["tok_s_mean"]),
                dsb_latency_range=[min(table[L]["progressive-sc"]["lat"] for L in Ls),
                                   max(table[L]["progressive-sc"]["lat"] for L in Ls)],
                fifo_latency_range=[min(table[L]["fifo"]["lat"] for L in Ls),
                                    max(table[L]["fifo"]["lat"] for L in Ls)],
                fifo_tok_mean=st.mean(table[L]["fifo"]["tok"] for L in Ls))

@section("tail")
def _():
    rows = tail_rows(load(FILES["tail"]))
    f = fit([r["tail"] for r in rows], [r["gain"] for r in rows])
    dsb = [r["dsb"]["mean"] for r in rows]; bat = [r["batch"]["mean"] for r in rows]
    fifo = [r["fifo"]["mean"] for r in rows]
    return dict(rows=[dict(tail=r["tail"], cuts=f"{r['maxcut']-7}--{r['maxcut']}",
                           fifo=[r["fifo"]["mean"], r["fifo"]["std"]],
                           batch=[r["batch"]["mean"], r["batch"]["std"]],
                           dsb=[r["dsb"]["mean"], r["dsb"]["std"]], gain=r["gain"],
                           t=welch_summary(r["dsb"], r["batch"])) for r in rows],
                r=f["r"], slope=f["slope"], zero_crossing=-f["intercept"] / f["slope"],
                monotonic=all(a["gain"] < b["gain"] for a, b in zip(rows, rows[1:])),
                dsb_near_constant=cv(dsb) < 6.0,          # 6% = six-trial noise floor (k=1 control)
                # does exact-match really fall below FIFO at the deepest fleet, or is it noise?
                deepest_batch_vs_fifo=pct(rows[0]["batch"]["mean"], rows[0]["fifo"]["mean"]),
                deepest_batch_vs_fifo_t=welch_summary(rows[0]["batch"], rows[0]["fifo"]),
                dsb_range=[min(dsb), max(dsb)], dsb_cv=cv(dsb),
                batch_range_pct=pct(max(bat), min(bat)), fifo_range_pct=pct(max(fifo), min(fifo)))

@section("diversity")
def _():
    d = load(FILES["div"]); ks = sorted(d, key=int)
    g = {int(k): pct(d[k]["progressive-sc"]["mean"], d[k]["batch"]["mean"]) for k in ks}
    fa = fit(list(g), list(g.values()))
    k2 = [k for k in g if k >= 2]; f2 = fit(k2, [g[k] for k in k2])
    return dict(gains=g, fit_all=dict(r=fa["r"], slope=fa["slope"]), fit_k_ge_2=dict(r=f2["r"], slope=f2["slope"]),
                arrays={c: [[d[k][c]["mean"], d[k][c]["std"]] for k in ks] for c in ("fifo", "batch", "progressive-sc")},
                dsb_cv=cv([d[k]["progressive-sc"]["mean"] for k in ks]),
                dsb_near_constant=cv([d[k]["progressive-sc"]["mean"] for k in ks]) < 6.0,
                monotonic_k_ge_2=all(g[a] < g[b] for a, b in zip(k2, k2[1:])),
                batch_decay_pct=100.0 * (1 - d[ks[-1]]["batch"]["mean"] / d[ks[0]]["batch"]["mean"]))

@section("crossover")
def _():
    P = pooled(FILES["sched"], ["fifo", "batch", "interleave", "progressive-sc"])
    r = {}
    for N in ("2", "4", "6", "8"):
        f = st.mean(P["fifo"][N])
        r[N] = dict(reps=len(P["fifo"][N]),
                    vs_fifo={c: pct(st.mean(P[c][N]), f) for c in P},
                    dsb_vs_interleave=pct(st.mean(P["progressive-sc"][N]), st.mean(P["interleave"][N])),
                    t_vs_interleave=welch(P["progressive-sc"][N], P["interleave"][N]),
                    dsb_vs_batch=pct(st.mean(P["progressive-sc"][N]), st.mean(P["batch"][N])),
                    t_vs_batch=welch(P["progressive-sc"][N], P["batch"][N]))
    # Operating-envelope thresholds: the smallest fleet size at which DSB is
    # significantly ahead. Computed, so the envelope wording is never eyeballed.
    first = lambda key: next((int(N) for N in ("2", "4", "6", "8") if r[N][key] > 2.5), None)
    r["threshold_vs_batch"] = first("t_vs_batch")
    r["threshold_vs_interleave"] = first("t_vs_interleave")
    return r

@section("matrix")
def _():
    d = load(FILES["matrix"]); rows = {}
    for k, v in d.items():
        rows[k] = dict(distinct=DISTINCT.get(k), gain=pct(v["progressive-sc"]["mean"], v["batch"]["mean"]),
                       t=welch_summary(v["progressive-sc"], v["batch"]))
    low = {k: r for k, r in rows.items() if r["distinct"] is not None and r["distinct"] <= 3}
    return dict(rows=rows, n_low=len(low), low_mean=st.mean(r["gain"] for r in low.values()),
                low_significant=[(k, r["gain"], r["t"]) for k, r in low.items() if abs(r["t"]) > 2.5],
                low_significant_negative=[(k, r["gain"], r["t"]) for k, r in low.items() if r["t"] < -2.5],
                spread=[(k, r["gain"], r["t"]) for k, r in rows.items() if r["distinct"] == 8],
                unmapped=[k for k, r in rows.items() if r["distinct"] is None])

@section("qwen")
def _():
    q = tail_rows(load(FILES["qwen"])); l = tail_rows(load(FILES["tail"]))
    fq = fit([r["tail"] for r in q], [r["gain"] for r in q])
    fl = fit([r["tail"] for r in l], [r["gain"] for r in l])
    nq, nl = fq["slope"] * L_QWEN / 100, fl["slope"] * L_LLAMA / 100
    return dict(r=fq["r"], slope=fq["slope"], slope_ratio=fq["slope"] / fl["slope"],
                monotonic=all(a["gain"] < b["gain"] for a, b in zip(q, q[1:])),
                shallowest_t=welch_summary(q[0]["dsb"], q[0]["batch"]),
                predicted_ratio=L_LLAMA / L_QWEN, norm_qwen=nq, norm_llama=nl,
                norm_discrepancy_pct=100 * abs(nq / nl - 1), shallowest_gain=q[0]["gain"],
                gains={r["tail"]: r["gain"] for r in q},
                ts={r["tail"]: welch_summary(r["dsb"], r["batch"]) for r in q})

def mem_table(fname, header):
    lines = open(path(fname)).read().splitlines()
    i = next(n for n, x in enumerate(lines) if x.strip().startswith(header))
    rows = {}
    for x in lines[i + 1:]:
        parts = x.split()
        if len(parts) != 5 or not parts[0].isdigit():
            break
        rows[int(parts[0])] = dict(cache_mb=float(parts[3]), pct_weights=float(parts[4].rstrip("%")))
    return rows

@section("memory")
def _():
    offA, onA = mem_table(FILES["mem"]["off"], "N "), mem_table(FILES["mem"]["on"], "N ")
    onB = mem_table(FILES["mem"]["on"], "max_new_tokens")
    growth = {N: pct(onA[N]["cache_mb"], offA[N]["cache_mb"]) for N in onA}
    lo, hi = min(onB), max(onB)
    per_tok_bytes = (onB[hi]["cache_mb"] - onB[lo]["cache_mb"]) * 1e6 / (hi - lo) / 4   # 4 sessions in part B
    analytic = 26 * 4096                                                                 # cut 2: 26 cached layers
    return dict(padding_growth_by_N=growth, padding_growth_range=[min(growth.values()), max(growth.values())],
                worst_pct_weights=max(r["pct_weights"] for r in onA.values()),
                genlen_pct_weights={t: onB[t]["pct_weights"] for t in onB},
                per_token_kb=per_tok_bytes / 1000, analytic_kb=analytic / 1000,
                per_token_agreement_pct=100 * abs(per_tok_bytes / analytic - 1))

@section("context")
def _():
    d = load(FILES["context"])
    items = sorted(d.items(), key=lambda kv: int(kv[0]))
    gains = {int(k): pct(v["progressive-sc"]["mean"], v["batch"]["mean"]) for k, v in items}
    t = {int(k): welch_summary(v["progressive-sc"], v["batch"]) for k, v in items}
    xs = sorted(gains)
    short = [gains[x] for x in xs if x <= 512]
    # claims the operating-envelope paragraph makes, each computed
    flat_upto_512 = (max(short) - min(short)) < 6.0            # within the six-trial noise floor
    above_5pct = [x for x in xs if gains[x] > 5.0 and t[x] > 2.5]
    x1, x2 = xs[-2], xs[-1]; g1, g2 = gains[x1], gains[x2]
    zero_x = x2 + g2 * (x2 - x1) / (g1 - g2) if g1 > g2 and g2 > 0 else None
    return dict(gains=gains, t=t, flat_upto_512=flat_upto_512, flat_band=[min(short), max(short)],
                significant_benefit_upto=max(above_5pct) if above_5pct else None,
                zero_crossing_extrapolated=zero_x)

def last_profile(fname):
    text = open(path(fname)).read()
    blocks = text.split("==== decode-step profile")
    if len(blocks) < 2:
        raise ValueError(f"no profile dump in {fname}")
    blk = blocks[-1]; phases = {}
    for m in re.finditer(r"^\s*(\w+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%", blk, re.M):
        phases[m.group(1)] = dict(ms=float(m.group(2)), calls=int(m.group(3)),
                                  us_per_call=float(m.group(4)), pct=float(m.group(5)))
    merged = re.search(r"mean sessions merged per \S+ step: ([\d.]+)", blk)
    return phases, float(merged.group(1)) if merged else None

@section("profile")
def _():
    (ps, ms), (pl, ml) = last_profile(FILES["profile"]["short"]), last_profile(FILES["profile"]["long"])
    def grow(p): return pl[p]["us_per_call"] / ps[p]["us_per_call"]
    def recon(p): return p["merge_caches_SLOW"]["pct"] + p["dissolve_range_group"]["pct"]
    def rebuild(p): return 100 * p["merge_caches_SLOW"]["calls"] / p["step_total"]["calls"]
    return dict(growth_forward=grow("forward_launch"), growth_merge=grow("merge_caches_SLOW"),
                growth_dissolve=grow("dissolve_range_group"),
                reconciliation_pct=dict(short=recon(ps), long=recon(pl)),
                sessions_merged=dict(short=ms, long=ml),
                rebuild_pct=dict(short=rebuild(ps), long=rebuild(pl)),
                steps=dict(short=ps["step_total"]["calls"], long=pl["step_total"]["calls"]))

@section("matrix_shape")
def _():
    """The <=3-cut configurations, split by what exact-match batching leaves unbatched:
    homogeneous (the two schedulers coincide), a majority plus exactly one outlier, and
    everything else (the minority is itself a pair, a cluster, or several outliers)."""
    from collections import Counter
    cats = dict(homogeneous=[], single_outlier=[], multi_minority=[])
    for k, v in load(FILES["matrix"]).items():
        if not isinstance(v, dict) or "batch" not in v:
            continue
        cuts = v["_meta"]["cut_layers"]; c = Counter(cuts)
        if len(c) > 3:
            continue
        sizes = sorted(c.values())
        cat = ("homogeneous" if len(c) == 1 else
               "single_outlier" if sizes == [1, len(cuts) - 1] else "multi_minority")
        o = v["_order"]
        cats[cat].append(dict(config=k, gain=pct(v["progressive-sc"]["mean"], v["batch"]["mean"]),
                              t=welch_summary(v["progressive-sc"], v["batch"]),
                              cuts={str(a): b for a, b in c.items()},
                              dsb_first=o.index("progressive-sc") < o.index("batch"),
                              minute=v["_meta"]["minutes_into_run"]))
    out = {}
    for name, rows in cats.items():
        g = [r["gain"] for r in rows]
        out[name] = dict(n=len(rows), gain_range=[min(g), max(g)], rows=rows,
                         n_sig_pos=sum(r["t"] >= 2.5 for r in rows), n_sig_neg=sum(r["t"] <= -2.5 for r in rows))
    low = [r for rows in cats.values() for r in rows]
    # Drift checks: do losses appear under both condition orders, and does gain track run
    # position once each category's mean is removed?
    out["sig_neg_dsb_first"] = sorted({r["dsb_first"] for r in low if r["t"] <= -2.5})
    out["gain_vs_run_position_r"] = fit([r["minute"] for r in low], [r["gain"] for r in low])["r"]
    resid = [r["gain"] - st.mean(x["gain"] for x in cats[c]) for c in cats for r in cats[c]]
    rr = fit([r["minute"] for c in cats for r in cats[c]], resid)["r"]
    out["within_category_run_position"] = dict(r=rr, t=rr * math.sqrt((len(low) - 3) / (1 - rr ** 2)))
    # The separate diversity sweep (shared tail fixed) as an independent check on clustered fleets.
    div = {}
    for k, v in load(FILES["div"]).items():
        if isinstance(v, dict) and "batch" in v:
            cuts = v["_meta"]["cut_layers"]
            div[k] = dict(gain=pct(v["progressive-sc"]["mean"], v["batch"]["mean"]),
                          t=welch_summary(v["progressive-sc"], v["batch"]),
                          groups=sorted(Counter(cuts).values()), tail=L_LLAMA - max(cuts))
    out["diversity_rows"] = div
    out["tails"] = {r["config"]: L_LLAMA - max(int(c) for c in r["cuts"]) for r in low}
    return out

@section("reproducibility")
def _():
    t20 = next(r["gain"] for r in OUT["tail"]["rows"] if r["tail"] == 20)
    return dict(tail20_gain=t20, diversity_k8_gain=OUT["diversity"]["gains"][8],
                diversity_k1_control=OUT["diversity"]["gains"][1])

@section("table4")
def _():
    tails = {r["tail"]: [r["gain"], r["t"]] for r in OUT["tail"]["rows"]}   # [gain, t] like the matrix rows
    mx = OUT["matrix"]["rows"]
    return dict(maj10_out2=[mx["s03_majmid10_out2"]["gain"], mx["s03_majmid10_out2"]["t"]],
                twoclust_2_10=[mx["s06_twoclust_2_10"]["gain"], mx["s06_twoclust_2_10"]["t"]],
                spread_20_27=tails[1], spread_13_20=tails[8], spread_1_8=tails[20])


# ---------------------------------------------------------------- report
def fmt(x):
    if isinstance(x, float):
        # correlations, ratios and normalized slopes need more than one decimal;
        # rounding r=0.933 to +0.9 is what hid a fit mismatch in the paper
        if abs(x) < 1: return f"{x:+.3f}"
        if abs(x) < 10: return f"{x:+.2f}"
        return f"{x:+.1f}" if abs(x) < 1000 else f"{x:.0f}"
    if isinstance(x, dict): return "{" + ", ".join(f"{k}: {fmt(v)}" for k, v in x.items()) + "}"
    if isinstance(x, (list, tuple)): return "[" + ", ".join(fmt(v) for v in x) + "]"
    return str(x)

skip_in_report = {"rows", "table", "arrays"}
print(f"=== paper numbers, family={FAMILY} ({FILES['gen_tokens']} generated tokens) ===")
for name, val in OUT.items():
    if not isinstance(val, dict):
        continue
    print(f"\n[{name}]")
    for k, v in val.items():
        if k in skip_in_report:
            continue
        print(f"  {k}: {fmt(v)}")
if MISSING:
    print("\nMISSING (not yet run or not yet written):")
    for m in MISSING:
        print("  -", m)
json.dump(OUT, open(os.path.join(HERE, "..", "results", f"analyzed_{FAMILY}.json"), "w"), indent=1, default=str)
print(f"\nwrote analyzed_{FAMILY}.json")
