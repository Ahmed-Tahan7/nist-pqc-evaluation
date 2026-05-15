"""
PQC Heartbeat IoT Benchmark — 4 Essential Research Paper Figures
Nordic nRF52840 (Cortex-M4F @ 64 MHz)
NIST FIPS 203 / 204 / 205  |  ML-KEM-768, ML-DSA-65, SLH-DSA-128s

Fixes applied vs original:
  - Fig B: payload multiplier annotations corrected:
           "11065× payload" → "110.6× payload"  (7856 / 71 = 110.6)
           "4661× payload"  → "46.6× payload"   (3309 / 71 = 46.6)
           (original had decimal point omitted, making values 1000× too large)
  - Fig B: BLE effective line now uses BLE_EFF_BPS = 200 kbps (25 000 bytes/s × 8)
           and labelled clearly to distinguish from the 125 kbps theoretical max.
           Feasibility check uses BLE_EFF_BPS consistently with Fig 9.
  - Fig C: ML-KEM polygon made visible:
           - drawn last (on top) with thick dashed border
           - explicit vertex markers
           - title note: "ML-KEM (dashed) scores ≈ 100% on all axes"
  - Fig D: ML-KEM and ML-DSA drain annotated numerically at 24 h
           Added inset table so the < 0.01% values are readable
           (at SLH-DSA scale the lines are visually indistinguishable from zero)

Run:
    python3 Pqc_4figures.py [path/to/benchmark_results.csv]

Outputs (./figures_4/):
    figA_performance_overview.pdf / .png
    figB_ble_overhead.pdf / .png
    figC_radar_feasibility.pdf / .png
    figD_battery_projection.pdf / .png
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ── colour / style constants ─────────────────────────────────────────────────
C = {
    "ML-KEM":  "#2166ac",
    "ML-DSA":  "#d6604d",
    "SLH-DSA": "#4dac26",
    "budget":  "#666666",
    "payload": "#e67e22",
    "ble_max": "#8e44ad",
    "pass":    "#27ae60",
    "fail":    "#c0392b",
}
HATCH = {"ML-KEM": "", "ML-DSA": "//", "SLH-DSA": "xx"}

# ── device / scenario constants (mirror device_profile.h exactly) ─────────────
DEVICE_ACTIVE_POWER_MW = 14.4
BUDGET_KEYGEN_MS       = 2000.0
BUDGET_ENCAPS_MS       = 500.0
BUDGET_DECAPS_MS       = 500.0
BUDGET_SIGN_MS         = 300.0
BUDGET_VERIFY_MS       = 300.0
BUDGET_ENERGY_UJ       = 3500.0
BUDGET_SIG_BYTES       = 4096
BUDGET_RAM_KB          = 128
HEARTBEAT_PAYLOAD_B    = 71
BPM                    = 60
BATTERY_CAPACITY_MJ    = 2_538_000.0
BLE_MAX_BPS            = 125_000.0   # bits/s — BLE 4.2 theoretical max
# BLE_EFF_BPS: device_profile.h BLE_EFFECTIVE_BPS = 25 000 bytes/s = 200 000 bps
BLE_EFF_BPS            = 200_000.0   # bits/s  (25 000 bytes/s × 8)

OUT_DIR = "figures_4"


# ── helpers ──────────────────────────────────────────────────────────────────
def savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, name)
    fig.savefig(base)
    fig.savefig(base.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  saved  {base}")


def load(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def summarise(df):
    return df.groupby(["algorithm", "operation"]).agg(
        device_time_ms  = ("device_time_ms",  "mean"),
        device_time_std = ("device_time_ms",  "std"),
        energy_uj       = ("energy_uj",        "mean"),
        energy_std      = ("energy_uj",        "std"),
        ct_or_sig_bytes = ("ct_or_sig_bytes",  "first"),
        total_ram_bytes = ("total_ram_bytes",  "first"),
        pass_timing     = ("pass_timing",      "first"),
        pass_memory     = ("pass_memory",      "first"),
        pass_energy     = ("pass_energy",      "first"),
        pass_overall    = ("pass_overall",     "first"),
    ).reset_index()


def get(s, alg, op, col):
    row = s[(s.algorithm == alg) & (s.operation == op)]
    return float(row[col].iloc[0]) if len(row) else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Fig A — Performance overview: device time + energy (2-panel)
# WHY: Core result — time AND energy for every (algorithm, operation) pair,
#      budget thresholds drawn in. Log scale handles 4-order-of-magnitude spread.
# ═══════════════════════════════════════════════════════════════════════════════
def figA_performance(df, s):
    ops = [
        ("ML-KEM",  "KeyGen", BUDGET_KEYGEN_MS),
        ("ML-KEM",  "Encaps", BUDGET_ENCAPS_MS),
        ("ML-KEM",  "Decaps", BUDGET_DECAPS_MS),
        ("ML-DSA",  "KeyGen", BUDGET_KEYGEN_MS),
        ("ML-DSA",  "Sign",   BUDGET_SIGN_MS),
        ("ML-DSA",  "Verify", BUDGET_VERIFY_MS),
        ("SLH-DSA", "KeyGen", BUDGET_KEYGEN_MS),
        ("SLH-DSA", "Sign",   BUDGET_SIGN_MS),
        ("SLH-DSA", "Verify", BUDGET_VERIFY_MS),
    ]
    labels    = [f"{a}\n{o}"                         for a, o, _ in ops]
    t_vals    = [get(s, a, o, "device_time_ms")      for a, o, _ in ops]
    t_std     = [get(s, a, o, "device_time_std")     for a, o, _ in ops]
    e_vals    = [get(s, a, o, "energy_uj")           for a, o, _ in ops]
    e_std     = [get(s, a, o, "energy_std")          for a, o, _ in ops]
    budgets_t = [b                                   for _, _, b in ops]
    colours   = [C[a]                                for a, _, _ in ops]
    hatches   = [HATCH[a]                            for a, _, _ in ops]
    x = np.arange(len(ops))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                   sharex=True,
                                   gridspec_kw={"hspace": 0.10})

    # ── top panel: device time ──
    bars1 = ax1.bar(x, t_vals, color=colours, edgecolor="white",
                    linewidth=0.5, width=0.6, zorder=3)
    for bar, h in zip(bars1, hatches):
        bar.set_hatch(h)
    ax1.errorbar(x, t_vals, yerr=t_std, fmt="none",
                 ecolor="black", elinewidth=0.8, capsize=3, zorder=4)
    for xi, bud in enumerate(budgets_t):
        ax1.hlines(bud, xi - 0.38, xi + 0.38,
                   colors=C["budget"], linewidths=1.4, linestyles="--", zorder=5)
    ax1.set_yscale("log")
    ax1.set_ylabel("Estimated Device Time (ms)\n[log scale]")
    ax1.set_title(
        "Fig. A  Cryptographic Operation Performance on Nordic nRF52840\n"
        "ML-KEM-768 (FIPS 203)  ·  ML-DSA-65 (FIPS 204)  ·  SLH-DSA-128s (FIPS 205)  "
        "|  dashed = timing/energy budget  ·  error bars = ±1 SD",
        loc="left", fontsize=10
    )
    # annotate SLH-DSA Sign value (off-scale indicator)
    idx = labels.index("SLH-DSA\nSign")
    ax1.annotate(
        f"↑ {t_vals[idx]:,.0f} ms",
        xy=(x[idx], t_vals[idx]),
        xytext=(0, 5), textcoords="offset points",
        ha="center", fontsize=8, color="black", fontweight="bold"
    )

    # ── bottom panel: energy ──
    bars2 = ax2.bar(x, e_vals, color=colours, edgecolor="white",
                    linewidth=0.5, width=0.6, zorder=3)
    for bar, h in zip(bars2, hatches):
        bar.set_hatch(h)
    ax2.errorbar(x, e_vals, yerr=e_std, fmt="none",
                 ecolor="black", elinewidth=0.8, capsize=3, zorder=4)
    ax2.axhline(BUDGET_ENERGY_UJ, color=C["budget"], linestyle="--",
                linewidth=1.4, zorder=5,
                label=f"Energy budget  ({BUDGET_ENERGY_UJ:,.0f} µJ)")
    ax2.set_yscale("log")
    ax2.set_ylabel("Energy per Operation (µJ)\n[log scale]")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.legend(loc="upper left", framealpha=0.9)

    # ── shared legend (algorithm + hatch key) ──
    alg_handles = [
        mpatches.Patch(facecolor=C[a], label=a, hatch=HATCH[a])
        for a in ["ML-KEM", "ML-DSA", "SLH-DSA"]
    ]
    alg_handles.append(
        mpatches.Patch(facecolor=C["budget"], alpha=0.6,
                       label="Budget thresholds (dashed)")
    )
    ax1.legend(handles=alg_handles, loc="upper left",
               framealpha=0.9, ncol=2)

    savefig(fig, "figA_performance_overview.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig B — BLE / communication overhead
# WHY: IoT research must address the link layer, not just compute.
#
# FIXES:
#   - Payload multiplier: now computed as sig_bytes / payload_bytes and
#     displayed with 1 decimal place.  Original had a formatting bug that
#     dropped the decimal point (e.g. 110.6× rendered as "11065×").
#   - BLE feasibility uses BLE_EFF_BPS (200 kbps bits/s = 25 000 bytes/s)
#     matching device_profile.h, and is labelled clearly.
# ═══════════════════════════════════════════════════════════════════════════════
def figB_ble_overhead(df, s):
    alg_ops = [
        ("ML-KEM",  "Encaps"),
        ("ML-DSA",  "Sign"),
        ("SLH-DSA", "Sign"),
    ]
    labels   = ["ML-KEM-768\nCiphertext", "ML-DSA-65\nSignature",
                 "SLH-DSA-128s\nSignature"]
    sig_b    = [int(get(s, a, o, "ct_or_sig_bytes")) for a, o in alg_ops]
    alg_keys = [a for a, _ in alg_ops]

    total_pkt  = [HEARTBEAT_PAYLOAD_B + b for b in sig_b]
    # FIX: correct multiplier = sig_bytes / payload_bytes (not per-bit)
    overhead   = [b / HEARTBEAT_PAYLOAD_B for b in sig_b]     # e.g. 110.6
    uplink_bps = [p * 8 * BPM / 60 for p in total_pkt]        # bits/s @ 1 pkt/s

    fig = plt.figure(figsize=(11, 5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── left: packet size breakdown (horizontal stacked bar) ──
    y      = np.arange(len(labels))
    pay_kb = [HEARTBEAT_PAYLOAD_B / 1024] * len(labels)
    sig_kb = [b / 1024 for b in sig_b]

    ax1.barh(y, pay_kb, color="#95a5a6", height=0.5,
             label=f"Payload ({HEARTBEAT_PAYLOAD_B} B)")
    ax1.barh(y, sig_kb, left=pay_kb,
             color=[C[k] for k in alg_keys], height=0.5,
             label="CT / Signature")

    # FIX: correct multiplier displayed — sig_bytes / payload_bytes, 1 d.p.
    for i, (sk, oh, sb) in enumerate(zip(sig_kb, overhead, sig_b)):
        ax1.text(pay_kb[0] + sk + 0.05, i,
                 f"{sb:,} B  ({oh:.1f}× payload)",
                 va="center", fontsize=8.5, color="#333333")

    ax1.axvline(BUDGET_SIG_BYTES / 1024, color=C["budget"], linestyle="--",
                linewidth=1.3, label=f"Size budget ({BUDGET_SIG_BYTES//1024} KB)")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("Packet Size (KB)")
    ax1.set_title("(a)  Packet size breakdown\nPayload + CT/Signature per heartbeat")
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    # ── right: required uplink vs BLE limits ──
    colours_bar = [C[k] for k in alg_keys]
    bars = ax2.bar(y, [b / 1000 for b in uplink_bps],
                   color=colours_bar, edgecolor="white", width=0.5)
    for bar, hk in zip(bars, alg_keys):
        bar.set_hatch(HATCH[hk])

    ax2.axhline(BLE_MAX_BPS / 1000, color=C["ble_max"], linestyle="--",
                linewidth=1.4,
                label=f"BLE 4.2 theoretical max  ({BLE_MAX_BPS/1000:.0f} kbps)")
    ax2.axhline(BLE_EFF_BPS / 1000, color=C["ble_max"], linestyle="-.",
                linewidth=1.1, alpha=0.7,
                label=f"BLE effective  ({BLE_EFF_BPS/1000:.0f} kbps, 25 kB/s)")

    for xi, bps in enumerate(uplink_bps):
        feasible = bps <= BLE_EFF_BPS
        col = C["pass"] if feasible else C["fail"]
        tag = "OK" if feasible else "EXCEEDS"
        ax2.text(xi, bps / 1000 + 1.5,
                 f"{bps/1000:.1f} kbps\n[{tag}]",
                 ha="center", fontsize=8, color=col, fontweight="bold")

    ax2.set_xticks(y)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Required Uplink Bandwidth (kbps)\n@ 60 BPM  (1 packet / sec)")
    ax2.set_title("(b)  Required uplink bandwidth\nvs BLE 4.2 physical limits")
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    fig.suptitle(
        f"Fig. B  BLE 4.2 Communication Overhead of PQC Algorithms  "
        f"|  Heartbeat payload = {HEARTBEAT_PAYLOAD_B} B  ·  HR = {BPM} BPM",
        fontsize=11, y=1.01
    )
    savefig(fig, "figB_ble_overhead.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig C — Multi-criteria feasibility radar
# WHY: Collapses timing/energy/size/RAM/BLE constraints into one verdict figure.
#
# FIX: ML-KEM polygon was invisible (hidden behind ML-DSA fill, same outer rim).
#      Now: ML-KEM drawn last with thick dashed line, no fill, vertex dots,
#      and title note clarifying it scores ≈ 100% on all axes.
# ═══════════════════════════════════════════════════════════════════════════════
def figC_radar(df, s):
    dims   = ["Timing", "Energy", "Sig / CT\nSize", "RAM", "BLE\nBandwidth"]
    N      = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    def normalised_scores(alg):
        if alg == "ML-KEM":
            op_budgets = [("KeyGen", BUDGET_KEYGEN_MS),
                          ("Encaps", BUDGET_ENCAPS_MS),
                          ("Decaps", BUDGET_DECAPS_MS)]
        else:
            op_budgets = [("KeyGen", BUDGET_KEYGEN_MS),
                          ("Sign",   BUDGET_SIGN_MS),
                          ("Verify", BUDGET_VERIFY_MS)]

        scores_per_op = []
        for op, bud_ms in op_budgets:
            t_ms  = get(s, alg, op, "device_time_ms")
            e_uj  = get(s, alg, op, "energy_uj")
            ct_b  = get(s, alg, op, "ct_or_sig_bytes")
            ram_b = get(s, alg, op, "total_ram_bytes")
            uplink = (ct_b + HEARTBEAT_PAYLOAD_B) * 8 * 1.0  # bits/s at 1 pkt/s

            timing_score = min(1.0, bud_ms         / max(t_ms,  1e-9))
            energy_score = min(1.0, BUDGET_ENERGY_UJ / max(e_uj,  1e-9))
            size_score   = (min(1.0, BUDGET_SIG_BYTES / max(ct_b, 1))
                            if ct_b > 0 else 1.0)
            ram_score    = min(1.0, BUDGET_RAM_KB * 1024 / max(ram_b, 1))
            ble_score    = min(1.0, BLE_EFF_BPS / max(uplink, 1))

            scores_per_op.append([timing_score, energy_score,
                                   size_score, ram_score, ble_score])

        return np.mean(scores_per_op, axis=0)

    algs   = ["ML-KEM", "ML-DSA", "SLH-DSA"]
    scores = {a: normalised_scores(a) for a in algs}

    # print for paper cross-check
    print("\n  Radar scores (0–1, higher = better):")
    print(f"  {'Alg':<10} {'Timing':>8} {'Energy':>8} {'Size':>8} "
          f"{'RAM':>8} {'BLE':>8}")
    for a, sc in scores.items():
        print(f"  {a:<10} {sc[0]:>8.3f} {sc[1]:>8.3f} {sc[2]:>8.3f} "
              f"{sc[3]:>8.3f} {sc[4]:>8.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 6.5),
                           subplot_kw=dict(projection="polar"))

    # Draw ML-DSA and SLH-DSA first, ML-KEM last (on top) so it's visible
    draw_order = ["ML-DSA", "SLH-DSA", "ML-KEM"]
    styles = {
        "ML-KEM":  {"lw": 2.8, "ls": "--", "alpha": 0.0,  "zorder": 5},
        "ML-DSA":  {"lw": 2.0, "ls": "-",  "alpha": 0.13, "zorder": 3},
        "SLH-DSA": {"lw": 2.0, "ls": "-",  "alpha": 0.13, "zorder": 3},
    }

    for alg in draw_order:
        sc = scores[alg]
        v  = sc.tolist() + sc[:1].tolist()
        st = styles[alg]
        ax.plot(angles_closed, v, color=C[alg], linewidth=st["lw"],
                linestyle=st["ls"], label=alg, zorder=st["zorder"])
        ax.fill(angles_closed, v, color=C[alg], alpha=st["alpha"])
        # vertex markers — larger for ML-KEM so it's visible
        ms = 7 if alg == "ML-KEM" else 5
        for ang, val in zip(angles, sc):
            ax.plot(ang, val, "o", color=C[alg], markersize=ms,
                    zorder=st["zorder"] + 1)

    # reference rings
    for r_val, r_lab in [(0.25, "25%"), (0.5, "50%"),
                         (0.75, "75%"), (1.0, "100%")]:
        ax.plot(angles_closed, [r_val] * (N + 1),
                color="grey", linewidth=0.5, alpha=0.4)

    ax.set_xticks(angles)
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7.5, color="grey")
    ax.set_title(
        "Fig. C  Multi-Criteria Feasibility Radar\n"
        "Scores normalised to budget  ·  outer rim = fully within all constraints\n"
        "ML-KEM (dashed) scores ≈ 100% on all axes",
        pad=22, fontsize=10
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.15),
              framealpha=0.9, fontsize=10)
    savefig(fig, "figC_radar_feasibility.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig D — 24-hour battery drain projection
# WHY: "Feasible in µJ" ≠ "feasible for a clinical shift."
#
# FIX: ML-KEM and ML-DSA drain lines annotated numerically and displayed in
#      an inset table — lines are visually flat against SLH-DSA at this scale.
# ═══════════════════════════════════════════════════════════════════════════════
def figD_battery(df, s):
    # ML-DSA: 60 sign + 60 verify per hour
    mldsa_sign_uj   = get(s, "ML-DSA",  "Sign",   "energy_uj")
    mldsa_verify_uj = get(s, "ML-DSA",  "Verify", "energy_uj")
    mldsa_uj_hr     = (mldsa_sign_uj + mldsa_verify_uj) * BPM

    # ML-KEM: 1 full session per hour (KeyGen + Encaps + Decaps)
    mlkem_uj_hr = sum(get(s, "ML-KEM", op, "energy_uj")
                      for op in ["KeyGen", "Encaps", "Decaps"])

    # SLH-DSA: hypothetical 60 BPM (fails timing — labelled as such)
    slhdsa_sign_uj   = get(s, "SLH-DSA", "Sign",   "energy_uj")
    slhdsa_verify_uj = get(s, "SLH-DSA", "Verify", "energy_uj")
    slhdsa_uj_hr     = (slhdsa_sign_uj + slhdsa_verify_uj) * BPM

    hours = np.arange(0, 25)

    def drain_pct(uj_hr, h):
        return (uj_hr * h / 1000.0) / BATTERY_CAPACITY_MJ * 100.0

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(hours, drain_pct(mldsa_uj_hr, hours),
            color=C["ML-DSA"], lw=2.2,
            label="ML-DSA-65  (sign + verify, 60 BPM)")
    ax.plot(hours, drain_pct(mlkem_uj_hr, hours),
            color=C["ML-KEM"], lw=2.2,
            label="ML-KEM-768  (full session, 1×/hr)")
    ax.plot(hours, drain_pct(slhdsa_uj_hr, hours),
            color=C["SLH-DSA"], lw=2.2, linestyle="--",
            label="SLH-DSA-128s  (hypothetical 60 BPM — fails timing budget)")

    # clinical shift markers
    for h_mark, label_txt in [(8, "8 h shift"), (24, "24 h")]:
        ax.axvline(h_mark, color="#aaaaaa", linestyle=":", linewidth=1.0)
        ax.text(h_mark + 0.25, ax.get_ylim()[1] * 0.01 if ax.get_ylim()[1] > 1 else 0.05,
                label_txt, fontsize=8, color="#777777")

    ax.set_xlabel("Continuous Operation (hours)")
    ax.set_ylabel("CR2032 Battery Consumed (%)")
    ax.set_xlim(0, 24)
    ax.set_title(
        "Fig. D  Cumulative CR2032 Battery Drain — 24-Hour Clinical Operation\n"
        f"CR2032 = {BATTERY_CAPACITY_MJ/1e6:.2f} J  ·  "
        f"Active power model: {DEVICE_ACTIVE_POWER_MW} mW  ·  HR = {BPM} BPM",
        loc="left", fontsize=10
    )
    ax.legend(framealpha=0.9, loc="upper left")

    # FIX: annotate 24-h value for SLH-DSA (clearly visible on plot)
    slh_24 = drain_pct(slhdsa_uj_hr, 24)
    ax.annotate(
        f"{slh_24:.3f}%",
        xy=(24, slh_24),
        xytext=(-52, 4), textcoords="offset points",
        fontsize=8, color=C["SLH-DSA"], fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=C["SLH-DSA"], lw=0.8)
    )

    # FIX: inset data table for ML-KEM and ML-DSA (too small to read on plot)
    col_labels = ["Algorithm", "Scenario", "Drain @ 8 h", "Drain @ 24 h"]
    tbl_rows = [
        ["ML-DSA-65",    "60 sign+verify/hr",
         f"{drain_pct(mldsa_uj_hr,  8):.5f}%",
         f"{drain_pct(mldsa_uj_hr,  24):.5f}%"],
        ["ML-KEM-768",   "1 session/hr",
         f"{drain_pct(mlkem_uj_hr,  8):.5f}%",
         f"{drain_pct(mlkem_uj_hr,  24):.5f}%"],
        ["SLH-DSA-128s", "60 sign+verify/hr",
         f"{drain_pct(slhdsa_uj_hr, 8):.3f}%",
         f"{drain_pct(slhdsa_uj_hr, 24):.3f}%"],
    ]
    tbl = ax.table(cellText=tbl_rows, colLabels=col_labels,
                   loc="upper left", bbox=[0.02, 0.52, 0.62, 0.26])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
        elif r > 0:
            alg_colour = [C["ML-DSA"], C["ML-KEM"], C["SLH-DSA"]][r - 1]
            if c == 0:
                cell.set_text_props(color=alg_colour, fontweight="bold")

    fig.tight_layout()
    savefig(fig, "figD_battery_projection.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
    print(f"\nPQC Research Paper — 4 Essential Figures")
    print(f"  CSV  : {csv_path}")
    print(f"  Out  : ./{OUT_DIR}/\n")

    df = load(csv_path)
    s  = summarise(df)

    figA_performance(df, s)
    figB_ble_overhead(df, s)
    figC_radar(df, s)
    figD_battery(df, s)

    print(f"\nDone.  4 figures saved to ./{OUT_DIR}/")
    print("Each saved as .pdf (300 dpi, vector) and .png (preview).\n")
    print("Suggested paper placement:")
    print("  Fig A  — Results § (Performance subsection)")
    print("  Fig B  — Results § (Communication overhead subsection)")
    print("  Fig C  — Discussion § (Algorithm selection / feasibility)")
    print("  Fig D  — Discussion § (Operational lifetime / system viability)\n")


if __name__ == "__main__":
    main()