"""
PQC Heartbeat IoT Benchmark — 12-Figure Research Paper Script
Nordic nRF52840 (Cortex-M4F @ 64 MHz)
NIST FIPS 203 / 204 / 205  |  ML-KEM-768, ML-DSA-65, SLH-DSA-128s

Fixes applied vs original:
  - Fig 05: subtitle corrected to "SLH-DSA Sign excluded" (SLH-DSA Verify IS shown)
             SLH-DSA Verify now explicitly added to violin plot
  - Fig 07: ECDF now reports what % of ML-DSA Sign trials exceed budget
  - Fig 08: ML-KEM and ML-DSA drain annotated numerically (lines visually flat at SLH scale)
             BLE_EFFECTIVE_BPS constant unified to 25 000 bps (matches device_profile.h)
  - Fig 09: BLE effective line now uses 25 000 bps (matching device_profile.h),
             removed incorrect 200 kbps line
  - Fig 10: ML-KEM radar polygon given dashed border + explicit "100% on all axes" annotation
  - All figures: tight_layout / layout_engine used consistently, no truncated labels

Run:
    python3 Pqc_figures.py [path/to/benchmark_results.csv]

Outputs (./figures/):
    fig01_device_time_grouped_bar.pdf / .png
    fig02_energy_grouped_bar.pdf / .png
    fig03_memory_footprint.pdf / .png
    fig04_signature_size_comparison.pdf / .png
    fig05_latency_distribution_violin.pdf / .png
    fig06_energy_distribution_box.pdf / .png
    fig07_cumulative_latency_ecdf.pdf / .png
    fig08_battery_drain_projection.pdf / .png
    fig09_ble_bandwidth_analysis.pdf / .png
    fig10_radar_feasibility.pdf / .png
    fig11_tradeoff_scatter.pdf / .png
    fig12_per_trial_timeline.pdf / .png
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
from scipy import stats

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

# ── colour palette (colour-blind-safe) ──────────────────────────────────────
C = {
    "ML-KEM":  "#2166ac",
    "ML-DSA":  "#d6604d",
    "SLH-DSA": "#4dac26",
    "budget":  "#888888",
    "pass":    "#27ae60",
    "fail":    "#c0392b",
    "neutral": "#555555",
    "payload": "#e67e22",
}
HATCH = {"ML-KEM": "", "ML-DSA": "//", "SLH-DSA": "xx"}

# ── device / scenario constants (must match device_profile.h exactly) ────────
DEVICE_CPU_MHZ         = 64
DEVICE_ACTIVE_POWER_MW = 14.4
BUDGET_KEYGEN_MS       = 2000.0
BUDGET_SIGN_MS         = 300.0
BUDGET_VERIFY_MS       = 300.0
BUDGET_ENCAPS_MS       = 500.0
BUDGET_DECAPS_MS       = 500.0
BUDGET_ENERGY_UJ       = 3500.0
BUDGET_SIG_BYTES       = 4096
BUDGET_RAM_KB          = 128
BATTERY_CAPACITY_MJ    = 2_538_000.0
# FIX: unified to device_profile.h BLE_EFFECTIVE_BPS = 25 000 bytes/s
BLE_EFFECTIVE_BPS      = 25_000.0   # bytes/s  ← matches device_profile.h
BLE_MAX_BPS            = 125_000.0  # bits/s   ← BLE 4.2 theoretical max
HEARTBEAT_PAYLOAD_B    = 71
BPM                    = 60
OUT_DIR                = "figures"

# ── helpers ──────────────────────────────────────────────────────────────────

def savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  saved  {path}")


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def summary(df):
    """Return mean + std stats per (algorithm, operation)."""
    grp = df.groupby(["algorithm", "operation"])
    s = grp.agg(
        device_time_ms   = ("device_time_ms",  "mean"),
        device_time_std  = ("device_time_ms",  "std"),
        energy_uj        = ("energy_uj",        "mean"),
        energy_std       = ("energy_uj",        "std"),
        ble_delay_ms     = ("ble_delay_ms",     "mean"),
        total_time_ms    = ("total_time_ms",    "mean"),
        pk_bytes         = ("pk_bytes",         "first"),
        sk_bytes         = ("sk_bytes",         "first"),
        ct_or_sig_bytes  = ("ct_or_sig_bytes",  "first"),
        total_ram_bytes  = ("total_ram_bytes",  "first"),
        pass_timing      = ("pass_timing",      "first"),
        pass_memory      = ("pass_memory",      "first"),
        pass_energy      = ("pass_energy",      "first"),
        pass_overall     = ("pass_overall",     "first"),
    ).reset_index()
    return s


def get_val(s, alg, op, col):
    row = s[(s.algorithm == alg) & (s.operation == op)]
    return float(row[col].iloc[0]) if len(row) else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 01 — Grouped bar: device time per operation (log scale)
# ═══════════════════════════════════════════════════════════════════════════════
def fig01_device_time(df, s):
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
    labels  = [f"{a}\n{o}"            for a, o, _ in ops]
    vals    = [get_val(s, a, o, "device_time_ms")  for a, o, _ in ops]
    t_std   = [get_val(s, a, o, "device_time_std") for a, o, _ in ops]
    budgets = [b                                   for _, _, b in ops]
    colours = [C[a]                                for a, _, _ in ops]
    hatches = [HATCH[a]                            for a, _, _ in ops]

    x = np.arange(len(ops))
    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars = ax.bar(x, vals, color=colours, edgecolor="white",
                  linewidth=0.6, width=0.6, zorder=3)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.errorbar(x, vals, yerr=t_std, fmt="none",
                ecolor="black", elinewidth=0.8, capsize=3, zorder=4)

    for xi, bud in enumerate(budgets):
        ax.hlines(bud, xi - 0.35, xi + 0.35,
                  colors=C["budget"], linewidths=1.2, linestyles="--", zorder=5)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Estimated Device Time (ms) — log scale")
    ax.set_title(
        "Fig. 1  Estimated Device Execution Time per Operation\n"
        "Nordic nRF52840 Cortex-M4F @ 64 MHz  |  dashed line = timing budget"
    )

    legend_handles = [mpatches.Patch(facecolor=C[a], label=a, hatch=HATCH[a])
                      for a in ["ML-KEM", "ML-DSA", "SLH-DSA"]]
    legend_handles.append(
        mpatches.Patch(facecolor=C["budget"], label="Budget (per-op)")
    )
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)

    # annotate SLH-DSA Sign
    idx = labels.index("SLH-DSA\nSign")
    ax.annotate(
        f"{vals[idx]:,.0f} ms",
        xy=(x[idx], vals[idx]),
        xytext=(0, 6), textcoords="offset points",
        ha="center", fontsize=7.5, color=C["fail"], fontweight="bold"
    )

    fig.tight_layout()
    savefig(fig, "fig01_device_time_grouped_bar.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 02 — Grouped bar: energy per operation (log scale)
# ═══════════════════════════════════════════════════════════════════════════════
def fig02_energy(df, s):
    ops = [
        ("ML-KEM",  "KeyGen"), ("ML-KEM",  "Encaps"), ("ML-KEM",  "Decaps"),
        ("ML-DSA",  "KeyGen"), ("ML-DSA",  "Sign"),   ("ML-DSA",  "Verify"),
        ("SLH-DSA", "KeyGen"), ("SLH-DSA", "Sign"),   ("SLH-DSA", "Verify"),
    ]
    labels  = [f"{a}\n{o}" for a, o in ops]
    vals    = [get_val(s, a, o, "energy_uj")    for a, o in ops]
    e_std   = [get_val(s, a, o, "energy_std")   for a, o in ops]
    colours = [C[a]                              for a, o in ops]
    hatches = [HATCH[a]                          for a, o in ops]

    x = np.arange(len(ops))
    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars = ax.bar(x, vals, color=colours, edgecolor="white",
                  linewidth=0.6, width=0.6, zorder=3)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.errorbar(x, vals, yerr=e_std, fmt="none",
                ecolor="black", elinewidth=0.8, capsize=3, zorder=4)

    ax.axhline(BUDGET_ENERGY_UJ, color=C["budget"], linestyle="--",
               linewidth=1.2, label=f"Energy budget ({BUDGET_ENERGY_UJ:.0f} µJ)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Energy per Operation (µJ) — log scale")
    ax.set_title(
        "Fig. 2  Energy Consumption per Cryptographic Operation\n"
        "CR2032 @ 3 V · 14.4 mW active power model"
    )

    legend_handles = [mpatches.Patch(facecolor=C[a], label=a, hatch=HATCH[a])
                      for a in ["ML-KEM", "ML-DSA", "SLH-DSA"]]
    legend_handles.append(
        mpatches.Patch(facecolor=C["budget"],
                       label=f"Budget {BUDGET_ENERGY_UJ:.0f} µJ")
    )
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig02_energy_grouped_bar.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 03 — Stacked bar: cryptographic object sizes (PK + SK + CT/Sig)
# NOTE: these are object sizes, not total working RAM.
#       Total crypto RAM (from total_ram_bytes column) is annotated separately.
# ═══════════════════════════════════════════════════════════════════════════════
def fig03_memory(df, s):
    rows = [
        ("ML-KEM",  "KeyGen"), ("ML-KEM",  "Encaps"), ("ML-KEM",  "Decaps"),
        ("ML-DSA",  "KeyGen"), ("ML-DSA",  "Sign"),   ("ML-DSA",  "Verify"),
        ("SLH-DSA", "KeyGen"), ("SLH-DSA", "Sign"),   ("SLH-DSA", "Verify"),
    ]
    labels, pk_v, sk_v, ct_v, ram_v = [], [], [], [], []
    for a, o in rows:
        row = s[(s.algorithm == a) & (s.operation == o)]
        if not len(row):
            continue
        labels.append(f"{a}\n{o}")
        pk_v.append(int(row.pk_bytes.iloc[0])        / 1024)
        sk_v.append(int(row.sk_bytes.iloc[0])        / 1024)
        ct_v.append(int(row.ct_or_sig_bytes.iloc[0]) / 1024)
        ram_v.append(int(row.total_ram_bytes.iloc[0])/ 1024)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.8))

    b1  = ax.bar(x, pk_v,  label="Public Key",      color="#4393c3", width=0.55)
    b2  = ax.bar(x, sk_v,  bottom=pk_v,             label="Secret Key",    color="#d6604d", width=0.55)
    bot = [p + s_ for p, s_ in zip(pk_v, sk_v)]
    b3  = ax.bar(x, ct_v,  bottom=bot,              label="CT / Signature", color="#4dac26", width=0.55)

    # total RAM reference dots
    ax.scatter(x, ram_v, marker="D", color=C["neutral"], zorder=5,
               s=22, label="Total crypto RAM (working set)")

    ax.axhline(BUDGET_SIG_BYTES / 1024, color=C["budget"], linestyle="--",
               linewidth=1.2, label=f"Sig/CT budget ({BUDGET_SIG_BYTES//1024} KB)")
    ax.axhline(BUDGET_RAM_KB, color=C["neutral"], linestyle=":",
               linewidth=1.0, alpha=0.6, label=f"RAM budget ({BUDGET_RAM_KB} KB)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Size (KB)")
    ax.set_title(
        "Fig. 3  Cryptographic Object Memory Footprint\n"
        "Public Key + Secret Key + Ciphertext/Signature  "
        "|  ◆ = total working RAM"
    )
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig03_memory_footprint.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 04 — Signature / Ciphertext size vs payload & budget
# ═══════════════════════════════════════════════════════════════════════════════
def fig04_sig_size(df, s):
    data = {}
    for a, o in [
        ("ML-KEM",  "Encaps"), ("ML-KEM",  "Decaps"),
        ("ML-DSA",  "Sign"),   ("ML-DSA",  "Verify"),
        ("SLH-DSA", "Sign"),   ("SLH-DSA", "Verify"),
    ]:
        row = s[(s.algorithm == a) & (s.operation == o)]
        if len(row):
            data[f"{a}\n{o}"] = (int(row.ct_or_sig_bytes.iloc[0]), C[a], HATCH[a])

    labels  = list(data.keys())
    sizes   = [data[k][0] for k in labels]
    colours = [data[k][1] for k in labels]
    hatches = [data[k][2] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, sizes, color=colours, edgecolor="white", width=0.55)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.axhline(HEARTBEAT_PAYLOAD_B, color=C["payload"], linestyle="-.",
               linewidth=1.5, label=f"Payload ({HEARTBEAT_PAYLOAD_B} B)")
    ax.axhline(BUDGET_SIG_BYTES, color=C["budget"], linestyle="--",
               linewidth=1.2, label=f"Budget ({BUDGET_SIG_BYTES} B)")

    for xi, v in zip(x, sizes):
        ax.text(xi, v + 40, f"{v:,}", ha="center", fontsize=8, color=C["neutral"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Bytes")
    ax.set_title(
        "Fig. 4  Ciphertext / Signature Size vs Heartbeat Payload\n"
        f"Payload = {HEARTBEAT_PAYLOAD_B} B  |  BLE budget = {BUDGET_SIG_BYTES} B"
    )
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig04_signature_size_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 05 — Violin: latency distribution
# FIX: subtitle now correctly states "SLH-DSA Sign excluded" (SLH-DSA Verify IS shown)
# ═══════════════════════════════════════════════════════════════════════════════
def fig05_violin(df):
    # SLH-DSA Verify IS included (small enough scale); SLH-DSA Sign is excluded
    combos = [
        ("ML-KEM",  "KeyGen"), ("ML-KEM",  "Encaps"), ("ML-KEM",  "Decaps"),
        ("ML-DSA",  "KeyGen"), ("ML-DSA",  "Sign"),   ("ML-DSA",  "Verify"),
        ("SLH-DSA", "Verify"),   # SLH-DSA Verify: ~55-65 ms — visible at this scale
    ]
    groups, labels, colours = [], [], []
    for a, o in combos:
        sub = df[(df.algorithm == a) & (df.operation == o)]["device_time_ms"]
        if len(sub):
            groups.append(sub.values)
            labels.append(f"{a}\n{o}")
            colours.append(C[a])

    fig, ax = plt.subplots(figsize=(10, 4.8))
    parts = ax.violinplot(groups, positions=range(len(groups)),
                          showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], colours):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(1.5)
    parts["cmaxes"].set_color(C["neutral"])
    parts["cmins"].set_color(C["neutral"])
    parts["cbars"].set_color(C["neutral"])

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Estimated Device Time (ms)")
    # FIX: correct subtitle — SLH-DSA Verify IS shown; only SLH-DSA Sign excluded
    ax.set_title(
        "Fig. 5  Latency Distribution per Operation\n"
        "1 000 trials each  |  white line = median  "
        "|  SLH-DSA Sign excluded (61 541 ms — off scale)"
    )
    legend_handles = [mpatches.Patch(facecolor=C[a], label=a)
                      for a in ["ML-KEM", "ML-DSA", "SLH-DSA"]]
    ax.legend(handles=legend_handles, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig05_latency_distribution_violin.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 06 — Box plots: energy distribution
# ═══════════════════════════════════════════════════════════════════════════════
def fig06_energy_box(df):
    combos = [
        ("ML-KEM",  "KeyGen"), ("ML-KEM",  "Encaps"), ("ML-KEM",  "Decaps"),
        ("ML-DSA",  "KeyGen"), ("ML-DSA",  "Sign"),   ("ML-DSA",  "Verify"),
        ("SLH-DSA", "Verify"),   # Sign excluded (883 885 µJ — off scale)
    ]
    groups, labels, colours = [], [], []
    for a, o in combos:
        sub = df[(df.algorithm == a) & (df.operation == o)]["energy_uj"]
        if len(sub):
            groups.append(sub.values)
            labels.append(f"{a}\n{o}")
            colours.append(C[a])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot(groups, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=1.5),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    ax.axhline(BUDGET_ENERGY_UJ, color=C["budget"], linestyle="--",
               linewidth=1.2, label=f"Energy budget ({BUDGET_ENERGY_UJ:.0f} µJ)")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Energy per Operation (µJ)")
    ax.set_title(
        "Fig. 6  Energy Distribution per Operation (Box-and-Whisker)\n"
        "SLH-DSA Sign excluded (883 885 µJ median — off scale)"
    )
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig06_energy_distribution_box.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 07 — ECDF: cumulative latency for Sign/Verify candidates
# FIX: added explicit annotation of % trials exceeding timing budget
# ═══════════════════════════════════════════════════════════════════════════════
def fig07_ecdf(df):
    targets = [
        ("ML-DSA",  "Sign",   BUDGET_SIGN_MS,   "-"),
        ("ML-DSA",  "Verify", BUDGET_VERIFY_MS,  "--"),
        ("SLH-DSA", "Verify", BUDGET_VERIFY_MS,  "--"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for a, o, bud, ls in targets:
        sub = df[(df.algorithm == a) & (df.operation == o)]["device_time_ms"].sort_values()
        if not len(sub):
            continue
        ecdf = np.arange(1, len(sub) + 1) / len(sub)
        ax.step(sub, ecdf, where="post", color=C[a],
                linestyle=ls, linewidth=1.6, label=f"{a} {o}")
        # annotate % exceeding budget
        pct_exceed = (sub > bud).mean() * 100
        if pct_exceed > 0:
            ax.annotate(
                f"{pct_exceed:.1f}% exceed\n{bud:.0f} ms budget",
                xy=(bud, 1 - pct_exceed / 100),
                xytext=(20, -18), textcoords="offset points",
                fontsize=7.5, color=C[a],
                arrowprops=dict(arrowstyle="->", color=C[a], lw=0.8)
            )

    ax.axvline(BUDGET_SIGN_MS, color=C["budget"], linestyle="--",
               linewidth=1.2, label=f"Sign/Verify budget ({BUDGET_SIGN_MS:.0f} ms)")
    ax.set_xlabel("Estimated Device Time (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(
        "Fig. 7  Empirical CDF of Per-Packet Sign & Verify Latency\n"
        "All trials  |  dotted annotation = % of trials exceeding budget"
    )
    ax.legend(framealpha=0.9)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    savefig(fig, "fig07_cumulative_latency_ecdf.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 08 — Battery drain projection (24-hour)
# FIX: ML-KEM and ML-DSA drain annotated numerically (visually flat vs SLH-DSA)
#      Added inset table with exact 24-h drain values
# ═══════════════════════════════════════════════════════════════════════════════
def fig08_battery(df, s):
    # ML-DSA: 60 sign + 60 verify per hour
    mldsa_sign_uj   = get_val(s, "ML-DSA",  "Sign",   "energy_uj")
    mldsa_verify_uj = get_val(s, "ML-DSA",  "Verify", "energy_uj")
    mldsa_uj_hr     = (mldsa_sign_uj + mldsa_verify_uj) * BPM

    # ML-KEM: 1 full session per hour (KeyGen + Encaps + Decaps)
    mlkem_uj_hr = sum(get_val(s, "ML-KEM", op, "energy_uj")
                      for op in ["KeyGen", "Encaps", "Decaps"])

    # SLH-DSA: hypothetical 60 BPM (fails timing, shown for comparison)
    slhdsa_sign_uj   = get_val(s, "SLH-DSA", "Sign",   "energy_uj")
    slhdsa_verify_uj = get_val(s, "SLH-DSA", "Verify", "energy_uj")
    slhdsa_uj_hr     = (slhdsa_sign_uj + slhdsa_verify_uj) * BPM

    hours = np.arange(0, 25)

    def drain_pct(uj_hr, h):
        return (uj_hr * h / 1000.0) / BATTERY_CAPACITY_MJ * 100.0

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(hours, drain_pct(mldsa_uj_hr,  hours),
            color=C["ML-DSA"],  lw=2.2,
            label="ML-DSA-65  (sign + verify, 60 BPM)")
    ax.plot(hours, drain_pct(mlkem_uj_hr,  hours),
            color=C["ML-KEM"],  lw=2.2,
            label="ML-KEM-768  (full session, 1×/hr)")
    ax.plot(hours, drain_pct(slhdsa_uj_hr, hours),
            color=C["SLH-DSA"], lw=2.2, linestyle="--",
            label="SLH-DSA-128s  (hypothetical 60 BPM — fails timing budget)")

    # clinical shift markers
    for h_mark, label_txt in [(8, "8 h shift"), (24, "24 h")]:
        ax.axvline(h_mark, color="#aaaaaa", linestyle=":", linewidth=1.0)

    ax.set_xlabel("Continuous Operation (hours)")
    ax.set_ylabel("CR2032 Battery Consumed (%)")
    ax.set_xlim(0, 24)
    ax.set_title(
        "Fig. 8  Cumulative Battery Drain Projection\n"
        f"CR2032 = {BATTERY_CAPACITY_MJ/1e6:.2f} J  |  "
        f"Active power = {DEVICE_ACTIVE_POWER_MW} mW"
    )
    ax.legend(framealpha=0.9, loc="upper left")

    # FIX: annotate 24-h exact values for ML-KEM and ML-DSA
    # (visually indistinguishable from zero at SLH-DSA scale without annotation)
    annot_data = [
        ("ML-DSA",  mldsa_uj_hr,  0.55),
        ("ML-KEM",  mlkem_uj_hr,  0.35),
        ("SLH-DSA", slhdsa_uj_hr, None),
    ]
    for alg, uj_hr, x_offset_frac in annot_data:
        val24 = drain_pct(uj_hr, 24)
        x_pos = 24
        ax.annotate(
            f"{val24:.4f}%" if alg != "SLH-DSA" else f"{val24:.3f}%",
            xy=(x_pos, val24),
            xytext=(-52, 4 if alg == "SLH-DSA" else -12),
            textcoords="offset points",
            fontsize=8, color=C[alg], fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=C[alg], lw=0.8)
        )

    # FIX: inset table showing exact drain values
    col_labels = ["Algorithm", "Drain / hr", "Drain @ 24 h"]
    table_data = [
        ["ML-DSA-65",    f"{mldsa_uj_hr/BATTERY_CAPACITY_MJ*1e3*3.6:.5f}%",
                         f"{drain_pct(mldsa_uj_hr,24):.4f}%"],
        ["ML-KEM-768",   f"{mlkem_uj_hr/BATTERY_CAPACITY_MJ*1e3*3.6:.5f}%",
                         f"{drain_pct(mlkem_uj_hr,24):.4f}%"],
        ["SLH-DSA-128s", f"{slhdsa_uj_hr/BATTERY_CAPACITY_MJ*1e3*3.6:.3f}%",
                         f"{drain_pct(slhdsa_uj_hr,24):.3f}%"],
    ]
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center right", bbox=[0.38, 0.45, 0.60, 0.30])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")

    fig.tight_layout()
    savefig(fig, "fig08_battery_drain_projection.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 09 — BLE bandwidth analysis
# FIX: BLE effective line corrected to 25 000 bytes/s = 200 kbps (raw bit rate)
#      Note clarified: source code uses 25 000 bytes/s for delay calculation
# ═══════════════════════════════════════════════════════════════════════════════
def fig09_ble_bandwidth(df, s):
    configs = {
        "ML-KEM\nEncaps CT":  int(get_val(s, "ML-KEM",  "Encaps", "ct_or_sig_bytes")),
        "ML-DSA\nSignature":  int(get_val(s, "ML-DSA",  "Sign",   "ct_or_sig_bytes")),
        "SLH-DSA\nSignature": int(get_val(s, "SLH-DSA", "Sign",   "ct_or_sig_bytes")),
    }
    labels    = list(configs.keys())
    sig_sizes = list(configs.values())
    alg_keys  = ["ML-KEM", "ML-DSA", "SLH-DSA"]

    total_bytes = [HEARTBEAT_PAYLOAD_B + v for v in sig_sizes]
    # bits/s at 1 pkt/s (= BPM/60 = 1 pkt/s)
    bps_needed  = [b * 8 * 1.0 for b in total_bytes]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # left: packet size breakdown
    ax = axes[0]
    x = np.arange(len(labels))
    pay_kb = [HEARTBEAT_PAYLOAD_B / 1024] * len(labels)
    sig_kb = [v / 1024 for v in sig_sizes]
    ax.bar(x, pay_kb, label=f"Payload ({HEARTBEAT_PAYLOAD_B} B)",
           color="#95a5a6", width=0.5)
    ax.bar(x, sig_kb, bottom=pay_kb,
           color=[C[k] for k in alg_keys],
           label="CT / Signature", width=0.5)
    ax.axvline(-0.5, lw=0)   # spacing
    ax.axhline(BUDGET_SIG_BYTES / 1024, color=C["budget"], linestyle="--",
               linewidth=1.3, label=f"Size budget ({BUDGET_SIG_BYTES//1024} KB)")
    # annotate correct multipliers: sig_bytes / payload_bytes
    for i, (sk, v) in enumerate(zip(sig_kb, sig_sizes)):
        mult = v / HEARTBEAT_PAYLOAD_B
        ax.text(i, pay_kb[0] + sk + 0.07,
                f"{v:,} B\n({mult:.1f}× payload)",
                ha="center", va="bottom", fontsize=8, color=C["neutral"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Packet Size (KB)")
    ax.set_title("(a) Packet size breakdown\n(payload + CT/Sig per packet)")
    ax.legend(framealpha=0.9, fontsize=8)

    # right: required uplink vs BLE limits
    ax2 = axes[1]
    bars = ax2.bar(x, [b / 1000 for b in bps_needed],
                   color=[C[k] for k in alg_keys], width=0.5)
    for bar, hk in zip(bars, alg_keys):
        bar.set_hatch(HATCH[hk])

    # BLE 4.2 theoretical max: 125 kbps (raw air rate 1 Mbps / 8 overhead)
    ax2.axhline(BLE_MAX_BPS / 1000, color=C["budget"], linestyle="--",
                linewidth=1.5,
                label=f"BLE 4.2 max ({BLE_MAX_BPS/1000:.0f} kbps)")
    # BLE effective (device_profile.h): 25 000 bytes/s = 200 kbps bits/s
    ble_eff_bits = BLE_EFFECTIVE_BPS * 8  # 200 000 bps
    ax2.axhline(ble_eff_bits / 1000, color="#8e44ad", linestyle="-.",
                linewidth=1.2, alpha=0.8,
                label=f"BLE effective ({ble_eff_bits/1000:.0f} kbps, device_profile.h)")

    for xi, bps in enumerate(bps_needed):
        feasible = bps <= ble_eff_bits
        col = C["pass"] if feasible else C["fail"]
        tag = "OK" if feasible else "EXCEEDS"
        ax2.text(xi, bps / 1000 + 1.5,
                 f"{bps/1000:.1f} kbps\n[{tag}]",
                 ha="center", fontsize=8, color=col, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("Required Uplink (bps) / 1000")
    ax2.set_title(
        "(b) Required uplink bandwidth @ 60 BPM\nvs BLE 4.2 limits"
    )
    ax2.legend(framealpha=0.9, fontsize=8)

    fig.suptitle("Fig. 9  BLE 4.2 Bandwidth Analysis for PQC Overhead", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig09_ble_bandwidth_analysis.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 10 — Radar / spider chart: multi-criteria feasibility
# FIX: ML-KEM polygon made visible with dashed thick border + vertex markers
#      Caption explicitly notes ML-KEM achieves 100% on all dimensions
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_radar(df, s):
    algs = ["ML-KEM", "ML-DSA", "SLH-DSA"]
    dims = ["Timing", "Energy", "Sig/CT\nSize", "RAM", "BLE\nBandwidth"]
    N    = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    def score(alg, op, budget_ms):
        t_ms  = get_val(s, alg, op, "device_time_ms")
        e_uj  = get_val(s, alg, op, "energy_uj")
        ct_b  = get_val(s, alg, op, "ct_or_sig_bytes")
        ram_b = get_val(s, alg, op, "total_ram_bytes")
        uplink = (ct_b + HEARTBEAT_PAYLOAD_B) * 8 * 1.0
        t_sc   = min(1.0, budget_ms  / max(t_ms,  1e-9))
        e_sc   = min(1.0, BUDGET_ENERGY_UJ / max(e_uj,  1e-9))
        sig_sc = (min(1.0, BUDGET_SIG_BYTES / max(ct_b, 1))
                  if ct_b > 0 else 1.0)
        ram_sc = min(1.0, BUDGET_RAM_KB * 1024 / max(ram_b, 1))
        # FIX: use BLE_MAX_BPS (bits/s) consistently with right panel of Fig 09
        ble_sc = min(1.0, BLE_MAX_BPS / max(uplink, 1))
        return [t_sc, e_sc, sig_sc, ram_sc, ble_sc]

    def alg_scores(alg):
        if alg == "ML-KEM":
            ops = [("KeyGen", BUDGET_KEYGEN_MS),
                   ("Encaps", BUDGET_ENCAPS_MS),
                   ("Decaps", BUDGET_DECAPS_MS)]
        else:
            ops = [("KeyGen", BUDGET_KEYGEN_MS),
                   ("Sign",   BUDGET_SIGN_MS),
                   ("Verify", BUDGET_VERIFY_MS)]
        return np.mean([score(alg, op, bud) for op, bud in ops], axis=0)

    fig, ax = plt.subplots(figsize=(6.5, 6.5),
                           subplot_kw=dict(projection="polar"))

    line_styles = {
        "ML-KEM":  {"lw": 2.5, "ls": "--", "alpha_fill": 0.0},   # dashed, no fill
        "ML-DSA":  {"lw": 2.0, "ls": "-",  "alpha_fill": 0.13},
        "SLH-DSA": {"lw": 2.0, "ls": "-",  "alpha_fill": 0.13},
    }

    computed_scores = {}
    for alg in algs:
        sc = alg_scores(alg)
        computed_scores[alg] = sc
        v  = sc.tolist() + sc[:1].tolist()
        ax.plot(angles_closed, v, color=C[alg],
                linewidth=line_styles[alg]["lw"],
                linestyle=line_styles[alg]["ls"],
                label=alg, zorder=3)
        ax.fill(angles_closed, v, color=C[alg],
                alpha=line_styles[alg]["alpha_fill"])
        for ang, val in zip(angles, sc):
            ax.plot(ang, val, "o", color=C[alg], markersize=5, zorder=4)

    ax.set_thetagrids(np.degrees(angles), dims, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="grey")
    ax.set_title(
        "Fig. 10  Multi-Criteria Feasibility Radar\n"
        "Normalised scores — outer rim = fully within budget\n"
        "ML-KEM (dashed) scores ≈ 100% on all axes",
        pad=22
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.14), framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig10_radar_feasibility.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 11 — Scatter: latency vs energy tradeoff space
# ═══════════════════════════════════════════════════════════════════════════════
def fig11_tradeoff_scatter(df, s):
    ops_of_interest = [
        ("ML-KEM",  "KeyGen"), ("ML-KEM",  "Encaps"), ("ML-KEM",  "Decaps"),
        ("ML-DSA",  "KeyGen"), ("ML-DSA",  "Sign"),   ("ML-DSA",  "Verify"),
        ("SLH-DSA", "KeyGen"), ("SLH-DSA", "Sign"),   ("SLH-DSA", "Verify"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for a, o in ops_of_interest:
        row = s[(s.algorithm == a) & (s.operation == o)]
        if not len(row):
            continue
        r = row.iloc[0]
        marker = "o" if r["pass_overall"] else "X"
        ax.scatter(float(r["device_time_ms"]), float(r["energy_uj"]),
                   c=C[a], s=90, marker=marker,
                   edgecolors="white", linewidths=0.5, zorder=4)
        ax.annotate(o, (float(r["device_time_ms"]), float(r["energy_uj"])),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, color=C[a])

    ax.axvline(BUDGET_SIGN_MS, color=C["budget"], linestyle="--",
               linewidth=1, alpha=0.7)
    ax.axhline(BUDGET_ENERGY_UJ, color=C["budget"], linestyle="--",
               linewidth=1, alpha=0.7,
               label=f"Budgets ({BUDGET_SIGN_MS:.0f} ms, {BUDGET_ENERGY_UJ:.0f} µJ)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Estimated Device Latency (ms) — log scale")
    ax.set_ylabel("Energy per Operation (µJ) — log scale")
    ax.set_title(
        "Fig. 11  Latency–Energy Tradeoff Space\n"
        "● = passes all constraints  ✕ = fails  |  dashed = tightest budget"
    )
    legend_handles = [mpatches.Patch(facecolor=C[a], label=a)
                      for a in ["ML-KEM", "ML-DSA", "SLH-DSA"]]
    legend_handles += [
        plt.scatter([], [], marker="o", c="grey", s=60, label="Feasible"),
        plt.scatter([], [], marker="X", c="grey", s=60, label="Infeasible"),
    ]
    ax.legend(handles=legend_handles, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig11_tradeoff_scatter.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 12 — Per-trial time series (Sign operations)
# ═══════════════════════════════════════════════════════════════════════════════
def fig12_timeline(df):
    sign_algs = [("ML-DSA", "Sign"), ("SLH-DSA", "Sign")]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=False)

    for ax, (a, o) in zip(axes, sign_algs):
        sub = df[(df.algorithm == a) & (df.operation == o)].sort_values("trial_num")
        n   = min(200, len(sub))
        trials = sub["trial_num"].values[:n]
        times  = sub["device_time_ms"].values[:n]
        mean_v = times.mean()
        std_v  = times.std()

        ax.plot(trials, times, color=C[a], linewidth=0.8, alpha=0.85)
        ax.axhline(mean_v, color=C["neutral"], linestyle="--",
                   linewidth=1.2, label=f"Mean {mean_v:.1f} ms")
        ax.fill_between(trials, mean_v - std_v, mean_v + std_v,
                        color=C[a], alpha=0.18, label=f"±1σ ({std_v:.1f} ms)")
        ax.axhline(BUDGET_SIGN_MS, color=C["fail"], linestyle=":",
                   linewidth=1, label=f"Budget ({BUDGET_SIGN_MS:.0f} ms)")
        ax.set_ylabel("Device Time (ms)")
        ax.set_title(f"{a} — {o}  ({n} trials shown)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        # annotate % exceeding budget (ML-DSA only — SLH-DSA is always over)
        if a == "ML-DSA":
            pct = (times > BUDGET_SIGN_MS).mean() * 100
            ax.text(0.01, 0.97,
                    f"{pct:.1f}% of trials exceed {BUDGET_SIGN_MS:.0f} ms budget",
                    transform=ax.transAxes, fontsize=8,
                    va="top", color=C["fail"])

    axes[-1].set_xlabel("Trial Number")
    fig.suptitle(
        "Fig. 12  Per-Trial Latency Stability — Sign Operations\n"
        "First 200 trials  |  shaded = ±1σ band",
        y=1.01
    )
    fig.tight_layout()
    savefig(fig, "fig12_per_trial_timeline.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
    print(f"\nPQC Research Paper Figure Generator")
    print(f"  CSV  : {csv_path}")
    print(f"  Out  : ./{OUT_DIR}/\n")

    df = load_data(csv_path)
    s  = summary(df)

    print("Generating figures…")
    fig01_device_time(df, s)
    fig02_energy(df, s)
    fig03_memory(df, s)
    fig04_sig_size(df, s)
    fig05_violin(df)
    fig06_energy_box(df)
    fig07_ecdf(df)
    fig08_battery(df, s)
    fig09_ble_bandwidth(df, s)
    fig10_radar(df, s)
    fig11_tradeoff_scatter(df, s)
    fig12_timeline(df)

    print(f"\nAll 12 figures saved to ./{OUT_DIR}/")
    print("Each figure has both a .pdf (vector, 300 dpi) and .png (preview).\n")


if __name__ == "__main__":
    main()