"""
PQC Heartbeat IoT Benchmark — 12-Figure Research Paper Script
Nordic nRF52840 (Cortex-M4F @ 64 MHz)
NIST FIPS 203 / 204 / 205  |  ML-KEM-768, ML-DSA-65, SLH-DSA-128s

Run:
    python3 pqc_figures.py [path/to/benchmark_results.csv]

Outputs (in ./figures/):
    fig01_device_time_grouped_bar.pdf
    fig02_energy_grouped_bar.pdf
    fig03_memory_footprint.pdf
    fig04_signature_size_comparison.pdf
    fig05_latency_distribution_violin.pdf
    fig06_energy_distribution_box.pdf
    fig07_cumulative_latency_ecdf.pdf
    fig08_battery_drain_projection.pdf
    fig09_ble_bandwidth_analysis.pdf
    fig10_radar_feasibility.pdf
    fig11_tradeoff_scatter.pdf
    fig12_per_trial_timeline.pdf
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
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,   # embed fonts
    "ps.fonttype":       42,
})

# ── colour palette (colour-blind-safe) ──────────────────────────────────────
C = {
    "ML-KEM":  "#2166ac",   # blue
    "ML-DSA":  "#d6604d",   # red-orange
    "SLH-DSA": "#4dac26",   # green
    "budget":  "#888888",
    "pass":    "#27ae60",
    "fail":    "#c0392b",
    "neutral": "#555555",
}

HATCH = {"ML-KEM": "", "ML-DSA": "//", "SLH-DSA": "xx"}

# ── device / scenario constants (must match device_profile.h) ────────────────
DEVICE_CPU_MHZ         = 64
DEVICE_ACTIVE_POWER_MW = 14.4
BUDGET_KEYGEN_MS       = 2000.0
BUDGET_SIGN_MS         = 300.0
BUDGET_VERIFY_MS       = 300.0
BUDGET_ENCAPS_MS       = 500.0
BUDGET_DECAPS_MS       = 500.0
BUDGET_ENERGY_UJ       = 3500.0
BUDGET_SIG_BYTES       = 4096
BATTERY_CAPACITY_MJ    = 2_538_000.0
BLE_EFFECTIVE_BPS      = 25_000.0          # bytes/s
BLE_MAX_BPS            = 125_000.0         # bits/s
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
    """Return mean stats per (algorithm, operation)."""
    grp = df.groupby(["algorithm", "operation"])
    s = grp.agg(
        device_time_ms   = ("device_time_ms",   "mean"),
        energy_uj        = ("energy_uj",         "mean"),
        ble_delay_ms     = ("ble_delay_ms",      "mean"),
        total_time_ms    = ("total_time_ms",     "mean"),
        pk_bytes         = ("pk_bytes",          "first"),
        sk_bytes         = ("sk_bytes",          "first"),
        ct_or_sig_bytes  = ("ct_or_sig_bytes",   "first"),
        total_ram_bytes  = ("total_ram_bytes",   "first"),
        pass_timing      = ("pass_timing",       "first"),
        pass_memory      = ("pass_memory",       "first"),
        pass_energy      = ("pass_energy",       "first"),
        pass_overall     = ("pass_overall",      "first"),
    ).reset_index()
    return s


# ── operation ordering helpers ────────────────────────────────────────────────
KEM_OPS = ["KeyGen", "Encaps", "Decaps"]
SIG_OPS = ["KeyGen", "Sign",   "Verify"]
ALL_OPS_ORDER = {"KeyGen": 0, "Encaps": 1, "Decaps": 1, "Sign": 2, "Verify": 3}


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 01 — Grouped bar: device time per operation (log scale)
# ═══════════════════════════════════════════════════════════════════════════════
def fig01_device_time(df, s):
    algs  = ["ML-KEM", "ML-DSA", "SLH-DSA"]
    ops   = [("ML-KEM","KeyGen"), ("ML-KEM","Encaps"), ("ML-KEM","Decaps"),
             ("ML-DSA","KeyGen"), ("ML-DSA","Sign"),   ("ML-DSA","Verify"),
             ("SLH-DSA","KeyGen"),("SLH-DSA","Sign"),  ("SLH-DSA","Verify")]
    labels = [f"{a}\n{o}" for a,o in ops]
    budgets= [BUDGET_KEYGEN_MS, BUDGET_ENCAPS_MS, BUDGET_DECAPS_MS,
              BUDGET_KEYGEN_MS, BUDGET_SIGN_MS,   BUDGET_VERIFY_MS,
              BUDGET_KEYGEN_MS, BUDGET_SIGN_MS,   BUDGET_VERIFY_MS]

    vals, colours, hatches = [], [], []
    for a, o in ops:
        row = s[(s.algorithm==a) & (s.operation==o)]
        vals.append(float(row.device_time_ms.iloc[0]) if len(row) else 0)
        colours.append(C[a])
        hatches.append(HATCH[a])

    x    = np.arange(len(ops))
    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars = ax.bar(x, vals, color=colours, edgecolor="white",
                  linewidth=0.6, width=0.6)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    # budget markers
    for xi, bud in enumerate(budgets):
        ax.hlines(bud, xi-0.35, xi+0.35, colors=C["budget"],
                  linewidths=1.2, linestyles="--", zorder=5)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Estimated Device Time (ms) — log scale")
    ax.set_title("Fig. 1  Estimated Device Execution Time per Operation\n"
                 "Nordic nRF52840 Cortex-M4F @ 64 MHz  |  dashed line = timing budget")
    # legend
    legend_handles = [mpatches.Patch(facecolor=C[a], label=a, hatch=HATCH[a])
                      for a in algs]
    legend_handles.append(mpatches.Patch(facecolor=C["budget"], label="Budget"))
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)

    # annotate SLH-DSA sign value
    slh_sign_idx = labels.index("SLH-DSA\nSign")
    ax.annotate(f"{vals[slh_sign_idx]:,.0f} ms",
                xy=(x[slh_sign_idx], vals[slh_sign_idx]),
                xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=7, color=C["fail"], fontweight="bold")

    fig.tight_layout()
    savefig(fig, "fig01_device_time_grouped_bar.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 02 — Grouped bar: energy per operation (log scale)
# ═══════════════════════════════════════════════════════════════════════════════
def fig02_energy(df, s):
    ops = [("ML-KEM","KeyGen"), ("ML-KEM","Encaps"), ("ML-KEM","Decaps"),
           ("ML-DSA","KeyGen"), ("ML-DSA","Sign"),   ("ML-DSA","Verify"),
           ("SLH-DSA","KeyGen"),("SLH-DSA","Sign"),  ("SLH-DSA","Verify")]
    labels = [f"{a}\n{o}" for a,o in ops]
    vals, colours = [], []
    for a, o in ops:
        row = s[(s.algorithm==a) & (s.operation==o)]
        vals.append(float(row.energy_uj.iloc[0]) if len(row) else 0)
        colours.append(C[a])

    x   = np.arange(len(ops))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(x, vals, color=colours, edgecolor="white", linewidth=0.6, width=0.6)
    for bar, (a, _) in zip(bars, ops):
        bar.set_hatch(HATCH[a])

    ax.axhline(BUDGET_ENERGY_UJ, color=C["budget"], linestyle="--",
               linewidth=1.2, label=f"Energy budget ({BUDGET_ENERGY_UJ:.0f} µJ)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Energy per Operation (µJ) — log scale")
    ax.set_title("Fig. 2  Energy Consumption per Cryptographic Operation\n"
                 "CR2032 @ 3 V · 14.4 mW active power model")
    legend_handles = [mpatches.Patch(facecolor=C[a], label=a, hatch=HATCH[a])
                      for a in ["ML-KEM","ML-DSA","SLH-DSA"]]
    legend_handles.append(mpatches.Patch(facecolor=C["budget"], label=f"Budget {BUDGET_ENERGY_UJ:.0f} µJ"))
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig02_energy_grouped_bar.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 03 — Stacked bar: memory footprint (PK + SK + CT/Sig)
# ═══════════════════════════════════════════════════════════════════════════════
def fig03_memory(df, s):
    rows = [
        ("ML-KEM",  "KeyGen"), ("ML-KEM",  "Encaps"), ("ML-KEM",  "Decaps"),
        ("ML-DSA",  "KeyGen"), ("ML-DSA",  "Sign"),   ("ML-DSA",  "Verify"),
        ("SLH-DSA", "KeyGen"), ("SLH-DSA", "Sign"),   ("SLH-DSA", "Verify"),
    ]
    labels, pk_v, sk_v, ct_v = [], [], [], []
    for a, o in rows:
        row = s[(s.algorithm==a) & (s.operation==o)]
        if not len(row): continue
        labels.append(f"{a}\n{o}")
        pk_v.append(int(row.pk_bytes.iloc[0]) / 1024)
        sk_v.append(int(row.sk_bytes.iloc[0]) / 1024)
        ct_v.append(int(row.ct_or_sig_bytes.iloc[0]) / 1024)

    x   = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x, pk_v, label="Public Key",    color="#4393c3", width=0.55)
    b2 = ax.bar(x, sk_v, bottom=pk_v,           label="Secret Key",   color="#d6604d", width=0.55)
    bot3 = [p+s for p,s in zip(pk_v, sk_v)]
    b3 = ax.bar(x, ct_v, bottom=bot3,           label="CT / Signature",color="#4dac26", width=0.55)

    ax.axhline(BUDGET_SIG_BYTES/1024, color=C["budget"], linestyle="--",
               linewidth=1.2, label=f"Sig/CT budget ({BUDGET_SIG_BYTES//1024} KB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Size (KB)")
    ax.set_title("Fig. 3  Cryptographic Object Memory Footprint\n"
                 "Public Key + Secret Key + Ciphertext/Signature")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig03_memory_footprint.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 04 — Signature / Ciphertext sizes vs payload
# ═══════════════════════════════════════════════════════════════════════════════
def fig04_sig_size(df, s):
    data = {}
    for a, o in [("ML-KEM","Encaps"),("ML-KEM","Decaps"),
                 ("ML-DSA","Sign"),("ML-DSA","Verify"),
                 ("SLH-DSA","Sign"),("SLH-DSA","Verify")]:
        row = s[(s.algorithm==a)&(s.operation==o)]
        if len(row):
            key = f"{a}\n{o}"
            data[key] = (int(row.ct_or_sig_bytes.iloc[0]), C[a], HATCH[a])

    labels = list(data.keys())
    sizes  = [data[k][0] for k in labels]
    colours= [data[k][1] for k in labels]
    hatches= [data[k][2] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, sizes, color=colours, edgecolor="white", width=0.55)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.axhline(HEARTBEAT_PAYLOAD_B, color="#e67e22", linestyle="-.",
               linewidth=1.5, label=f"Payload ({HEARTBEAT_PAYLOAD_B} B)")
    ax.axhline(BUDGET_SIG_BYTES, color=C["budget"], linestyle="--",
               linewidth=1.2, label=f"Budget ({BUDGET_SIG_BYTES} B)")

    for xi, v in zip(x, sizes):
        ax.text(xi, v + 30, f"{v:,}", ha="center", fontsize=8, color=C["neutral"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Bytes")
    ax.set_title("Fig. 4  Ciphertext / Signature Size vs Heartbeat Payload\n"
                 f"Payload = {HEARTBEAT_PAYLOAD_B} B  |  BLE budget = {BUDGET_SIG_BYTES} B")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig04_signature_size_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 05 — Violin plots: latency distribution (device_time_ms per trial)
# ═══════════════════════════════════════════════════════════════════════════════
def fig05_violin(df):
    combos = [
        ("ML-KEM","KeyGen"),("ML-KEM","Encaps"),("ML-KEM","Decaps"),
        ("ML-DSA","KeyGen"),("ML-DSA","Sign"),  ("ML-DSA","Verify"),
    ]  # exclude SLH-DSA (too few trials / extreme scale)
    groups  = []
    labels  = []
    colours = []
    for a, o in combos:
        sub = df[(df.algorithm==a)&(df.operation==o)]["device_time_ms"]
        if len(sub):
            groups.append(sub.values)
            labels.append(f"{a}\n{o}")
            colours.append(C[a])

    fig, ax = plt.subplots(figsize=(9, 4.5))
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
    ax.set_title("Fig. 5  Latency Distribution per Operation (ML-KEM & ML-DSA)\n"
                 "1 000 trials each  |  white line = median  |  SLH-DSA excluded (extreme scale)")
    legend_handles = [mpatches.Patch(facecolor=C[a], label=a)
                      for a in ["ML-KEM","ML-DSA"]]
    ax.legend(handles=legend_handles, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig05_latency_distribution_violin.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 06 — Box plots: energy distribution
# ═══════════════════════════════════════════════════════════════════════════════
def fig06_energy_box(df):
    combos = [
        ("ML-KEM","KeyGen"),("ML-KEM","Encaps"),("ML-KEM","Decaps"),
        ("ML-DSA","KeyGen"),("ML-DSA","Sign"),  ("ML-DSA","Verify"),
        ("SLH-DSA","Verify"),
    ]
    groups, labels, colours = [], [], []
    for a, o in combos:
        sub = df[(df.algorithm==a)&(df.operation==o)]["energy_uj"]
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
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Energy per Operation (µJ)")
    ax.set_title("Fig. 6  Energy Distribution per Operation (Box-and-Whisker)\n"
                 "SLH-DSA Sign excluded (883 885 µJ median — off scale)")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig06_energy_distribution_box.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 07 — ECDF: cumulative latency for Sign / Verify candidates
# ═══════════════════════════════════════════════════════════════════════════════
def fig07_ecdf(df):
    targets = [
        ("ML-DSA",  "Sign",   BUDGET_SIGN_MS),
        ("ML-DSA",  "Verify", BUDGET_VERIFY_MS),
        ("SLH-DSA", "Verify", BUDGET_VERIFY_MS),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    for a, o, bud in targets:
        sub = df[(df.algorithm==a)&(df.operation==o)]["device_time_ms"].sort_values()
        if not len(sub): continue
        ecdf = np.arange(1, len(sub)+1) / len(sub)
        label = f"{a} {o}"
        ax.step(sub, ecdf, where="post", color=C[a],
                linestyle="-" if o=="Sign" else "--", linewidth=1.6, label=label)
        ax.axvline(bud, color=C[a], linestyle=":", linewidth=1, alpha=0.6)

    ax.axvline(BUDGET_SIGN_MS, color=C["budget"], linestyle="--",
               linewidth=1.2, label=f"Sign/Verify budget ({BUDGET_SIGN_MS:.0f} ms)")
    ax.set_xlabel("Estimated Device Time (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Fig. 7  Empirical CDF of Per-Packet Sign & Verify Latency\n"
                 "All trials  |  dotted verticals = timing budget")
    ax.legend(framealpha=0.9)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    savefig(fig, "fig07_cumulative_latency_ecdf.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 08 — Battery drain projection (hourly / daily / annual)
# ═══════════════════════════════════════════════════════════════════════════════
def fig08_battery(df, s):
    # ML-DSA: 60 sign + 60 verify per hour
    mldsa_sign   = float(s[(s.algorithm=="ML-DSA")&(s.operation=="Sign")].energy_uj.iloc[0])
    mldsa_verify = float(s[(s.algorithm=="ML-DSA")&(s.operation=="Verify")].energy_uj.iloc[0])
    mldsa_uj_hr  = (mldsa_sign + mldsa_verify) * BPM

    # ML-KEM: 1 full session per hour
    mlkem_kg  = float(s[(s.algorithm=="ML-KEM")&(s.operation=="KeyGen")].energy_uj.iloc[0])
    mlkem_enc = float(s[(s.algorithm=="ML-KEM")&(s.operation=="Encaps")].energy_uj.iloc[0])
    mlkem_dec = float(s[(s.algorithm=="ML-KEM")&(s.operation=="Decaps")].energy_uj.iloc[0])
    mlkem_uj_hr = mlkem_kg + mlkem_enc + mlkem_dec

    slhdsa_sign  = float(s[(s.algorithm=="SLH-DSA")&(s.operation=="Sign")].energy_uj.iloc[0])
    slhdsa_verify= float(s[(s.algorithm=="SLH-DSA")&(s.operation=="Verify")].energy_uj.iloc[0])
    slhdsa_uj_hr = (slhdsa_sign + slhdsa_verify) * BPM

    hours = np.arange(0, 25)

    def pct(uj_hr, h):
        return (uj_hr * h / 1000.0) / BATTERY_CAPACITY_MJ * 100.0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hours, pct(mldsa_uj_hr,  hours), color=C["ML-DSA"],  lw=2, label="ML-DSA-65 (60 BPM sign+verify)")
    ax.plot(hours, pct(mlkem_uj_hr,  hours), color=C["ML-KEM"],  lw=2, label="ML-KEM-768 (1 session/hr)")
    ax.plot(hours, pct(slhdsa_uj_hr, hours), color=C["SLH-DSA"], lw=2, label="SLH-DSA-128s (60 BPM sign+verify)",
            linestyle="--")
    ax.set_xlabel("Operation Duration (hours)")
    ax.set_ylabel("CR2032 Battery Consumed (%)")
    ax.set_title("Fig. 8  Cumulative Battery Drain Projection\n"
                 f"CR2032 = {BATTERY_CAPACITY_MJ/1e6:.2f} J  |  Active power = {DEVICE_ACTIVE_POWER_MW} mW")
    ax.legend(framealpha=0.9)
    ax.set_xlim(0, 24)
    fig.tight_layout()
    savefig(fig, "fig08_battery_drain_projection.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 09 — BLE bandwidth analysis
# ═══════════════════════════════════════════════════════════════════════════════
def fig09_ble_bandwidth(df, s):
    configs = {
        "ML-KEM\nEncaps CT":  int(s[(s.algorithm=="ML-KEM")&(s.operation=="Encaps")].ct_or_sig_bytes.iloc[0]),
        "ML-DSA\nSignature":  int(s[(s.algorithm=="ML-DSA")&(s.operation=="Sign")].ct_or_sig_bytes.iloc[0]),
        "SLH-DSA\nSignature": int(s[(s.algorithm=="SLH-DSA")&(s.operation=="Sign")].ct_or_sig_bytes.iloc[0]),
    }
    labels      = list(configs.keys())
    sig_sizes   = list(configs.values())
    alg_keys    = ["ML-KEM", "ML-DSA", "SLH-DSA"]

    # total bytes/packet = payload + sig/ct
    total_bytes = [HEARTBEAT_PAYLOAD_B + v for v in sig_sizes]
    bps_needed  = [b * 8 * BPM / 60 for b in total_bytes]   # bits/s at 1 pkt/s

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # left: packet size breakdown
    ax = axes[0]
    x  = np.arange(len(labels))
    pay = [HEARTBEAT_PAYLOAD_B / 1024] * len(labels)
    sig_kb = [v / 1024 for v in sig_sizes]
    ax.bar(x, pay, label="Payload", color="#95a5a6", width=0.5)
    ax.bar(x, sig_kb, bottom=pay, color=[C[k] for k in alg_keys],
           label="CT / Signature", width=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Packet Size (KB)")
    ax.set_title("(a) Packet size breakdown\n(payload + CT/Sig per packet)")
    ax.legend(framealpha=0.9)

    # right: uplink BPS vs BLE limits
    ax2 = axes[1]
    bars = ax2.bar(x, bps_needed, color=[C[k] for k in alg_keys], width=0.5)
    ax2.axhline(BLE_MAX_BPS, color=C["budget"], linestyle="--", lw=1.5,
                label=f"BLE 4.2 max ({BLE_MAX_BPS/1000:.0f} kbps)")
    ax2.axhline(25_000*8, color="#e67e22", linestyle="-.", lw=1.2,
                label="BLE effective (~200 kbps)")
    for bar, bps in zip(bars, bps_needed):
        ax2.text(bar.get_x()+bar.get_width()/2, bps+500,
                 f"{bps/1000:.1f} kbps", ha="center", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("Required Uplink (bps)")
    ax2.set_title("(b) Required uplink bandwidth @ 60 BPM\nvs BLE 4.2 limits")
    ax2.legend(framealpha=0.9)

    fig.suptitle("Fig. 9  BLE 4.2 Bandwidth Analysis for PQC Overhead", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig09_ble_bandwidth_analysis.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 10 — Radar / spider chart: multi-criteria feasibility
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_radar(df, s):
    """
    5 normalised dimensions (higher = better / more feasible):
      1. Timing score  = min(1, budget / device_ms)
      2. Energy score  = min(1, budget / energy_uj)
      3. Sig/CT score  = min(1, budget_bytes / ct_bytes)  [for sign ops]
      4. RAM score     = min(1, 128*1024 / total_ram)
      5. BLE score     = 1 if uplink < BLE max else BLE_max/uplink
    We show KeyGen + Sign/Encaps averaged per algorithm.
    """
    algs = ["ML-KEM", "ML-DSA", "SLH-DSA"]
    dims = ["Timing", "Energy", "Sig/CT\nSize", "RAM", "BLE\nBandwidth"]
    N    = len(dims)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw=dict(projection="polar"))

    def score(a, op, budget_ms):
        row = s[(s.algorithm==a)&(s.operation==op)]
        if not len(row): return [0]*5
        r = row.iloc[0]
        t_score  = min(1.0, budget_ms / max(float(r["device_time_ms"]), 1e-9))
        e_score  = min(1.0, BUDGET_ENERGY_UJ / max(float(r["energy_uj"]), 1e-9))
        ct_bytes = float(r["ct_or_sig_bytes"])
        sig_score= min(1.0, BUDGET_SIG_BYTES / max(ct_bytes, 1)) \
                   if ct_bytes > 0 else 1.0
        ram_score= min(1.0, 128*1024 / max(float(r["total_ram_bytes"]), 1))
        uplink   = (ct_bytes + HEARTBEAT_PAYLOAD_B) * 8 * 1.0
        ble_score= min(1.0, BLE_MAX_BPS / max(uplink, 1))
        return [t_score, e_score, sig_score, ram_score, ble_score]

    for alg in algs:
        # combine primary op per algorithm
        if alg == "ML-KEM":
            sc = np.mean([score(alg,"KeyGen",BUDGET_KEYGEN_MS),
                          score(alg,"Encaps",BUDGET_ENCAPS_MS),
                          score(alg,"Decaps",BUDGET_DECAPS_MS)], axis=0)
        else:
            sc = np.mean([score(alg,"KeyGen",BUDGET_KEYGEN_MS),
                          score(alg,"Sign",BUDGET_SIGN_MS),
                          score(alg,"Verify",BUDGET_VERIFY_MS)], axis=0)
        values = sc.tolist() + sc[:1].tolist()
        ax.plot(angles, values, color=C[alg], linewidth=2, label=alg)
        ax.fill(angles, values, color=C[alg], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%","50%","75%","100%"], fontsize=7, color="grey")
    ax.set_title("Fig. 10  Multi-Criteria Feasibility Radar\n"
                 "Normalised scores — outer rim = fully within budget",
                 pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "fig10_radar_feasibility.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 11 — Scatter: latency vs energy (tradeoff space)
# ═══════════════════════════════════════════════════════════════════════════════
def fig11_tradeoff_scatter(df, s):
    ops_of_interest = [
        ("ML-KEM",  "KeyGen"), ("ML-KEM",  "Encaps"), ("ML-KEM",  "Decaps"),
        ("ML-DSA",  "KeyGen"), ("ML-DSA",  "Sign"),   ("ML-DSA",  "Verify"),
        ("SLH-DSA", "KeyGen"), ("SLH-DSA", "Sign"),   ("SLH-DSA", "Verify"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for a, o in ops_of_interest:
        row = s[(s.algorithm==a)&(s.operation==o)]
        if not len(row): continue
        r = row.iloc[0]
        pass_marker = "o" if r["pass_overall"] else "X"
        ax.scatter(float(r["device_time_ms"]), float(r["energy_uj"]),
                   c=C[a], s=90, marker=pass_marker,
                   edgecolors="white", linewidths=0.5, zorder=4)
        ax.annotate(o, (float(r["device_time_ms"]), float(r["energy_uj"])),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, color=C[a])

    # budget box
    ax.axvline(BUDGET_SIGN_MS, color=C["budget"], linestyle="--",
               linewidth=1, alpha=0.7)
    ax.axhline(BUDGET_ENERGY_UJ, color=C["budget"], linestyle="--",
               linewidth=1, alpha=0.7, label=f"Budgets (300 ms, {BUDGET_ENERGY_UJ:.0f} µJ)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Estimated Device Latency (ms) — log scale")
    ax.set_ylabel("Energy per Operation (µJ) — log scale")
    ax.set_title("Fig. 11  Latency–Energy Tradeoff Space\n"
                 "● = passes all constraints  ✕ = fails  |  dashed = tightest budget")

    legend_handles = [mpatches.Patch(facecolor=C[a], label=a)
                      for a in ["ML-KEM","ML-DSA","SLH-DSA"]]
    legend_handles += [
        plt.scatter([],[],marker="o",c="grey",s=60,label="Feasible"),
        plt.scatter([],[],marker="X",c="grey",s=60,label="Infeasible"),
    ]
    ax.legend(handles=legend_handles, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig11_tradeoff_scatter.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 12 — Per-trial time series (first 200 trials of Sign operations)
# ═══════════════════════════════════════════════════════════════════════════════
def fig12_timeline(df):
    sign_algs = [("ML-DSA","Sign"), ("SLH-DSA","Sign")]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)

    for ax, (a, o) in zip(axes, sign_algs):
        sub = df[(df.algorithm==a)&(df.operation==o)].sort_values("trial_num")
        n = min(200, len(sub))
        trials = sub["trial_num"].values[:n]
        times  = sub["device_time_ms"].values[:n]
        mean_v = times.mean()
        std_v  = times.std()

        ax.plot(trials, times, color=C[a], linewidth=0.8, alpha=0.85)
        ax.axhline(mean_v, color=C["neutral"], linestyle="--",
                   linewidth=1.2, label=f"Mean {mean_v:.1f} ms")
        ax.fill_between(trials, mean_v - std_v, mean_v + std_v,
                        color=C[a], alpha=0.18, label=f"±1σ ({std_v:.1f} ms)")
        bud = BUDGET_SIGN_MS
        ax.axhline(bud, color=C["fail"], linestyle=":", linewidth=1,
                   label=f"Budget ({bud:.0f} ms)")
        ax.set_ylabel("Device Time (ms)")
        ax.set_title(f"{a} — {o}  ({n} trials shown)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel("Trial Number")
    fig.suptitle("Fig. 12  Per-Trial Latency Stability — Sign Operations\n"
                 "First 200 trials  |  shaded = ±1σ band", y=1.01)
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
    print("Each figure has both a .pdf (vector, 300 dpi) and .png (preview) version.\n")


if __name__ == "__main__":
    main()