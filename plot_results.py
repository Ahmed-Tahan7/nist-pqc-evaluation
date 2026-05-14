import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('benchmark_results.csv')

# Filter for the "Main" operations we care about for the heartbeat sensor
# (Session Handshake for KEM, Signing for DSA)
heartbeat_ops = df[df['operation'].isin(['Encaps', 'Sign'])]

# Set the Medical Safety Threshold (Budget) from your device_profile.h
SAFETY_THRESHOLD_MS = 300

# Set Plotting Style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# ---------------------------------------------------------
# 1. Comparative End-to-End Latency Analysis (Bar Chart)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
avg_latency = heartbeat_ops.groupby('algorithm')['total_time_ms'].mean()
colors = ['#2ecc71' if x < SAFETY_THRESHOLD_MS else '#e74c3c' for x in avg_latency]

ax = avg_latency.plot(kind='bar', color=colors, edgecolor='black')
plt.axhline(y=SAFETY_THRESHOLD_MS, color='darkred', linestyle='--', linewidth=2, label='Medical Safety Limit (300ms)')

plt.title('Figure 1: Mean End-to-End Latency (Computation + BLE Delay)')
plt.ylabel('Time (ms)')
plt.xlabel('Algorithm')
plt.legend()
plt.tight_layout()
plt.savefig('latency_comparison.png')

# ---------------------------------------------------------
# 2. Temporal Determinism & Tail Latency (Box Plot)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x='algorithm', y='total_time_ms', data=heartbeat_ops, palette="Set2")
plt.yscale('log') # Log scale because SLH-DSA is so much higher

plt.title('Figure 2: Distribution of Execution Time (1,000 Iterations)')
plt.ylabel('Latency (ms) - Log Scale')
plt.xlabel('Algorithm')
plt.tight_layout()
plt.savefig('jitter_distribution.png')

# ---------------------------------------------------------
# 3. Cryptographic Energy Overhead (Pie Chart)
# ---------------------------------------------------------
plt.figure(figsize=(8, 8))
energy_data = heartbeat_ops.groupby('algorithm')['energy_uj'].mean()
energy_data.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))

plt.title('Figure 3: Relative Energy Consumption per Heartbeat Packet')
plt.ylabel('')
plt.tight_layout()
plt.savefig('energy_overhead.png')

# ---------------------------------------------------------
# 4. Storage Footprint Comparison (Grouped Bar)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
size_metrics = heartbeat_ops.groupby('algorithm')[['pk_bytes', 'ct_or_sig_bytes']].mean()
size_metrics.plot(kind='bar', stacked=False, figsize=(10,6))

plt.title('Figure 4: Communication & Storage Overhead (Bytes)')
plt.ylabel('Size (Bytes)')
plt.xlabel('Algorithm')
plt.legend(['Public Key', 'Ciphertext/Signature'])
plt.tight_layout()
plt.savefig('size_overhead.png')

print("All figures generated successfully.")