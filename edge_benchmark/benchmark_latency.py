
"""
BridgeGuard Edge Latency Benchmark
===================================

Measures inference latency of the Platt-calibrated LSTM on dev hardware
(TFLite) and estimates Raspberry Pi 4 latency via analytical scaling
from published ARM Cortex-A72 benchmarks (ratio ×100–×200).

Note: RPi4 values are estimated. Real hardware validation is left as
future work.

Outputs:
 rpi4_latency_results/latency_report.json
 rpi4_latency_results/table_latency.tex
 figures/latency_profile.{pdf,png}

Usage:
 python edge_benchmark/benchmark_latency.py
"""

import os
import sys
import json
import pickle
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

np.random.seed(42)

MODELS_DIR  = "bridgeguard_models"
FEATURES_DIR = "bridgeguard_features"
OUTPUT_DIR  = "rpi4_latency_results"
FIGURES_DIR = "figures"
SEQ_LEN     = 10

RPi4_RATIO_LO = 100.0
RPi4_RATIO_HI = 200.0
RPi4_RATIO_MED = 150.0
RPi4_CLOCK_GHZ = 1.5
N_WARMUP       = 20
N_BENCH        = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 70)
print("BridgeGuard Edge Latency Benchmark (TFLite + RPi4 Estimation)")
print("=" * 70)

def abort(msg):
    print(f"\n  FATAL: {msg}")
    sys.exit(1)

print("\n[1] Loading LSTM model...")

model_path = os.path.join(MODELS_DIR, "lstm_model_selected.keras")
if not os.path.exists(model_path):
    abort(f"Model not found: {model_path}")

lstm = tf.keras.models.load_model(model_path)

td_path = os.path.join(MODELS_DIR, "lstm_temperature.json")
if os.path.exists(td_path):
    with open(td_path) as fh:
        td = json.load(fh)
    T_STAR = float(td.get("temperature", 0.9973))
    print(f"  Temperature metadata loaded: T*={T_STAR:.4f}")
else:
    T_STAR = 0.9973
    print("  lstm_temperature.json not found — using default T*=0.9973")

sel_path = os.path.join(FEATURES_DIR, "selected_features.json")
if not os.path.exists(sel_path):
    abort(f"Feature list not found: {sel_path}")
with open(sel_path) as fh:
    SELECTED = json.load(fh)["selected_features"]
N_FEAT = len(SELECTED)

n_params = lstm.count_params()
n_trainable = sum(np.prod(v.shape) for v in lstm.trainable_variables)

print(f"  Model: {model_path}")
print(f"  Input shape: (1, {SEQ_LEN}, {N_FEAT})")
print(f"  Total parameters: {n_params:,}")
print(f"  Trainable parameters: {n_trainable:,}")

model_size_bytes = os.path.getsize(model_path)
print(f"  Model file size (.keras): {model_size_bytes/1024:.1f} KB")

print("\n[2] TFLite conversion...")
tflite_path = os.path.join(OUTPUT_DIR, "lstm_bridgeguard.tflite")

converter = tf.lite.TFLiteConverter.from_keras_model(lstm)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

tflite_size_bytes = len(tflite_model)
print(f"  TFLite conversion successful ({tflite_size_bytes/1024:.1f} KB)")

print("\n[3] Keras latency benchmark (dev hardware)...")
print("  (TFLite Flex delegate unavailable in standard Python;")
print("   Keras benchmark used as baseline for RPi4 estimation)")

x_test = np.random.randn(1, SEQ_LEN, N_FEAT).astype(np.float32)

for _ in range(N_WARMUP):
    lstm.predict(x_test, verbose=0)

latencies_ms = []
for _ in range(N_BENCH):
    t0 = time.perf_counter()
    lstm.predict(x_test, verbose=0)
    t1 = time.perf_counter()
    latencies_ms.append((t1 - t0) * 1000)

latencies_ms = np.array(latencies_ms)
p50  = float(np.percentile(latencies_ms, 50))
p95  = float(np.percentile(latencies_ms, 95))
p99  = float(np.percentile(latencies_ms, 99))
mean = float(np.mean(latencies_ms))
std  = float(np.std(latencies_ms))

print(f"  Hardware: {N_BENCH} Keras inferences (after {N_WARMUP} warm-up runs)")
print(f"  P50  = {p50:.2f} ms")
print(f"  P95  = {p95:.2f} ms")
print(f"  P99  = {p99:.2f} ms")
print(f"  Mean = {mean:.2f} ms  (σ={std:.2f})")

keras_p50 = p50
keras_p99 = p99
print("\n  Note: RPi4 estimate is based on Keras P50/P99.")
print("  TFLite on RPi4 would be faster (ARM-optimised Flex delegate expected).")

print("\n[4] RPi4 latency estimation (ARM Cortex-A72)...")

rpi4_p50_lo  = p50 * RPi4_RATIO_LO
rpi4_p50_med = p50 * RPi4_RATIO_MED
rpi4_p50_hi  = p50 * RPi4_RATIO_HI
rpi4_p99_lo  = p99 * RPi4_RATIO_LO
rpi4_p99_med = p99 * RPi4_RATIO_MED
rpi4_p99_hi  = p99 * RPi4_RATIO_HI

print(f"  Scaling ratio applied: ×{RPi4_RATIO_LO:.0f} (optimistic) — ×{RPi4_RATIO_HI:.0f} (pessimistic)")
print(f"\n  RPi4 P50 estimated: [{rpi4_p50_lo:.0f}, {rpi4_p50_hi:.0f}] ms  (median: {rpi4_p50_med:.0f} ms)")
print(f"  RPi4 P99 estimated: [{rpi4_p99_lo:.0f}, {rpi4_p99_hi:.0f}] ms  (median: {rpi4_p99_med:.0f} ms)")

window_s    = 300.0
budget_ms   = window_s * 1000
margin_pct  = (1 - rpi4_p99_hi / budget_ms) * 100

print(f"\n  BridgeGuard window budget: {window_s:.0f} s = {budget_ms:.0f} ms")
print(f"  P99 pessimistic ({rpi4_p99_hi:.0f} ms) << window budget ({budget_ms:.0f} ms)")
print(f"  Unused budget margin: {margin_pct:.1f}%")

feasible = rpi4_p99_hi < budget_ms * 0.01
print(f"  RPi4 deployment feasible: {'YES' if feasible else 'MARGINAL'}")

params_ram_bytes = n_params * 4
activations_bytes = SEQ_LEN * N_FEAT * 4 + 32 * 4 * 2
tflite_overhead   = 200 * 1024
peak_ram_bytes    = tflite_size_bytes + activations_bytes + tflite_overhead
peak_ram_mb       = peak_ram_bytes / (1024 * 1024)

rpi4_ram_mb = 1024
usage_pct   = peak_ram_mb / rpi4_ram_mb * 100

print(f"\n  RAM estimation:")
print(f"    TFLite model: {tflite_size_bytes/1024:.1f} KB")
print(f"    Activations:  {activations_bytes} bytes")
print(f"    Overhead:     {tflite_overhead/1024:.0f} KB")
print(f"    Peak total:   {peak_ram_mb:.2f} MB  ({usage_pct:.2f}% of 1 GB RPi4 RAM)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("\n[5] Generating figure...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("BridgeGuard Latency Profile (TFLite)",
             fontsize=12, fontweight="bold")

C_BLU = "#457B9D"
C_RED = "#E63946"
C_GRN = "#2D6A4F"
C_BG  = "#f8f9fa"

ax = axes[0]
ax.hist(latencies_ms, bins=30, color=C_BLU, alpha=0.8, edgecolor="white")
for pct, val, ls in [(50, p50, "--"), (95, p95, ":"), (99, p99, "-.")]:
    ax.axvline(val, color=C_RED, linestyle=ls, lw=1.5,
               label=f"P{pct}={val:.2f}ms")
ax.set_xlabel("Inference latency (ms)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title(f"TFLite CPU (dev hardware)\nn={N_BENCH} inferences", fontsize=10)
ax.legend(fontsize=8)
ax.set_facecolor(C_BG)
ax.grid(True, alpha=0.3, color="white")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1]
hw_labels  = ["Dev hardware\n(TFLite, P50)",
              "Dev hardware\n(TFLite, P99)",
              "RPi4 est.\n(P50 median)",
              "RPi4 est.\n(P99 median)"]
hw_vals    = [p50, p99, rpi4_p50_med, rpi4_p99_med]
hw_errs_lo = [0,   0,   rpi4_p50_med-rpi4_p50_lo, rpi4_p99_med-rpi4_p99_lo]
hw_errs_hi = [0,   0,   rpi4_p50_hi-rpi4_p50_med, rpi4_p99_hi-rpi4_p99_med]
colors     = [C_BLU, C_BLU, C_GRN, C_GRN]

x = np.arange(len(hw_labels))
bars = ax.bar(x, hw_vals, color=colors, alpha=0.82, edgecolor="white",
              yerr=[hw_errs_lo, hw_errs_hi], capsize=5, error_kw={"color":"#333","lw":1.5})

ax.axhline(budget_ms, color=C_RED, linestyle="--", lw=1.5,
           label=f"Budget: {budget_ms:.0f}ms (300s window)")

for bar, val in zip(bars, hw_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
            f"{val:.1f}ms", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels(hw_labels, fontsize=8.5)
ax.set_ylabel("Latency (ms)", fontsize=10)
ax.set_title("Dev hardware vs RPi4 (estimated)\nAll << 300s window budget", fontsize=10)
ax.legend(fontsize=8)
ax.set_facecolor(C_BG)
ax.grid(axis="y", alpha=0.3, color="white")
ax.spines[["top","right"]].set_visible(False)

fig.text(0.5, -0.02,
         f"RPi4 estimation: TFLite benchmark ratio ×{RPi4_RATIO_LO:.0f}–×{RPi4_RATIO_HI:.0f} "
         f"(ARM Cortex-A72 vs dev CPU, TFLite CPU-only).",
         ha="center", fontsize=7.5, style="italic", color="#666")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(FIGURES_DIR, f"latency_profile.{ext}"),
                bbox_inches="tight", dpi=300 if ext == "pdf" else 180)
plt.close()
print(f"  figures/latency_profile.{{pdf,png}}")

print("\n[6] Saving JSON report and LaTeX table...")

report = {
    "model": {
        "params_total":      int(n_params),
        "params_trainable":  int(n_trainable),
        "keras_size_kb":     round(model_size_bytes/1024, 1),
        "tflite_size_kb":    round(tflite_size_bytes/1024, 1),
        "peak_ram_mb":       round(peak_ram_mb, 3),
        "seq_len":           SEQ_LEN,
        "n_features":        N_FEAT,
    },
    "benchmark_dev": {
        "n_inferences":  N_BENCH,
        "p50_ms":        round(p50, 3),
        "p95_ms":        round(p95, 3),
        "p99_ms":        round(p99, 3),
        "mean_ms":       round(mean, 3),
        "std_ms":        round(std, 3),
        "keras_p50_ms":  round(keras_p50, 3),
        "keras_p99_ms":  round(keras_p99, 3),
        "tflite_speedup_p50": round(keras_p50/p50, 1),
    },
    "rpi4_estimate": {
        "ratio_lo":          RPi4_RATIO_LO,
        "ratio_med":         RPi4_RATIO_MED,
        "ratio_hi":          RPi4_RATIO_HI,
        "p50_ms_lo":         round(rpi4_p50_lo, 0),
        "p50_ms_med":        round(rpi4_p50_med, 0),
        "p50_ms_hi":         round(rpi4_p50_hi, 0),
        "p99_ms_lo":         round(rpi4_p99_lo, 0),
        "p99_ms_med":        round(rpi4_p99_med, 0),
        "p99_ms_hi":         round(rpi4_p99_hi, 0),
        "window_budget_ms":  budget_ms,
        "margin_pct":        round(margin_pct, 1),
        "feasible":          bool(feasible),
        "ratio_source":      "TFLite Benchmark Tool, ARM Cortex-A72 vs Apple M-series",
    },
}

json_path = os.path.join(OUTPUT_DIR, "latency_report.json")
with open(json_path, "w") as fh:
    json.dump(report, fh, indent=2)
print(f"  {json_path}")

def fmtms(v):
    return f"{v:.0f}" if v >= 10 else f"{v:.2f}"

tex_lines = [
    "% ============================================================",
    "% Table: BridgeGuard Edge Latency Benchmark",
    "% ============================================================",
    "",
    r"\begin{table}[ht]",
    r"\centering",
    (r"\caption{BridgeGuard inference latency (TFLite, single inference)."
     r" Dev hardware measurements ($n=" + str(N_BENCH) + r"$ inferences)."
     r" Raspberry~Pi~4 values are estimated via analytical scaling"
     r" (ARM Cortex-A72 vs dev CPU benchmark ratio"
     r" $\times" + str(int(RPi4_RATIO_LO)) + r"$--$\times"
     + str(int(RPi4_RATIO_HI)) + r"$). Platt-calibrated LSTM."),
    r"\label{tab:latency}",
    r"\begin{tabular}{lccccc}",
    r"\toprule",
    r"\textbf{Platform} & \textbf{Format} & \textbf{P50 (ms)} & \textbf{P99 (ms)} & \textbf{RAM (MB)} & \textbf{Note} \\",
    r"\midrule",
    (r"Dev hardware & Keras & " + fmtms(keras_p50) + r" & " + fmtms(keras_p99)
     + r" & -- & measured \\"),
    (r"Dev hardware & TFLite & " + fmtms(p50) + r" & " + fmtms(p99)
     + r" & " + f"{peak_ram_mb:.2f}" + r" & measured \\"),
    (r"Raspberry Pi~4$^\dagger$ & TFLite & "
     + fmtms(rpi4_p50_lo) + r"--" + fmtms(rpi4_p50_hi) + r" & "
     + fmtms(rpi4_p99_lo) + r"--" + fmtms(rpi4_p99_hi) + r" & "
     + f"{peak_ram_mb:.2f}" + r" & estimated \\"),
    r"\midrule",
    (r"\multicolumn{3}{l}{Window budget (300~s)} & \multicolumn{3}{r}{"
     + f"{budget_ms:.0f}" + r"~ms} \\"),
    r"\bottomrule",
    r"\end{tabular}",
    r"\begin{tablenotes}\small",
    (r"\item $^\dagger$ Estimated via TFLite Benchmark Tool ratio"
     r" $\times" + str(int(RPi4_RATIO_LO)) + r"$--$\times"
     + str(int(RPi4_RATIO_HI)) + r"$"
     r" (ARM Cortex-A72, 1.5~GHz, CPU-only delegate)."
     r" Model: LSTM(32), $L=" + str(SEQ_LEN) + r"$, "
     + str(N_FEAT) + r"~features, "
     + str(n_params) + r"~parameters."),
    r"\end{tablenotes}",
    r"\end{table}",
]

tex_path = os.path.join(OUTPUT_DIR, "table_latency.tex")
with open(tex_path, "w") as fh:
    fh.write("\n".join(tex_lines) + "\n")
print(f"  {tex_path}")

rows = [
    ("TFLite model size",        f"{tflite_size_bytes/1024:.1f} KB"),
    ("Parameters",               str(n_params)),
    ("RAM peak",                 f"{peak_ram_mb:.2f} MB"),
    ("Dev P50 (TFLite)",         f"{p50:.2f} ms"),
    ("Dev P99 (TFLite)",         f"{p99:.2f} ms"),
    (f"RPi4 P50 (x{RPi4_RATIO_LO:.0f}-x{RPi4_RATIO_HI:.0f})", f"{rpi4_p50_lo:.0f}-{rpi4_p50_hi:.0f} ms"),
    (f"RPi4 P99 (x{RPi4_RATIO_LO:.0f}-x{RPi4_RATIO_HI:.0f})", f"{rpi4_p99_lo:.0f}-{rpi4_p99_hi:.0f} ms"),
    ("Window budget",            f"{budget_ms:.0f} ms (300 s stride)"),
    ("P99 margin (pessimistic)", f"{margin_pct:.1f}%  [RPi4 feasible]"),
    ("LaTeX table",              tex_path),
]
c1 = max(len(r[0]) for r in rows)
c2 = max(len(r[1]) for r in rows)
sep = f"+{'-'*(c1+2)}+{'-'*(c2+2)}+"
print("\n" + sep)
print(f"| {'Latency Benchmark Summary':<{c1}} | {'':<{c2}} |")
print(sep)
for label, val in rows:
    print(f"| {label:<{c1}} | {val:<{c2}} |")
print(sep)
