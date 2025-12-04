# src/plot_all_results.py (IGNORE FOR NOW <<EXPERIMENTAL>>)
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# ----- config-----
EPOCHS = 10                 # amortize sharding over this many epochs
BREAKDOWN_BATCH = 512       # batch size for stacked time breakdown
SHARD_JSON = "results/shard_prep_metrics.json"
# --------------------------------

BASE = os.path.dirname(os.path.dirname(__file__))  # project root
res_dir = os.path.join(BASE, "results")

# Load CSVs
df_local = pd.read_csv(os.path.join(res_dir, "part1_locality_results.csv"))
df_labl  = pd.read_csv(os.path.join(res_dir, "part1_labl_results.csv"))

# Combine and save
df_all = pd.concat([df_local, df_labl], ignore_index=True)
df_all.to_csv(os.path.join(res_dir, "part1_all_results.csv"), index=False)

# Try to load sharding metrics JSON
json_path = os.path.join(BASE, SHARD_JSON) if not os.path.isabs(SHARD_JSON) else SHARD_JSON
if not os.path.exists(json_path):
    # also try results/ relative to project root
    alt = os.path.join(res_dir, os.path.basename(SHARD_JSON))
    json_path = alt if os.path.exists(alt) else None

shard_meta = None
if json_path:
    with open(json_path, "r") as f:
        shard_meta = json.load(f)
        # expected keys from your file:
        # 'total_windows', 'window_len', 'shard_size_windows', 'num_shards',
        # 'load_time_s', 'write_time_s', 'total_time_s', ...
else:
    print("[warn] No shard JSON found; skipping effective A4 curve.")

# -------- Plot 1: Throughput vs batch for all configs (+ A4 effective) --------
plt.figure(figsize=(6,4))
for name, grp in df_all.groupby("config"):
    grp = grp.sort_values("batch_size")
    plt.plot(grp["batch_size"], grp["samples_per_s"], marker="o", label=name)

# Add A4 effective throughput curve if JSON is present
if shard_meta is not None:
    N_total = int(shard_meta.get("total_windows", 0))
    shard_time_s = float(shard_meta.get("total_time_s", 0.0))
    a4 = df_all[df_all["config"] == "A4_LABL"].copy()
    if not a4.empty and N_total > 0 and shard_time_s > 0.0:
        # Effective throughput per row:
        # epoch_time_s = N_total / samples_per_s
        # eff_samples_per_s = samples_per_s / (1 + shard_time_s / (EPOCHS * epoch_time_s))
        epoch_time_s = N_total / a4["samples_per_s"]
        factor = 1.0 / (1.0 + shard_time_s / (EPOCHS * epoch_time_s))
        a4["samples_per_s_eff"] = a4["samples_per_s"] * factor
        a4 = a4.sort_values("batch_size")
        plt.plot(a4["batch_size"], a4["samples_per_s_eff"],
                 marker="o", linestyle="--",
                 label=f"A4_LABL (eff, E={EPOCHS})")
    else:
        print("[warn] A4 effective curve skipped: missing/invalid totals in shard JSON.")

plt.xlabel("Batch size"); plt.ylabel("Samples / second")
plt.title("Throughput Comparison (A0–A4)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(res_dir, "throughput_comparison_A0_A4.png"), dpi=300)
plt.close()

# -------- Plot 2: Time breakdown bars at a chosen batch (A0–A4) --------
BATCH = BREAKDOWN_BATCH
df_b = df_all[df_all["batch_size"] == BATCH].copy()

# Compute amortized shard ms/step for A4 and add as a new column 'shard_ms'
df_b["shard_ms"] = 0.0
if shard_meta is not None:
    N_total = int(shard_meta.get("total_windows", 0))
    shard_time_s = float(shard_meta.get("total_time_s", 0.0))
    if N_total > 0 and shard_time_s > 0.0:
        # steps_per_epoch = N_total / batch_size
        # shard_ms_per_step = (shard_time_s / EPOCHS) / steps_per_epoch * 1e3
        mask_a4 = (df_b["config"] == "A4_LABL")
        if mask_a4.any():
            bs = df_b.loc[mask_a4, "batch_size"].astype(float)
            steps_per_epoch = N_total / bs
            shard_ms_per_step = (shard_time_s / EPOCHS) / steps_per_epoch * 1e3
            # assign (align index)
            df_b.loc[mask_a4, "shard_ms"] = shard_ms_per_step.values

# Keep a stable order (A0..A4)
order = ["A0_baseline", "A1_contiguous", "A2_contig_pinned",
         "A3_contig_pinned_nb", "A4_LABL"]
df_b["config"] = pd.Categorical(df_b["config"], categories=order, ordered=True)
df_b = df_b.sort_values("config")

plt.figure(figsize=(6.6,4.2))
bar_w = 0.6
x = range(len(df_b))

# Base stacked bars
plt.bar(x, df_b["data_ms"], width=bar_w, label="data_ms")
plt.bar(x, df_b["h2d_ms"], width=bar_w, bottom=df_b["data_ms"], label="h2d_ms")
plt.bar(x, df_b["compute_ms"], width=bar_w,
        bottom=df_b["data_ms"] + df_b["h2d_ms"], label="compute_ms")

# Add shard_ms on top (only A4 has non-zero)
if (df_b["shard_ms"] > 0).any():
    plt.bar(
        x, df_b["shard_ms"], width=bar_w,
        bottom=df_b["data_ms"] + df_b["h2d_ms"] + df_b["compute_ms"],
        label=f"shard_ms/step (E={EPOCHS})", alpha=0.7
    )

plt.xticks(x, df_b["config"], rotation=20)
plt.ylabel("Milliseconds per step")
plt.title(f"Time Breakdown per Step (batch={BATCH})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(res_dir, f"time_breakdown_batch{BATCH}_A0_A4.png"), dpi=300)
plt.close()

print("[OK] Wrote:",
      os.path.join(res_dir, "part1_all_results.csv"), "\n",
      os.path.join(res_dir, "throughput_comparison_A0_A4.png"), "\n",
      os.path.join(res_dir, f"time_breakdown_batch{BATCH}_A0_A4.png"))
if shard_meta is None:
    print("(Note: shard JSON not found; A4 effective line and shard_ms bars omitted.)")


'''import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(__file__))  # project root
res_dir = os.path.join(BASE, "results")

df_local = pd.read_csv(os.path.join(res_dir, "part1_locality_results.csv"))
df_labl  = pd.read_csv(os.path.join(res_dir, "part1_labl_results.csv"))

df_all = pd.concat([df_local, df_labl], ignore_index=True)
df_all.to_csv(os.path.join(res_dir, "part1_all_results.csv"), index=False)

# -------- Plot 1: Throughput vs batch for all configs --------
plt.figure(figsize=(6,4))
for name, grp in df_all.groupby("config"):
    grp = grp.sort_values("batch_size")
    plt.plot(grp["batch_size"], grp["samples_per_s"], marker="o", label=name)
plt.xlabel("Batch size"); plt.ylabel("Samples / second")
plt.title("Throughput Comparison (A0–A4)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(res_dir, "throughput_comparison_A0_A4.png"), dpi=300)
plt.close()

# -------- Plot 2: Time breakdown bars at a chosen batch (e.g., 512) --------
BATCH = 512  # change if you prefer 256
df_b = df_all[df_all["batch_size"] == BATCH].copy()
# Keep a stable order (A0..A4)
order = ["A0_baseline", "A1_contiguous", "A2_contig_pinned",
         "A3_contig_pinned_nb", "A4_LABL"]
df_b["config"] = pd.Categorical(df_b["config"], categories=order, ordered=True)
df_b = df_b.sort_values("config")

plt.figure(figsize=(6.3,4.2))
bar_w = 0.55
x = range(len(df_b))
plt.bar(x, df_b["data_ms"], width=bar_w, label="data_ms")
plt.bar(x, df_b["h2d_ms"],  width=bar_w, bottom=df_b["data_ms"], label="h2d_ms")
plt.bar(x, df_b["compute_ms"], width=bar_w,
        bottom=df_b["data_ms"] + df_b["h2d_ms"], label="compute_ms")
plt.xticks(x, df_b["config"], rotation=20)
plt.ylabel("Milliseconds per step")
plt.title(f"Time Breakdown per Step (batch={BATCH})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(res_dir, f"time_breakdown_batch{BATCH}_A0_A4.png"), dpi=300)
plt.close()

print("[OK] Wrote:",
      os.path.join(res_dir, "part1_all_results.csv"), "\n",
      os.path.join(res_dir, "throughput_comparison_A0_A4.png"), "\n",
      os.path.join(res_dir, f"time_breakdown_batch{BATCH}_A0_A4.png"))
'''
