import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Locate results CSVs (assumes this script is in PART-3/src)
# ---------------------------------------------------------------------
BASE = os.path.dirname(__file__)   # PART-3
res_dir = os.path.join(BASE, "results")

# Collect all per-world-size FedAvg results
csv_pattern = os.path.join(res_dir, "fedavg_results_*.csv")
csv_files = sorted(glob.glob(csv_pattern))

if not csv_files:
    raise FileNotFoundError(f"No fedavg_results_*.csv files found in {res_dir}")

dfs = []
for path in csv_files:
    df_part = pd.read_csv(path)
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)

# ---------------------------------------------------------------------
# Normalize config names a bit (optional but makes legends nicer)
#   - raw configs in CSV: "G0", "G1"
#   - we map them to human-friendly labels
# ---------------------------------------------------------------------
def canonical_cfg(name: str) -> str:
    name = str(name)
    if name.startswith("G0"):
        return "G0_baseline"
    if name.startswith("G1"):
        return "G1_overlap_amp"
    return name

df["config"] = df["config"].astype(str).apply(canonical_cfg)

# Keep only the two configs we care about
df = df[df["config"].isin(["G0_baseline", "G1_overlap_amp"])]
if df.empty:
    raise RuntimeError("No G0/G1 rows found in fedavg_results_*.csv")

# Optional: derive a total step time column (local train + comm)
df["step_ms"] = df["local_train_ms"] + df["comm_ms"]

# ---------------------------------------------------------------------
# Aggregate across multiple runs & ranks: mean per (world_size, config)
# ---------------------------------------------------------------------
agg = (
    df.groupby(["world_size", "config"])
      .agg(
          step_ms=("step_ms", "mean"),
          local_train_ms=("local_train_ms", "mean"),
          comm_ms=("comm_ms", "mean"),
          samples_per_s=("samples_per_s", "mean"),
      )
      .reset_index()
)

# For consistent x ordering
world_sizes = sorted(agg["world_size"].unique())

# ---------------------------------------------------------------------
# Plot 1: Throughput vs world size (line plot, G0 vs G1)
# ---------------------------------------------------------------------
plt.figure(figsize=(6.8, 4.2))

for cfg, label in [
    ("G0_baseline", "G0: baseline"),
    ("G1_overlap_amp", "G1: overlap+AMP"),
]:
    d = agg[agg["config"] == cfg].sort_values("world_size")
    if d.empty:
        continue
    plt.plot(
        d["world_size"],
        d["samples_per_s"],
        marker="o",
        linestyle="-",
        label=label,
    )

plt.xlabel("MPI world size (ranks)")
plt.ylabel("Samples / second (per rank, mean)")
plt.title("Part 3: FedAvg Throughput vs World Size")
plt.grid(True)
plt.legend()
plt.tight_layout()

out1 = os.path.join(res_dir, "part3_throughput_vs_world.png")
plt.savefig(out1, dpi=300)
plt.close()

# ---------------------------------------------------------------------
# Plot 2: Grouped stacked bars: latency breakdown vs world size
#   - x-axis: world_size
#   - for each world_size, two bars: G0 and G1
#   - each bar stacked: local_train_ms + comm_ms
# ---------------------------------------------------------------------
world_sizes = sorted(agg["world_size"].unique())
configs = ["G0_baseline", "G1_overlap_amp"]
labels = ["G0: baseline", "G1: overlap+AMP"]

# New distinct color families
color_map = {
    "G0_baseline": {
        "train": "#a6cee3",   # light blue
        "comm":  "#1f78b4",   # dark blue
    },
    "G1_overlap_amp": {
        "train": "#b2df8a",   # light green
        "comm":  "#33a02c",   # dark green
    }
}

x = np.arange(len(world_sizes))
bar_width = 0.35

plt.figure(figsize=(7.8, 4.6))

for idx, cfg in enumerate(configs):
    d = agg[agg["config"] == cfg].set_index("world_size").reindex(world_sizes)

    train_vals = d["local_train_ms"].fillna(0).values
    comm_vals  = d["comm_ms"].fillna(0).values

    # Shift G0 left, G1 right
    x_pos = x + (idx - 0.5) * bar_width

    # Stacked bar #1 (local_train)
    plt.bar(
        x_pos,
        train_vals,
        width=bar_width,
        color=color_map[cfg]["train"],
        label=f"{cfg} local_train_ms"
    )

    # Stacked bar #2 (comm)
    plt.bar(
        x_pos,
        comm_vals,
        width=bar_width,
        bottom=train_vals,
        color=color_map[cfg]["comm"],
        label=f"{cfg} comm_ms"
    )

plt.xticks(x, [str(w) for w in world_sizes])
plt.xlabel("MPI world size (ranks)")
plt.ylabel("Milliseconds per round (mean)")
plt.title("Part 3: FedAvg Time Breakdown (local_train + comm) [G0 vs G1]")

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(loc="upper left", fontsize=8)
plt.tight_layout()

out2 = os.path.join(res_dir, "part3_step_breakdown_grouped.png")
plt.savefig(out2, dpi=300)
plt.close()
