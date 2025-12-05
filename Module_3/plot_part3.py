import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Locate results CSV (assumes this script is in PART-3/src)
# ---------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(__file__))   # PART-3
res_dir = os.path.join(BASE, "results")
csv_path = os.path.join(res_dir, "part3_mpi_cuda_results.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find results CSV at: {csv_path}")

df = pd.read_csv(csv_path)

# ---------------------------------------------------------------------
# Normalize config names a bit (optional but makes legends nicer)
# ---------------------------------------------------------------------
# You may have plain "G0_baseline" / "G1_overlap_amp" or the *_gpu_cache variants.
def canonical_cfg(name: str) -> str:
    if name.startswith("G0"):
        return "G0_baseline"
    if name.startswith("G1"):
        return "G1_overlap_amp"
    return name

df["config"] = df["config"].astype(str).apply(canonical_cfg)

# Keep only the two configs we care about
df = df[df["config"].isin(["G0_baseline", "G1_overlap_amp"])]
if df.empty:
    raise RuntimeError("No G0/G1 rows found in part3_mpi_cuda_results.csv")

# ---------------------------------------------------------------------
# Aggregate across multiple runs & ranks: mean per (world_size, config)
# ---------------------------------------------------------------------
agg = (
    df.groupby(["world_size", "config"])
      .agg(
          step_ms=("step_ms", "mean"),
          data_ms=("data_ms", "mean"),
          h2d_ms=("h2d_ms", "mean"),
          compute_ms=("compute_ms", "mean"),
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
plt.title("Part 3: Throughput vs World Size")
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
#   - each bar stacked: data_ms + h2d_ms + compute_ms
# ---------------------------------------------------------------------
configs = ["G0_baseline", "G1_overlap_amp"]
labels = ["G0: baseline", "G1: overlap+AMP"]
colors = {
    "data_ms":   "#a6cee3",
    "h2d_ms":    "#1f78b4",
    "compute_ms": "#33a02c",
}

x = np.arange(len(world_sizes))  # one group per world size
bar_width = 0.35                 # width per bar within the group

plt.figure(figsize=(7.2, 4.4))

for idx, (cfg, lab) in enumerate(zip(configs, labels)):
    d = agg[agg["config"] == cfg]
    d = d.set_index("world_size").reindex(world_sizes)

    if d["step_ms"].isna().all():
        continue

    x_pos = x + (idx - 0.5) * bar_width

    h2d_vals = d["h2d_ms"].fillna(0.0).values
    comp_vals = d["compute_ms"].fillna(0.0).values

    # Stack ONLY: h2d and compute
    b1 = plt.bar(
        x_pos, h2d_vals,
        width=bar_width,
        color=colors["h2d_ms"],
        label=(lab + " h2d_ms") if idx == 0 else None,
    )
    b2 = plt.bar(
        x_pos, comp_vals,
        width=bar_width,
        bottom=h2d_vals,
        color=colors["compute_ms"],
        label=(lab + " compute_ms") if idx == 0 else None,
    )

plt.xticks(x, [str(w) for w in world_sizes])
plt.xlabel("MPI world size (ranks)")
plt.ylabel("Milliseconds per step (mean)")
plt.title("Part 3: Time Breakdown (h2d + compute only) [G0 + G1]")

# Legend for h2d + compute
handles, leg_labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], ["h2d_ms", "compute_ms"], loc="upper left")

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

out2 = os.path.join(res_dir, "part3_step_breakdown_grouped.png")
plt.savefig(out2, dpi=300)
plt.close()

print("[OK] Wrote:")
print(" ", out1)
print(" ", out2)
