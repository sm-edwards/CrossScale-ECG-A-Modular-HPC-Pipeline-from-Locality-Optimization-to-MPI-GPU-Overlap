#PLOT LOCALITY ONLY A0-A3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot A0–A3 locality results only")
    ap.add_argument("--results-dir", default=None,
                    help="Path to results/ (default: auto-detect)")
    ap.add_argument("--csv", default="part1_locality_results.csv",
                    help="CSV with A0–A3 results")
    ap.add_argument("--batch", type=int, default=512,
                    help="Batch size to use for time-breakdown bar chart")
    args = ap.parse_args()

    # project root = parent of this file's directory
    base = os.path.dirname(os.path.dirname(__file__))
    res_dir = args.results_dir or os.path.join(base, "results")
    csv_path = os.path.join(res_dir, args.csv)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # keep only A0–A3
    keep = {"A0_baseline", "A1_contiguous", "A2_contig_pinned", "A3_contig_pinned_nb"}
    df = df[df["config"].isin(keep)].copy()

    # -------- Plot 1: Throughput vs batch (A0–A3) --------
    plt.figure(figsize=(6.2, 4.2))
    for name, grp in df.groupby("config"):
        grp = grp.sort_values("batch_size")
        plt.plot(grp["batch_size"], grp["samples_per_s"], marker="o", label=name)
    plt.xlabel("Batch size")
    plt.ylabel("Samples / second")
    plt.title("Throughput Comparison (A0–A3)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join(res_dir, "throughput_A0_A3.png")
    plt.savefig(out1, dpi=300)
    plt.close()

    # -------- Plot 2: Stacked time breakdown at chosen batch --------
    BATCH = args.batch
    df_b = df[df["batch_size"] == BATCH].copy()

    # Stable order
    order = ["A0_baseline", "A1_contiguous", "A2_contig_pinned", "A3_contig_pinned_nb"]
    df_b["config"] = pd.Categorical(df_b["config"], categories=order, ordered=True)
    df_b = df_b.sort_values("config")

    plt.figure(figsize=(6.6, 4.2))
    x = range(len(df_b))
    w = 0.6
    plt.bar(x, df_b["data_ms"], width=w, label="data_ms")
    plt.bar(x, df_b["h2d_ms"], width=w, bottom=df_b["data_ms"], label="h2d_ms")
    plt.bar(x, df_b["compute_ms"], width=w,
            bottom=df_b["data_ms"] + df_b["h2d_ms"], label="compute_ms")
    plt.xticks(x, df_b["config"], rotation=20)
    plt.ylabel("Milliseconds per step")
    plt.title(f"Time Breakdown per Step (batch={BATCH})")
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(res_dir, f"time_breakdown_batch{BATCH}_A0_A3.png")
    plt.savefig(out2, dpi=300)
    plt.close()

    print("[OK] Wrote:")
    print("  ", out1)
    print("  ", out2)

if __name__ == "__main__":
    main()
