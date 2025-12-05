import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/part2_openmp_simd_results.csv")

# scaling
plt.figure(figsize=(6,4))
for bs, grp in df.groupby("batch"):
    plt.plot(grp["threads"], grp["samples_per_s"], marker="o", label=f"BS={bs}")
plt.xlabel("Threads"); plt.ylabel("Samples/s")
plt.title("OpenMP Scaling")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig("results/part2_scaling.png", dpi=300)

# SIMD vs no-SIMD (requires optional runs)
# ... can be added if needed
