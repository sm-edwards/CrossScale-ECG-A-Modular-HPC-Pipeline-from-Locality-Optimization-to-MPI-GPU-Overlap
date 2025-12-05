import os, time, ctypes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DLL  = os.path.join(HERE, "conv1d.dll")   # built in step 4
lib  = ctypes.CDLL(DLL)

conv = lib.conv1d_batch_omp_simd
conv.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # x
    ctypes.POINTER(ctypes.c_float),  # w
    ctypes.POINTER(ctypes.c_float),  # y
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32
]
conv.restype = None

def run(batch=256, L=500, K=32, threads=8, iters=50, warmup=5, seed=1337):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(batch, L)).astype(np.float32)
    w = rng.normal(0, 1, size=(K,)).astype(np.float32)
    y = np.zeros((batch, L-K+1), np.float32)

    xp = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    wp = w.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    yp = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # warmup
    for _ in range(warmup):
        conv(xp, wp, yp, batch, L, K, threads)

    t0 = time.perf_counter()
    for _ in range(iters):
        conv(xp, wp, yp, batch, L, K, threads)
    t1 = time.perf_counter()

    step_ms = (t1 - t0) * 1000.0 / iters
    sps = batch / (step_ms / 1000.0)
    return step_ms, sps

if __name__ == "__main__":
    out_dir = os.path.join(HERE, "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for th in [1, 2, 4, 8, 16]:
        for bs in [64, 128, 256, 512]:
            ms, sps = run(batch=bs, threads=th)
            rows.append({"threads": th, "batch": bs,
                         "compute_ms": ms, "samples_per_s": sps})
            print(f"threads={th:2d} bs={bs:3d}  compute_ms={ms:7.2f}  sps={sps:9.1f}")

    df = pd.DataFrame(rows)
    csv = os.path.join(out_dir, "part2_openmp_simd_results.csv")
    df.to_csv(csv, index=False)
    print("[OK] wrote", csv)

    # quick scaling plot
    plt.figure(figsize=(6.2,4.2))
    for bs, grp in df.groupby("batch"):
        g = grp.sort_values("threads")
        plt.plot(g["threads"], g["samples_per_s"], marker="o", label=f"BS={bs}")
    plt.xlabel("Threads"); plt.ylabel("Samples / second")
    plt.title("OpenMP + AVX2 CPU Scaling")
    plt.grid(True); plt.legend()
    png = os.path.join(out_dir, "part2_scaling.png")
    plt.tight_layout(); plt.savefig(png, dpi=300); plt.close()
    print("[OK] wrote", png)
