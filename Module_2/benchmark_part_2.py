import os, time, ctypes, statistics as stats
import numpy as np
import torch
import pandas as pd

# ---- make matplotlib non-interactive BEFORE pyplot import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- Config --------------------
BATCH_SIZES  = [64, 128, 256, 512]
KERNEL_SIZES = [3, 5, 7]
L            = 500
NTHREADS     = os.cpu_count()  # use all cores; change if desired

TRIALS       = 15      # repeated measurements per (batch, K)
WARMUP_STEPS = 3       # light warmup before each timed trial
DTYPE        = np.float32

# -------------------- Paths --------------------
HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.abspath(os.path.join(HERE, os.pardir))            # .../PART-2
RES  = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

# -------------------- DLL load --------------------
dll_path = os.path.join(BASE, "src", "conv1d.dll")
if not os.path.exists(dll_path):
    raise FileNotFoundError(f"conv1d.dll not found at: {dll_path}")

conv1d = ctypes.CDLL(dll_path)
conv1d.conv1d_batch_omp_simd.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]
conv1d.conv1d_batch_omp_simd.restype = None

# Hint Torch to be deterministic-ish and use the same threads
torch.set_num_threads(NTHREADS)
try:
    torch.set_num_interop_threads(max(1, NTHREADS // 2))
except Exception:
    pass  # older torch versions may not have this

def run_omp_conv(x_np, w_np):
    batch, L_ = x_np.shape
    K = w_np.shape[0]
    outL = L_ - K + 1
    y_np = np.zeros((batch, outL), dtype=DTYPE)
    conv1d.conv1d_batch_omp_simd(
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        batch, L_, K, NTHREADS
    )
    return y_np

def time_once(fn, warmup_steps=WARMUP_STEPS):
    for _ in range(warmup_steps):
        fn()
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3  # ms

def bench_pair(bs, K, rng):
    # fixed inputs per (bs,K) across trials to reduce randomness
    x_np = rng.normal(0, 1, size=(bs, L)).astype(DTYPE)
    w_np = rng.normal(0, 1, size=(K,)).astype(DTYPE)

    # Torch module built once per (bs,K)
    xt4 = torch.from_numpy(x_np).unsqueeze(1)  # [B,1,L]
    wt = torch.from_numpy(w_np).reshape(1, 1, K)
    conv = torch.nn.Conv1d(1, 1, K, bias=False)
    with torch.no_grad():
        conv.weight[:] = wt

    def torch_step():
        _ = conv(xt4)

    def omp_step():
        _ = run_omp_conv(x_np, w_np)

    torch_ms, omp_ms = [], []
    for _ in range(TRIALS):
        torch_ms.append(time_once(torch_step))
        omp_ms.append(time_once(omp_step))

    agg = {
        "batch_size": bs,
        "kernel_size": K,
        "nthreads": NTHREADS,
        "torch_ms_median": float(stats.median(torch_ms)),
        "torch_ms_mean":   float(stats.fmean(torch_ms)),
        "torch_ms_std":    float(stats.pstdev(torch_ms)),
        "torch_ms_p95":    float(np.percentile(torch_ms, 95)),
        "omp_ms_median":   float(stats.median(omp_ms)),
        "omp_ms_mean":     float(stats.fmean(omp_ms)),
        "omp_ms_std":      float(stats.pstdev(omp_ms)),
        "omp_ms_p95":      float(np.percentile(omp_ms, 95)),
    }
    samples = bs
    agg["torch_sps"]  = samples / (agg["torch_ms_median"] / 1000.0)
    agg["omp_sps"]    = samples / (agg["omp_ms_median"] / 1000.0)
    agg["speedup_med"] = agg["torch_ms_median"] / agg["omp_ms_median"]
    return agg, torch_ms, omp_ms

def safe_write_csv(df, path):
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        # Excel or another process may be locking it; write a timestamped fallback.
        fallback = os.path.join(os.path.dirname(path),
                                f"{os.path.splitext(os.path.basename(path))[0]}_{int(time.time())}.csv")
        df.to_csv(fallback, index=False)
        print(f"[WARN] {os.path.abspath(path)} locked (Excel open?). Wrote {os.path.abspath(fallback)}")
        return fallback

def main():
    rng = np.random.default_rng(1337)
    rows, raw_rows = [], []

    for bs in BATCH_SIZES:
        for K in KERNEL_SIZES:
            print(f"\n=== Batch {bs} | Kernel {K} | threads={NTHREADS} ===")
            agg, t_trials, o_trials = bench_pair(bs, K, rng)
            rows.append(agg)
            print(f"Torch median: {agg['torch_ms_median']:.3f} ms | {agg['torch_sps']:.1f} sps")
            print(f"OMP   median: {agg['omp_ms_median']:.3f} ms | {agg['omp_sps']:.1f} sps")
            print(f"Speedup (median): {agg['speedup_med']:.2f}x")
            for idx, (tm, om) in enumerate(zip(t_trials, o_trials)):
                raw_rows.append({
                    "batch_size": bs, "kernel_size": K, "trial": idx,
                    "torch_ms": tm, "omp_ms": om
                })

    df     = pd.DataFrame(rows)
    df_raw = pd.DataFrame(raw_rows)

    out1 = safe_write_csv(df,     os.path.join(RES, "part2_openmp_results.csv"))
    out2 = safe_write_csv(df_raw, os.path.join(RES, "part2_openmp_results_raw.csv"))

    # --------- Plots ----------
    # Throughput (OMP) median ± std
    fig = plt.figure(figsize=(6.8, 4.2))
    for K in KERNEL_SIZES:
        d = df[df.kernel_size == K].sort_values("batch_size")
        ms     = d["omp_ms_median"].values
        bs_arr = d["batch_size"].values
        sps    = d["omp_sps"].values
        ms_std = d["omp_ms_std"].values
        sps_std = np.abs(-bs_arr * 1000.0 / (ms**2)) * ms_std
        plt.errorbar(d["batch_size"], sps, yerr=sps_std, marker="o", capsize=3, label=f"K={K}")
    plt.xlabel("Batch size"); plt.ylabel("Samples / second")
    plt.title("OpenMP+AVX2 Throughput (median ± std)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    th_png = os.path.join(RES, "part2_throughput.png")
    plt.savefig(th_png, dpi=300); plt.close(fig)

    # Speedup (median)
    fig = plt.figure(figsize=(6.8, 4.2))
    for K in KERNEL_SIZES:
        d = df[df.kernel_size == K].sort_values("batch_size")
        plt.plot(d["batch_size"], d["speedup_med"], marker="o", label=f"K={K}")
    plt.xlabel("Batch size"); plt.ylabel("Speedup (OMP / Torch, median)")
    plt.title("Part 2: Speedup over PyTorch CPU (median)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    sp_png = os.path.join(RES, "part2_speedup.png")
    plt.savefig(sp_png, dpi=300); plt.close(fig)

    print("\n[OK] Wrote:")
    print(" ", os.path.abspath(out1))
    print(" ", os.path.abspath(out2))
    print(" ", os.path.abspath(th_png))
    print(" ", os.path.abspath(sp_png))

if __name__ == "__main__":
    main()
