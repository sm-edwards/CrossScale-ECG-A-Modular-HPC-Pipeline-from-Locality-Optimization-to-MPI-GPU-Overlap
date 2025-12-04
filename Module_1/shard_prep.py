# shard_prep.py
#IGNORE THIS (EXPERIMENTAL)!!!!!
import os, numpy as np
from glob import glob
import time

OUT_DIR = "data/shards"
os.makedirs(OUT_DIR, exist_ok=True)

def write_shard(path, windows_np: np.ndarray):
    """
    windows_np: shape [N, L] float32
    Format: [int64 N][int64 L][N*L float32]
    """
    N, L = windows_np.shape
    with open(path, "wb") as f:
        f.write(np.asarray([N], dtype=np.int64).tobytes())
        f.write(np.asarray([L], dtype=np.int64).tobytes())
        windows_np.tofile(f)

def make_mitbih_windows(records=('100','101','103','105','106'),
                        win_len=500, stride=250, channel=0):
    try:
        import wfdb
    except Exception as e:
        raise RuntimeError("wfdb not installed. pip install wfdb") from e
    xs = []
    for rid in records:
        sig, info = wfdb.rdsamp(f"mitdb/{rid}", pn_dir="mitdb")
        x = sig[:, channel].astype(np.float32)
        for start in range(0, len(x) - win_len, stride):
            xs.append(x[start:start+win_len])
    return np.stack(xs, axis=0).astype(np.float32)  # [N,L]

def make_synth_windows(N=20000, L=500, seed=1337):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(N, L)).astype(np.float32)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mitbih","synthetic"], default="mitbih")
    ap.add_argument("--win_len", type=int, default=500)
    ap.add_argument("--stride", type=int, default=250)
    ap.add_argument("--shard_size", type=int, default=32768, help="windows per shard")
    args = ap.parse_args()

    start = time.perf_counter()

    if args.dataset == "mitbih":
        windows = make_mitbih_windows(win_len=args.win_len, stride=args.stride)
    else:
        windows = make_synth_windows(N=200000, L=args.win_len)

    load_end = time.perf_counter()
    N, L = windows.shape
    print(f"[prep] total windows: {N}, L={L}")
    i = 0
    shard_id = 0
    while i < N:
        j = min(i + args.shard_size, N)
        shard = windows[i:j]
        out = os.path.join(OUT_DIR, f"ecg_{shard_id:05d}.bin")
        write_shard(out, shard)
        print(f"[prep] wrote {out} with {len(shard)} windows")
        i = j
        shard_id += 1
    end = time.perf_counter()

    total_time = end - start
    io_time = end - load_end

    print(f"[prep] done. shards in {OUT_DIR}",
          f"[prep] dataset loaded in {load_end - start:.2f}s, "
          f"shards written in {io_time:.2f}s, total = {total_time:.2f}s"
          )

# --- write a metrics JSON so plots can amortize the cost ---
import json, time as _time, os
metrics = {
    "dataset": args.dataset,
    "total_windows": int(windows.shape[0]),
    "window_len": int(windows.shape[1]),
    "shard_size_windows": int(args.shard_size),
    "num_shards": int(shard_id),
    "load_time_s": float(load_end - start),
    "write_time_s": float(end - load_end),
    "total_time_s": float(end - start),
    "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
}
os.makedirs("results", exist_ok=True)
with open("results/shard_prep_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"[prep] metrics -> results/shard_prep_metrics.json")
