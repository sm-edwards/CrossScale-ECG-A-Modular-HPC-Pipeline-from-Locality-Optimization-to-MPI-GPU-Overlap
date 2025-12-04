# train_ecg_labl.py (IGNORE THIS <<EXPERIMENTAL>>)
import os, time, glob, csv
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from .labl_loader import LABLShardedReader, LABLPrefetcher

RESULTS_DIR = "results"; os.makedirs(RESULTS_DIR, exist_ok=True)

class Tiny1D(nn.Module):
    def __init__(self, in_ch=1, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, 7, padding=3), nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, 5, padding=2),    nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Linear(16, num_classes)
    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.head(x)

def bench_labl(shards_glob, batch_size=256, iters=100, normalize=True, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model = Tiny1D().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    model.train()

    shard_paths = sorted(glob.glob(shards_glob))
    assert shard_paths, f"No shards found for pattern: {shards_glob}"
    reader = LABLShardedReader(shard_paths)
    pf = LABLPrefetcher(reader, batch_size=batch_size, num_slots=4, normalize=normalize)
    pf.start()

    # warmup
    warm = 5
    for _ in range(warm):
        out = pf.next_batch_cpu()
        if out is None: break
        slot, x_cpu, _fill_ms = out
        n = x_cpu.size(0)
        y_cpu = torch.zeros(n, dtype=torch.long, pin_memory=True)
        x = x_cpu.to(device, non_blocking=True); y = y_cpu.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True); F.cross_entropy(model(x), y).backward(); opt.step()
        pf.recycle(slot)

    data_ms = h2d_ms = comp_ms = 0.0
    total_samples = 0
    n_steps = 0

    while n_steps < iters:
        out = pf.next_batch_cpu()
        if out is None: break
        slot, x_cpu, fill_ms = out
        n = x_cpu.size(0)
        # (1) H2D (single, batch-coalesced)
        if device.type == "cuda": torch.cuda.synchronize()
        t_h0 = time.perf_counter()
        x = x_cpu.to(device, non_blocking=True)
        y_cpu = torch.zeros(n, dtype=torch.long, pin_memory=True)
        y = y_cpu.to(device, non_blocking=True)
        if device.type == "cuda": torch.cuda.synchronize()
        t_h1 = time.perf_counter()

        # (2) Compute
        t_c0 = time.perf_counter()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if device.type == "cuda": torch.cuda.synchronize()
        t_c1 = time.perf_counter()

        # We consider "data_ms" as prefetch time hidden by background thread.
        # To expose it, you can instrument inside LABLPrefetcher; for now we
        # account for 0 here and emphasize h2d+compute (the data path win shows
        # up as increased throughput and reduced stalls).
        data_ms += float(fill_ms)
        h2d_ms  += (t_h1 - t_h0) * 1000.0
        comp_ms += (t_c1 - t_c0) * 1000.0
        total_samples += n
        n_steps += 1
        pf.recycle(slot)

    pf.shutdown()

    if n_steps == 0:
        return dict(step_ms=0, samples_per_s=0, data_ms=0, h2d_ms=0, compute_ms=0)

    step_ms = (data_ms + h2d_ms + comp_ms) / n_steps  # data_ms = pre-fetch
    sps = (total_samples / n_steps) / (step_ms / 1000.0)
    return dict(step_ms=step_ms, samples_per_s=sps,
                data_ms=data_ms / max(1, n_steps),
                h2d_ms=h2d_ms / n_steps,
                compute_ms=comp_ms / n_steps)

if __name__ == "__main__":
    import argparse, pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=str, default="data/shards/ecg_*.bin")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[64,128,256,512])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    rows = []
    for bs in args.batch_sizes:
        stats = bench_labl(args.shards, batch_size=bs, iters=args.iters, device=args.device)
        rows.append(dict(config="A4_LABL", batch_size=bs, **stats))
        print(rows[-1])

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "part1_labl_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"[OK] CSV -> {out_csv}")

    # Simple plot: throughput vs batch (LABL)
    plt.figure()
    df = df.sort_values("batch_size")
    plt.plot(df["batch_size"], df["samples_per_s"], marker="o", label="A4_LABL")
    plt.xlabel("Batch size"); plt.ylabel("Samples / second"); plt.title("LABL Throughput vs Batch")
    plt.legend(); plt.savefig(os.path.join(RESULTS_DIR, "labl_throughput.png"), bbox_inches="tight"); plt.close()
