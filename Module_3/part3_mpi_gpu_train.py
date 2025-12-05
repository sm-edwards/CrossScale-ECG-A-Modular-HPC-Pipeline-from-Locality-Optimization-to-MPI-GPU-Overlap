# PART-3/src/part3_mpi_gpu_train.py

"""
Part 3: MPI + CUDA/AMP for pseudo-federated ECG training.

Run (example, 2 ranks):

    mpiexec -n 2 python -m src.part3_mpi_gpu_train \
        --batch-size 256 --steps 200

This script expects Part 1 shards at: ../PART-1/data/shards/ecg_*.bin
"""

import os
import time
from glob import glob
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F

from mpi4py import MPI

from .tiny_ecg_model import TinyECG
from .shard_dataset import make_dataloader, assign_shards_evenly, load_shards_to_gpu, make_gpu_batch_iter


# ---------- Config / helpers ----------
import os
import pandas as pd

def append_results(df_new: pd.DataFrame, path: str, max_retries: int = 20):
    """
    Append df_new to a CSV at `path` without losing existing rows.
    If the file exists, we align columns to its header and append with no header.
    If it doesn't, we create it with a header.

    On Windows, the file can be briefly locked (e.g., from a previous run),
    so we retry a few times on PermissionError.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for attempt in range(max_retries):
        try:
            if os.path.exists(path):
                # Read only header to get column order
                existing_cols = pd.read_csv(path, nrows=0).columns.tolist()
                # Reindex to match existing columns (extra cols will be dropped; missing -> NaN)
                df_aligned = df_new.reindex(columns=existing_cols)
                df_aligned.to_csv(path, mode="a", header=False, index=False)
            else:
                df_new.to_csv(path, mode="w", header=True, index=False)
            # success – just return
            return
        except PermissionError:
            # brief backoff then retry
            time.sleep(0.25)

    # If we escape the loop, all retries failed
    raise RuntimeError(f"Could not write CSV after {max_retries} attempts: {path}")


@dataclass
class BenchStats:
    config: str
    world_size: int
    rank: int
    batch_size: int
    steps: int
    data_ms: float
    h2d_ms: float
    compute_ms: float
    step_ms: float
    samples_per_s: float


def mpi_avg(comm, value: float) -> float:
    return comm.allreduce(value, op=MPI.SUM) / comm.Get_size()


def setup_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def get_shards_for_rank(rank: int, world_size: int, base_dir: str):
    shard_paths = sorted(glob(os.path.join(base_dir, "ecg_*.bin")))
    if not shard_paths:
        raise RuntimeError(f"No shards found in {base_dir}")
    # Stripe shards across ranks
    my_shards = shard_paths[rank::world_size]
    return my_shards


# ---------- Baseline GPU (G0) ----------

def run_baseline_gpu(model, batch_iter, device, steps: int, rank: int, batch_size: int) -> BenchStats:
    """
    G0: pinned DataLoader, blocking H2D, default stream, no overlap.
    """

    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    data_ms = h2d_ms = compute_ms = 0.0
    step_ms = 0.0
    n_samples = 0
    n_steps = 0

    model.train()
    #loader_iter = iter(dl)

    while n_steps < steps:
        t_step0 = time.perf_counter()
        if rank == 0 and (n_steps % 10 == 0):
            print(f"[G0][rank {rank}] step {n_steps}/{steps}")

        '''
        # (1) CPU data load
        t_d0 = time.perf_counter()
        try:
            x_cpu, y_cpu = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dl)
            x_cpu, y_cpu = next(loader_iter)
        t_d1 = time.perf_counter()

        n = x_cpu.size(0)
        n_samples += n

        # (2) H2D blocking copy
        t_h0 = time.perf_counter()
        if device.type == "cuda":
            x = x_cpu.to(device, non_blocking=False)
            y = y_cpu.to(device, non_blocking=False)
            torch.cuda.synchronize()
        else:
            x, y = x_cpu, y_cpu
        t_h1 = time.perf_counter()
        '''

        # ---- GPU-resident batch ----
        x, y = next(batch_iter)        # already on device
        n = x.size(0)
        n_samples += n


        # (3) Compute
        t_c0 = time.perf_counter()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_c1 = time.perf_counter()

        t_step1 = time.perf_counter()

        data_ms += data_ms#(t_d1 - t_d0) * 1e3
        h2d_ms += h2d_ms#(t_h1 - t_h0) * 1e3
        compute_ms += (t_c1 - t_c0) * 1e3
        step_ms += (t_step1 - t_step0) * 1e3
        n_steps += 1

    avg_step_ms = step_ms / n_steps
    sps = (n_samples / n_steps) / (avg_step_ms / 1000.0)

    return BenchStats(
        config="G0_baseline_GPU_CACHE",
        world_size=MPI.COMM_WORLD.Get_size(),
        rank=rank,
        batch_size=batch_size,
        steps=n_steps,
        data_ms=data_ms / n_steps,
        h2d_ms=h2d_ms / n_steps,
        compute_ms=compute_ms / n_steps,
        step_ms=avg_step_ms,
        samples_per_s=sps,
    )


'''
# ---------- Overlap GPU (G1) ----------

def run_overlap_gpu(model, batch_iter, device, steps: int, rank: int) -> BenchStats:
    """
    G1: pinned DataLoader, non-blocking H2D on a separate stream, AMP,
    one-batch lookahead (double buffer). Compute runs on default stream.
    """

    assert device.type == "cuda", "Overlap config requires CUDA"

    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler()

    h2d_stream = torch.cuda.Stream(priority=0)

    data_ms = h2d_ms = compute_ms = 0.0
    step_ms = 0.0
    n_samples = 0
    n_steps = 0

    model.train()
    loader_iter = iter(dl)


    # Prefetch first batch to GPU
    t_d0 = time.perf_counter()
    try:
        x_cpu, y_cpu = next(loader_iter)
    except StopIteration:
        loader_iter = iter(dl)
        x_cpu, y_cpu = next(loader_iter)
    t_d1 = time.perf_counter()

    with torch.cuda.stream(h2d_stream):
        t_h0 = time.perf_counter()
        x_next = x_cpu.to(device, non_blocking=True)
        y_next = y_cpu.to(device, non_blocking=True)
        t_h1 = time.perf_counter()
    # No compute yet; just set prefetch
    torch.cuda.current_stream().wait_stream(h2d_stream)
    x_prev, y_prev = x_next, y_next
    data_ms += (t_d1 - t_d0) * 1e3
    h2d_ms += (t_h1 - t_h0) * 1e3
    n_samples += x_prev.size(0)
    n_steps += 0  # first batch will be counted in loop below

    # Main loop: at each iteration we
    #   (1) launch H2D of next batch on h2d_stream
    #   (2) compute on previous batch on default stream

    while n_steps < steps:
        t_step0 = time.perf_counter()

        # ---- Launch H2D for next batch (if available) ----
        t_d0 = time.perf_counter()
        try:
            x_cpu, y_cpu = next(loader_iter)
            got_next = True
        except StopIteration:
            got_next = False
        t_d1 = time.perf_counter()

        if got_next:
            with torch.cuda.stream(h2d_stream):
                t_h0 = time.perf_counter()
                x_next = x_cpu.to(device, non_blocking=True)
                y_next = y_cpu.to(device, non_blocking=True)
                t_h1 = time.perf_counter()
            data_ms += (t_d1 - t_d0) * 1e3
            h2d_ms += (t_h1 - t_h0) * 1e3

        # ---- Compute on previous batch ----
        torch.cuda.current_stream().wait_stream(h2d_stream)

        t_c0 = time.perf_counter()
        with torch.cuda.amp.autocast():
            logits = model(x_prev)
            loss = F.cross_entropy(logits, y_prev)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t_c1 = time.perf_counter()

        n = x_prev.size(0)
        n_samples += n
        compute_ms += (t_c1 - t_c0) * 1e3
        n_steps += 1

        # Shift buffer
        if got_next:
            x_prev, y_prev = x_next, y_next

        t_step1 = time.perf_counter()
        step_ms += (t_step1 - t_step0) * 1e3

        if not got_next:
            # No more data to prefetch; break after this compute
            break

    avg_step_ms = step_ms / n_steps
    sps = (n_samples / n_steps) / (avg_step_ms / 1000.0)

    return BenchStats(
        config="G1_overlap_amp",
        world_size=MPI.COMM_WORLD.Get_size(),
        rank=rank,
        batch_size=dl.batch_size,
        steps=n_steps,
        data_ms=data_ms / max(1, n_steps),
        h2d_ms=h2d_ms / max(1, n_steps),
        compute_ms=compute_ms / n_steps,
        step_ms=avg_step_ms,
        samples_per_s=sps,
    )
'''
def run_overlap_gpu(model, batch_iter, device, steps: int, rank: int, batch_size: int) -> BenchStats:
    """
    G1 (GPU-cached variant):
      - Batches come from `batch_iter` already on the GPU.
      - No per-step H2D; we treat h2d_ms as 0.
      - Still uses AMP and a one-batch lookahead pattern.
    """
    assert device.type == "cuda", "Overlap config requires CUDA"

    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler()

    # No H2D stream anymore – data is already on GPU
    data_ms = 0.0
    h2d_ms = 0.0
    compute_ms = 0.0
    step_ms = 0.0
    n_samples = 0
    n_steps = 0

    model.train()
    loader_iter = iter(batch_iter)   # <-- use the GPU batch iterator

    # ---- Prefetch first GPU batch ----
    t_d0 = time.perf_counter()
    try:
        x_prev, y_prev = next(loader_iter)   # already on device
    except StopIteration:
        # No data; return empty stats
        return BenchStats(
            config="G1_overlap_amp",
            world_size=MPI.COMM_WORLD.Get_size(),
            rank=rank,
            batch_size=0,
            steps=0,
            data_ms=0.0,
            h2d_ms=0.0,
            compute_ms=0.0,
            step_ms=0.0,
            samples_per_s=0.0,
        )
    t_d1 = time.perf_counter()
    data_ms += (t_d1 - t_d0) * 1e3
    n_samples += x_prev.size(0)

    # ---- Main loop ----
    while n_steps < steps:
        t_step0 = time.perf_counter()
        if rank == 0 and (n_steps % 10 == 0):
            print(f"[G1][rank {rank}] step {n_steps}/{steps}")

        # Try to prefetch the *next* GPU batch from the iterator
        t_d0 = time.perf_counter()
        try:
            x_next, y_next = next(loader_iter)  # already on device
            got_next = True
        except StopIteration:
            got_next = False
        t_d1 = time.perf_counter()
        if got_next:
            data_ms += (t_d1 - t_d0) * 1e3

        # ---- Compute on previous batch ----
        t_c0 = time.perf_counter()
        with torch.cuda.amp.autocast():
            logits = model(x_prev)
            loss = F.cross_entropy(logits, y_prev)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t_c1 = time.perf_counter()

        n = x_prev.size(0)
        n_samples += n
        compute_ms += (t_c1 - t_c0) * 1e3
        n_steps += 1

        # Shift buffer to the prefetched batch
        if got_next:
            x_prev, y_prev = x_next, y_next

        t_step1 = time.perf_counter()
        step_ms += (t_step1 - t_step0) * 1e3

        if not got_next:  # ran out of data
            break

    avg_step_ms = step_ms / max(1, n_steps)
    sps = 0.0
    if avg_step_ms > 0 and n_steps > 0:
        sps = (n_samples / n_steps) / (avg_step_ms / 1000.0)

    return BenchStats(
        config="G1_overlap_amp",
        world_size=MPI.COMM_WORLD.Get_size(),
        rank=rank,
        batch_size=x_prev.size(0),
        steps=n_steps,
        data_ms=data_ms / max(1, n_steps),
        h2d_ms=0.0,                     # per-step H2D is now eliminated
        compute_ms=compute_ms / max(1, n_steps),
        step_ms=avg_step_ms,
        samples_per_s=sps,
    )





# ---------- Main MPI driver ----------

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200,
                    help="number of training steps per config")
    ap.add_argument("--max-windows", type=int, default=20000,
                    help="limit windows per rank for memory control")
    ap.add_argument("--data-root", type=str,
                    default=os.path.join("..", "PART-1", "data", "shards"))
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    device = setup_device()
    if rank == 0:
        print(f"[MPI] world_size={world_size}, device={device}")

    # Discover all shard files
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
    shard_root = os.path.join(PROJECT_ROOT, "PART-1", "data", "shards")  # or your actual path
    all_shards = sorted(glob(os.path.join(shard_root, "ecg_*.bin")))

    if rank == 0:
        print(f"[MPI] Found {len(all_shards)} shard(s) in {shard_root}")
    comm.Barrier()

    # Assign a non-empty shard list to each rank
    local_shards = assign_shards_evenly(all_shards, world_size, rank)
    print(f"[rank {rank}] using {len(local_shards)} shard(s)")

    '''
    dl, n_windows = make_dataloader(
        local_shards,
        batch_size=args.batch_size,
        max_windows=args.max_windows,
        num_workers=2,
        pin_memory=True,
    )
    '''
    # GPU-resident data: load once per rank
    x_gpu, y_gpu = load_shards_to_gpu(
        local_shards, device=device, max_windows=args.max_windows
    )
    n_windows = x_gpu.size(0)
    if rank == 0:
        print(f"[MPI] Each rank sees up to {n_windows} windows (GPU-cached)")

    batch_iter = make_gpu_batch_iter(x_gpu, y_gpu, args.batch_size)


    # NEW: cap the number of steps so we never overrun the dataset too much
    max_steps_epoch = n_windows // args.batch_size
    if max_steps_epoch == 0:
        raise RuntimeError("Not enough windows for a single batch on this rank.")

    effective_steps = min(args.steps, max_steps_epoch)
    if rank == 0:
        print(f"[MPI] Limiting steps per config to {effective_steps} "
              f"(<= one epoch of {max_steps_epoch} steps)")


    # --- Baseline GPU config ---
    model = TinyECG(num_classes=2)
    stats0 = run_baseline_gpu(model, batch_iter, device, steps=args.steps, rank=rank, batch_size=args.batch_size)

    # --- Overlap+AMP config (if CUDA) ---
    stats1 = None
    if device.type == "cuda":
        model = TinyECG(num_classes=2)
        stats1 = run_overlap_gpu(model, batch_iter, device, steps=args.steps, rank=rank, batch_size=args.batch_size)
    else:
        if rank == 0:
            print("[WARN] CUDA not available; skipping overlap config")

    # --- Aggregate and write results on rank 0 ---
    import pandas as pd

    local_rows = [asdict(stats0)]
    if stats1 is not None:
        local_rows.append(asdict(stats1))

    # Gather dicts from all ranks
    all_rows = comm.gather(local_rows, root=0)

    if rank == 0:
        flat = [r for per_rank in all_rows for r in per_rank]
        df = pd.DataFrame(flat)

        base = os.path.dirname(os.path.dirname(__file__))
        out_dir = os.path.join(base, "results")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "part3_mpi_cuda_results.csv")
        
        append_results(df, csv_path)
        print(f"[OK] Appended results to {csv_path}")

        # Quick sanity print: global averages per config
        for cfg, grp in df.groupby("config"):
            print(f"\n=== {cfg} (global avg over {world_size} ranks) ===")
            print(f"  step_ms       ~ {grp['step_ms'].mean():.3f}")
            print(f"  data_ms       ~ {grp['data_ms'].mean():.3f}")
            print(f"  h2d_ms        ~ {grp['h2d_ms'].mean():.3f}")
            print(f"  compute_ms    ~ {grp['compute_ms'].mean():.3f}")
            print(f"  samples/s     ~ {grp['samples_per_s'].mean():.1f}")


if __name__ == "__main__":
    main()
