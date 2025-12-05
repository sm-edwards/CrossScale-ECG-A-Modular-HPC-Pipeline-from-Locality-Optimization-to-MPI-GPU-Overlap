# ================================================================
# TRUEFL/part3_fedavg_overlap_mpi_gpu.py
# ---------------------------------------------------------------
# Pseudo-Federated Learning with MPI + CUDA + AMP + GPU-overlap
#
# EDIT THESE PLACEHOLDERS:
#    <DATA_ROOT>  -> path to shared ecg_*.bin shards
# ================================================================

import os, time
from glob import glob
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
from mpi4py import MPI
import pandas as pd

from shard_dataset import (
    assign_shards_evenly,
    load_shards_to_gpu,
    make_gpu_batch_iter
)

from tiny_ecg_model import TinyECG


# ---------------- CSV helper ----------------

def append_results(df_new, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        existing_cols = pd.read_csv(path, nrows=0).columns.tolist()
        df_aligned = df_new.reindex(columns=existing_cols)
        df_aligned.to_csv(path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(path, mode="w", header=True, index=False)


# ---------------- Stats per round ----------------

@dataclass
class RoundStats:
    config: str
    world_size: int
    rank: int
    round_idx: int
    batch_size: int
    local_steps: int
    local_train_ms: float
    comm_ms: float
    samples_per_s: float
    avg_loss: float


# ---------------- Utilities ----------------

def setup_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def set_basic_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- Model sync (FedAvg) ----------------

def broadcast_model(comm, model):
    rank = comm.Get_rank()

    for p in model.parameters():
        if rank == 0:
            arr = p.data.detach().cpu().numpy()
        else:
            arr = np.empty_like(p.data.detach().cpu().numpy())

        comm.Bcast(arr, root=0)
        p.data = torch.from_numpy(arr).to(p.data.device)


def fedavg_allreduce(comm, model):
    world = comm.Get_size()

    for p in model.parameters():
        arr = p.data.detach().cpu().numpy()
        recv = np.empty_like(arr)

        comm.Allreduce(arr, recv, op=MPI.SUM)
        recv /= float(world)

        p.data = torch.from_numpy(recv).to(p.data.device)


# ---------------- Local training ----------------

def train_step_G0(model, x, y, opt, device):
    model.train()
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    return float(loss.item())


def train_step_G1(model, x, y, opt, scaler, device):
    model.train()
    opt.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast():
        logits = model(x)
        loss = F.cross_entropy(logits, y)

    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize()

    return float(loss.item())


# ---------------- Main ----------------

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=str)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local-steps", type=int, default=50)
    ap.add_argument("--max-windows", type=int, default=30000)
    ap.add_argument("--config", choices=["G0", "G1", "both"], default="both")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    set_basic_seeds(1234 + rank)
    device = setup_device()

    shard_paths = sorted(glob(os.path.join(args.data_root, "ecg_*.bin")))
    if not shard_paths:
        raise RuntimeError("No ecg_*.bin files found")

    local_shards = assign_shards_evenly(shard_paths, world, rank)

    x_gpu, y_gpu = load_shards_to_gpu(
        local_shards, device=device, max_windows=args.max_windows
    )
    batch_iter = make_gpu_batch_iter(x_gpu, y_gpu, args.batch_size)

    BASE = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(os.path.dirname(BASE), "results", "fedavg_results.csv")

    configs = []
    if args.config in ("G0", "both"):
        configs.append("G0")
    if args.config in ("G1", "both"):
        configs.append("G1")

    for cfg in configs:
        model = TinyECG(num_classes=2).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        scaler = torch.cuda.amp.GradScaler() if (cfg == "G1") else None

        round_rows = []

        for r in range(args.rounds):
            # Broadcast global model
            t_c0 = time.perf_counter()
            broadcast_model(comm, model)
            t_c1 = time.perf_counter()

            # Local steps
            t_l0 = time.perf_counter()
            n_samples = 0
            loss_acc = 0.0

            for _ in range(args.local_steps):
                x, y = next(batch_iter)
                n_samples += x.size(0)

                if cfg == "G0":
                    loss_acc += train_step_G0(model, x, y, opt, device)
                else:
                    loss_acc += train_step_G1(model, x, y, opt, scaler, device)

            t_l1 = time.perf_counter()

            # FedAvg reduce
            t_c2 = time.perf_counter()
            fedavg_allreduce(comm, model)
            t_c3 = time.perf_counter()

            local_ms = (t_l1 - t_l0) * 1000
            comm_ms = (t_c1 - t_c0 + t_c3 - t_c2) * 1000

            samples_s = n_samples / (local_ms / 1000)

            round_rows.append(asdict(
                RoundStats(
                    config=cfg,
                    world_size=world,
                    rank=rank,
                    round_idx=r,
                    batch_size=args.batch_size,
                    local_steps=args.local_steps,
                    local_train_ms=local_ms,
                    comm_ms=comm_ms,
                    samples_per_s=samples_s,
                    avg_loss=loss_acc / args.local_steps,
                )
            ))

        gathered = comm.gather(round_rows, root=0)

        if rank == 0:
            flat = [row for rlist in gathered for row in rlist]
            df = pd.DataFrame(flat)
            append_results(df, csv_path)
            print(f"[OK] wrote {len(flat)} rows -> {csv_path}")


if __name__ == "__main__":
    main()
