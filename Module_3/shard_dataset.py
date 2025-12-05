# PART-3/src/shard_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def assign_shards_evenly(shard_paths, world_size, rank):
    """Return the list of shards assigned to this rank.
    Guarantees >=1 shard per rank (cycling if needed)."""
    if len(shard_paths) == 0:
        raise RuntimeError("No shards exist on disk.")

    # Cycle shards if we have fewer shards than ranks
    shards = sorted(shard_paths)
    assigned = []

    for i, s in enumerate(shards):
        if (i % world_size) == rank:
            assigned.append(s)

    # If a rank received none, give it 1 shard (wrap-around)
    if len(assigned) == 0:
        assigned = [shards[rank % len(shards)]]

    return assigned


def load_shard(path: str):
    """
    Reads one shard written by Part 1 shard_prep.py.

    Format:
      int64 N, int64 L, then N*L float32 values.
    Returns:
      x: np.ndarray [N, L], dtype float32
    """
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int64, count=2)
        if header.size != 2:
            raise RuntimeError(f"Bad shard header in {path}")
        N, L = int(header[0]), int(header[1])
        data = np.fromfile(f, dtype=np.float32, count=N * L)
    if data.size != N * L:
        raise RuntimeError(f"Unexpected size in {path}")
    return data.reshape(N, L)


class ShardDataset(Dataset):
    """
    Concatenates a list of shard files into a single dataset.

    For HPC timing we don't need real labels, so we generate
    dummy 2-class labels (all zeros).
    """

    def __init__(self, shard_paths, max_windows: int | None = None):
        xs = []
        for p in shard_paths:
            x = load_shard(p)
            xs.append(x)
        if not xs:
            raise RuntimeError("No shards assigned to this rank.")
        arr = np.concatenate(xs, axis=0).astype(np.float32)
        if max_windows is not None:
            arr = arr[:max_windows]

        self.x = torch.from_numpy(arr)                 # [N, L]
        self.y = torch.zeros(len(self.x), dtype=torch.long)  # dummy labels

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # Return as [1, L] so model sees channel dimension later
        return self.x[idx].unsqueeze(0), self.y[idx]




def make_dataloader(
    shard_paths,
    batch_size: int,
    max_windows: int | None = None,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    ds = ShardDataset(shard_paths, max_windows=max_windows)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return dl, len(ds)


# ------------ GPU-resident helpers (for Part 3) ------------

def load_shards_to_gpu(shard_paths, device, max_windows: int | None = None):
    """
    Load this rank's shards with ShardDataset, then move the full
    [N, L] tensor and labels to `device` once.

    Returns:
        x_gpu: [N, L] float32 on device
        y_gpu: [N] long on device
    """
    ds = ShardDataset(shard_paths, max_windows=max_windows)
    x_gpu = ds.x.to(device, non_blocking=True)
    y_gpu = ds.y.to(device, non_blocking=True)
    return x_gpu, y_gpu


def make_gpu_batch_iter(x_gpu, y_gpu, batch_size: int):
    """
    Infinite iterator that yields random mini-batches *on the GPU*.

    x_gpu: [N, L] on device
    y_gpu: [N] on device
    """
    N = x_gpu.size(0)
    device = x_gpu.device

    # Pre-allocate index tensor so we don't re-allocate every step
    base_idx = torch.arange(N, device=device)

    while True:
        # New random permutation each "epoch"
        perm = base_idx[torch.randperm(N, device=device)]
        for start in range(0, N - batch_size + 1, batch_size):
            sel = perm[start:start + batch_size]
            yield x_gpu[sel].unsqueeze(1), y_gpu[sel]

