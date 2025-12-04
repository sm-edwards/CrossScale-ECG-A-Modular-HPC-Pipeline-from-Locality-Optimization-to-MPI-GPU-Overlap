# labl_loader.py (IGNORE THIS!! EXPERIMENTAL)
import mmap, os, threading, queue, numpy as np, torch
from contextlib import contextmanager
import time
import queue

class LABLShardedReader:
    """
    Binary shard format:
      [int64 N][int64 L] followed by N * L * float32 (row-major, contiguous)
    Windows are stored sequentially to favor OS readahead.
    """
    def __init__(self, shard_paths):
        self.paths = list(shard_paths)

    @contextmanager
    def open_shard(self, path):
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.path.getsize(path)
            mm = mmap.mmap(fd, length=size, access=mmap.ACCESS_READ)
            N = int(np.frombuffer(mm, dtype=np.int64, count=1, offset=0)[0])
            L = int(np.frombuffer(mm, dtype=np.int64, count=1, offset=8)[0])
            base = 16
            yield (mm, base, N, L)
        finally:
            mm.close()
            os.close(fd)

class PinnedRing:
    """Ring of preallocated page-locked slabs shaped like a batch [B,1,L]."""
    def __init__(self, num_slots, batch_shape, dtype=torch.float32):
        self.slots = [torch.empty(batch_shape, pin_memory=True, dtype=dtype)
                      for _ in range(num_slots)]
        self.q_free, self.q_full = queue.Queue(), queue.Queue()
        for i in range(num_slots): self.q_free.put(i)

class LABLPrefetcher:
    """
    Background thread: reads windows sequentially via mmap, optional normalize,
    writes into pinned slabs. Training loop does a single H2D per batch.
    """
    def __init__(self, reader: LABLShardedReader, batch_size: int,
                 num_slots: int = 4, normalize: bool = True):
        self.reader = reader
        # probe L from first shard
        with reader.open_shard(reader.paths[0]) as (mm, base, N, L):
            self.L = L
        self.B = batch_size
        self.ring = PinnedRing(num_slots, (batch_size, 1, self.L), dtype=torch.float32)
        self.normalize = normalize
        self.stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self._shard_idx, self._offset = 0, 0  # window offset within shard

    def start(self): self.thread.start()
    def shutdown(self):
        self.stop = True
        self.thread.join(timeout=2.0)

    def _fill_one(self, mm, base, idx, L, out_tensor_view):
        off = base + (idx * L * 4)
        arr = np.frombuffer(mm, dtype=np.float32, count=L, offset=off)

        if self.normalize:
            # keep intermediate math in float64, but cast back to float32 once
            m = arr.mean(dtype=np.float64)
            s = arr.std(dtype=np.float64) + 1e-8
            arr = ((arr - m) / s).astype(np.float32, copy=False)

        # Copy numpy -> pinned torch view; dtypes now match (float32)
        out_tensor_view.copy_(torch.from_numpy(arr), non_blocking=False)

    def _run(self):
        try:
            while not self.stop:
                try:
                    slot = self.ring.q_free.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self.stop:
                    self.ring.q_full.put((slot, 0, 0.0));
                    break

                if self._shard_idx >= len(self.reader.paths):
                    # end of data
                    self.ring.q_full.put((slot, 0, 0.0))
                    break

                path = self.reader.paths[self._shard_idx]
                with self.reader.open_shard(path) as (mm, base, N, L):
                    n_filled = 0
                    batch = self.ring.slots[slot]
                    t0 = time.perf_counter()
                    while n_filled < self.B and not self.stop:
                        if self._offset >= N:
                            self._shard_idx += 1
                            self._offset = 0
                            break
                        self._fill_one(mm, base, self._offset, L, batch[n_filled, 0, :])
                        n_filled += 1
                        self._offset += 1
                    fill_ms = (time.perf_counter() - t0) * 1e3
                    # hand off: (slot, n_filled, fill_ms)
                    self.ring.q_full.put((slot, n_filled, fill_ms))
        except Exception:
            # wake main loop and re-raise
            try:
                self.ring.q_full.put_nowait((0, 0, 0.0))
            except Exception:
                pass
            raise

    def next_batch_cpu(self):
        # returns (slot, batch_view, fill_ms) or None
        while not self.stop:
            try:
                slot, n, fill_ms = self.ring.q_full.get(timeout=0.1)
                break
            except queue.Empty:
                continue
        else:
            return None

        if n == 0:
            self.ring.q_free.put(slot)
            return None
        return slot, self.ring.slots[slot][:n], float(fill_ms)

    def recycle(self, slot: int) -> None:
        """Return a ring slot to the free queue."""
        try:
            self.ring.q_free.put_nowait(slot)
        except Exception:
            # best-effort: don't crash training if queue is temporarily full
            pass
