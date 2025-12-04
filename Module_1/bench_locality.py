import argparse, time, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

class Tiny1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(16, 2)
    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.head(x)

def measure_step(dl: DataLoader, device: torch.device, non_blocking: bool, iters: int=50):
    model = Tiny1D().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    model.train()

    # warmup
    it = iter(dl)
    for _ in range(5):
        try:
            x_cpu, y = next(it)
        except StopIteration:
            it = iter(dl); x_cpu, y = next(it)
        x = x_cpu.to(device, non_blocking=non_blocking) if device.type=='cuda' else x_cpu
        y = y.to(device, non_blocking=non_blocking) if device.type=='cuda' else y
        out = model(x); loss = nn.functional.cross_entropy(out, y)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)

    data_ms = h2d_ms = comp_ms = 0.0
    total_samples = 0
    it = iter(dl)
    for _ in range(iters):
        t0 = time.perf_counter()
        try:
            x_cpu, y = next(it)
        except StopIteration:
            it = iter(dl); x_cpu, y = next(it)
        t1 = time.perf_counter()

        if device.type == 'cuda':
            torch.cuda.synchronize()
            t_h0 = time.perf_counter()
            x = x_cpu.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            torch.cuda.synchronize()
            t_h1 = time.perf_counter()
        else:
            x, y = x_cpu, y
            t_h0 = t_h1 = time.perf_counter()

        t2 = time.perf_counter()
        out = model(x); loss = nn.functional.cross_entropy(out, y)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if device.type == 'cuda': torch.cuda.synchronize()
        t3 = time.perf_counter()

        data_ms += (t1 - t0) * 1000.0
        h2d_ms  += (t_h1 - t_h0) * 1000.0
        comp_ms += (t3 - t2) * 1000.0
        total_samples += x.shape[0]

    step_ms = (data_ms + h2d_ms + comp_ms) / iters
    sps = (total_samples / iters) / (step_ms / 1000.0)
    return dict(data_ms=data_ms/iters, h2d_ms=h2d_ms/iters, compute_ms=comp_ms/iters,
                step_ms=step_ms, samples_per_s=sps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mitbih","synthetic"], default="mitbih")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[64,128,256,512])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--contiguous", action="store_true")
    ap.add_argument("--random", dest="contiguous", action="store_false")
    ap.set_defaults(contiguous=True)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    # choose dataset maker
    if args.dataset == "mitbih":
        try:
            from .datasets.mitbih import make_mitbih_loader
            dl_maker = lambda bs,pin: make_mitbih_loader(batch_size=bs, num_workers=args.num_workers,
                                                         pin_memory=pin, contiguous=args.contiguous)
        except Exception as e:
            print(f"[WARN] MIT-BIH unavailable ({e}); Fallback to synthetic.")
            from .datasets.synth import make_synth_loader
            dl_maker = lambda bs,pin: make_synth_loader(batch_size=bs, num_workers=args.num_workers,
                                                        pin_memory=pin, contiguous=args.contiguous)
    else:
        from .datasets.synth import make_synth_loader
        dl_maker = lambda bs,pin: make_synth_loader(batch_size=bs, num_workers=args.num_workers,
                                                    pin_memory=pin, contiguous=args.contiguous)

    rows = []
    configs = [
        ("A0_baseline",         False, False),  # random, no pinned, blocking
        ("A1_contiguous",       False, False),  # contiguous sampler
        ("A2_contig_pinned",    True,  False),  # + pin_memory
        ("A3_contig_pinned_nb", True,  True),   # + non_blocking H2D
    ]

    for bs in args.batch_sizes:
        for name, pin, nb in configs:
            dl = dl_maker(bs, pin)
            stats = measure_step(dl, device, non_blocking=nb, iters=args.iters)
            row = dict(config=name, batch_size=bs, pin_memory=pin, contiguous=True, non_blocking=nb, **stats)
            print(row); rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, "part1_locality_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV -> {csv_path}")

    # Plot 1: throughput vs batch size
    plt.figure()
    for cfg in df['config'].unique():
        sub = df[df['config']==cfg].sort_values('batch_size')
        plt.plot(sub['batch_size'], sub['samples_per_s'], marker='o', label=cfg)
    plt.xlabel("Batch size"); plt.ylabel("Samples / second")
    plt.title("Throughput vs Batch Size")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "throughput_vs_batch.png"), bbox_inches="tight")
    plt.close()

    # Plot 2: stacked breakdown for largest batch
    max_bs = int(df['batch_size'].max())
    sub = df[df['batch_size']==max_bs][['config','data_ms','h2d_ms','compute_ms']].set_index('config')
    sub.plot(kind='bar', stacked=True)
    plt.xlabel("Configuration"); plt.ylabel("Milliseconds per step")
    plt.title(f"Time Breakdown per Step (batch={max_bs})")
    plt.savefig(os.path.join(results_dir, "time_breakdown_stacked.png"), bbox_inches="tight")
    plt.close()

    print("Plots saved in results/")

if __name__ == "__main__":
    main()
