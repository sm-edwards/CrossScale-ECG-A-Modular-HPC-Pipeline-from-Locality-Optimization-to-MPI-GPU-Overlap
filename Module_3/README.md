# HPC Course Project – Pseudo-Federated ECG Workload

Author: Shawn Edwards  
Course: High Performance Computing (Fall 2025)  
Instructor: Dr. Abdullah Al-Mamun

This project studies how *classical HPC mechanisms* impact the
performance of an ECG classification workload derived from the
MIT-BIH Arrhythmia Database.  Rather than optimizing accuracy, the
focus is on **data locality**, **CPU parallelism/SIMD**, and
**GPU/MPI execution**.

The project is implemented in three parts:

1. **Part 1 – Data and Cache Locality (CPU + GPU H2D)**
2. **Part 2 – OpenMP + AVX2 1D Convolution Kernel (CPU)**
3. **Part 3 – MPI + CUDA Streams/AMP (GPU, federated)**

All parts use a small 1D CNN (`TinyECG`) as the test workload.

---

## 0. Environment and Dependencies

Tested on:

- Windows 11
- Intel Core i7-11375H (8 hardware threads used)
- NVIDIA RTX 3060 Laptop GPU (6 GB)
- Anaconda Python 3.9
- PyTorch with CUDA 11/12

### Create and activate the conda environment

```bash
conda create -n hpc-part1 python=3.9 -y
conda activate hpc-part1

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib mpi4py wfdb

### Prep shards

conda activate hpc-part1
python shard_prep.py --dataset mitbih --win_len 500 --stride 250 --shard_size 32768

### Benchmark locality

python bench_locality.py

### Benchmark OpenMP & Kernel

cd path\to\HPCPROJECT\PART-2\src
cl /O2 /openmp /arch:AVX2 /LD conv1d_openmp_simd.c /Fe:conv1d.dll

python benchmark_part_2.py

### MPI & CUDA (FOR PSEUDO-FEDERATED CONFIG)

conda activate hpc-part1

# Example: 2 simulated clients, batch 256, 200 steps
mpiexec -n 2 python -m src.part3_mpi_gpu_train --batch-size 256 --steps 200 --max-windows 20000

# activate your hpc-part1 conda env first
mpiexec -n 2 python -m src.part3_mpi_gpu_train --batch-size 256 --steps 200


python -m src.plot_part3_results

### SEE OTHER README FOR INSTRUCTIONS ON CLUSTER DEPLOYMENT
