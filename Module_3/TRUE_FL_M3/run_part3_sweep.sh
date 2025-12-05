#!/bin/bash
# ================================================================
# TRUEFL/run_part3_sweep.sh
#
# Run multiple world sizes and repeat several times.
# Must be run INSIDE an allocation (`salloc`) on the cluster.
#
# EDIT THESE PLACEHOLDERS:
#    <PROJECT_ROOT>   -> your project folder on cluster
#    <DATA_ROOT>      -> path to shared ecg_*.bin shards
# ================================================================

set -euo pipefail

PROJECT_ROOT="<PROJECT_ROOT>/TRUEFL"   # EDIT
DATA_ROOT="<DATA_ROOT>"                # EDIT

CONDA_ENV="hpc-ecg"

WORLD_SIZES=(1 2 4 8)
REPEATS=5

module load anaconda
module load cuda
module load openmpi

source activate "${CONDA_ENV}" || conda activate "${CONDA_ENV}"

cd "${PROJECT_ROOT}"

mkdir -p results

for W in "${WORLD_SIZES[@]}"; do
    echo "===== WORLD SIZE = ${W} ====="
    for ((i=1; i<=REPEATS; i++)); do
        echo "[${i}/${REPEATS}] Running…"

        srun \
          --ntasks="${W}" \
          --gpus-per-node=1 \
          --ntasks-per-node=1 \
          --cpus-per-task=4 \
          python part3_fedavg_overlap_mpi_gpu.py \
              --data-root "${DATA_ROOT}" \
              --batch-size 256 \
              --rounds 5 \
              --local-steps 50 \
              --config both \
              --max-windows 20000
    done
done

echo "[DONE] All sweeps finished."
