#!/bin/bash
#SBATCH --job-name=fdtd_mpi_pb2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=fdtd_mpi_pb2%j.out

PROJECT_DIR="${HOME}/project_info0939/project_info0939"
EXEC_DIR="${PROJECT_DIR}/bin"
RUN_DIR="$SCRATCH/INFO0939/question_4/$SLURM_JOB_ID"

module load OpenMPI 

mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

CFL_VALUES=(0.30 0.50 0.80 0.90 0.95 0.98 1.00 1.02 1.05)

for S in "${CFL_VALUES[@]}"; do
    TAG="probleme2_cfl${S//./p}"        # ex: probleme1_cfl0p50
    echo "=== Running problem 2 with CFL=$S ===" | tee -a index.txt

    CFL=$S srun "$EXEC_DIR/fdtd" 2 > "${TAG}.out"

    echo "${TAG}.out" >> index.txt
done
