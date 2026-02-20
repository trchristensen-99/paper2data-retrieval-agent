#!/bin/bash
#SBATCH --job-name=al_baseline
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=14
#SBATCH --mem-per-gpu=128G
#SBATCH --partition=kooq
#SBATCH --qos=koolab
#SBATCH --time=48:00:00

# SLURM job script for running baseline subset experiments on CSHL HPC
#
# Usage:
#   sbatch scripts/slurm/baseline.sh [dataset]
#
# Examples:
#   sbatch scripts/slurm/baseline.sh k562
#   sbatch scripts/slurm/baseline.sh yeast

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Get dataset from command line argument, default to k562
DATASET=${1:-k562}
echo "Dataset: $DATASET"

# Load modules (adjust as needed for your HPC environment)
# module load cuda/11.8
# module load python/3.11

# Navigate to project directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Activate virtual environment
source .venv/bin/activate
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Create log directory if it doesn't exist
mkdir -p logs

# Print GPU info
nvidia-smi
echo "=========================================="

# Run experiment
echo "Starting baseline experiment..."
echo "Config: experiments/configs/baseline.yaml"
echo "Dataset: $DATASET"
echo ""

python experiments/01_baseline_subsets.py \
    --config experiments/configs/baseline.yaml \
    --dataset $DATASET

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
