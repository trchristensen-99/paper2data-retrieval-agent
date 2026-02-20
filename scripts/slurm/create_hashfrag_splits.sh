#!/bin/bash
#SBATCH --job-name=hashfrag_k562
#SBATCH --output=logs/hashfrag_k562_%j.log
#SBATCH --error=logs/hashfrag_k562_%j.log
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=kooq
#SBATCH --qos=koolab

#################################################################
# HashFrag Split Creation for K562 Dataset
#
# This creates homology-aware train/validation/test splits
# using BLAST and Smith-Waterman alignment scores.
#
# Runtime: ~4-8 hours for full K562 dataset (~367K sequences)
# Memory: ~64GB peak (BLAST database + alignment)
# CPUs: 8 (BLAST can parallelize)
#
# Usage:
#   sbatch scripts/slurm/create_hashfrag_splits.sh
#
# Output:
#   - Splits saved to: data/k562/hashfrag_splits/
#   - Log file: logs/hashfrag_k562_{job_id}.log
#################################################################

echo "========================================"
echo "HashFrag Split Creation - K562 Dataset"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "========================================"
echo ""

# Setup environment (BLAST+ and HashFrag)
echo "Setting up environment..."
source setup_env.sh
echo ""

# Change to project directory
cd /home/trevor/al-genomics-benchmark || exit 1
echo "Working directory: $(pwd)"
echo ""

# Activate virtual environment (already done by setup_env.sh)
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Set BLAST threading (use available CPUs)
export BLAST_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "BLAST will use $BLAST_NUM_THREADS threads"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

echo "========================================"
echo "Starting HashFrag split creation..."
echo "========================================"
echo ""

# Run the split creation script
python scripts/create_hashfrag_splits.py \
    --data-dir ./data/k562 \
    --threshold 60

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ HashFrag split creation complete!"
    echo "========================================"
    echo "End time: $(date)"
    echo ""
    echo "Splits saved to: data/k562/hashfrag_splits/"
    echo ""
    echo "Next steps:"
    echo "  1. Run baseline experiments"
    echo "  2. Experiments will automatically use these splits"
    echo ""
else
    echo ""
    echo "========================================"
    echo "✗ HashFrag split creation FAILED"
    echo "========================================"
    echo "End time: $(date)"
    echo ""
    echo "Check the log file for details:"
    echo "  logs/hashfrag_k562_${SLURM_JOB_ID}.log"
    echo ""
    exit 1
fi
