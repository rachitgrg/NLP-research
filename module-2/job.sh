#!/bin/bash
# job.sh — SLURM job script for HPC supercomputing lab
# ─────────────────────────────────────────────────────────────
# Usage:
#   sbatch job.sh
#
# Before submitting:
#   1. Fill in the SLURM directives below (partition, gres, email)
#   2. Set CONDA_ENV_NAME to your conda environment name
#   3. Upload this entire module-2/ folder to the HPC
#   4. Run: sbatch job.sh
#
# To check job status:
#   squeue -u $USER
#   tail -f logs/train_%j.out     # live log
# ─────────────────────────────────────────────────────────────

#SBATCH --job-name=yolo_finetune
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=04:00:00           # 4hr walltime (plenty for A100; increase to 12:00:00 for V100)

# ── GPU resource request ─────────────────────────────────────
# Uncomment ONE of the lines below based on your college HPC:

#SBATCH --gres=gpu:a100:1         # 1× A100  (use batch=32, fastest)
##SBATCH --gres=gpu:v100:1        # 1× V100  (use batch=16)
##SBATCH --gres=gpu:1             # any GPU  (generic, ask lab admin which is assigned)

# ── CPU and memory ───────────────────────────────────────────
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                 # 32GB RAM for DataLoader workers

# ── Partition ────────────────────────────────────────────────
# Replace 'gpu' with whatever your HPC's GPU partition is called.
# Common names: gpu, gpuq, compute-gpu, short-gpu
#SBATCH --partition=gpu

# ── Notifications ────────────────────────────────────────────
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com  # ← CHANGE THIS

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — set these before submitting
# ─────────────────────────────────────────────────────────────

# Your conda environment name (must have ultralytics, pandas, scikit-learn)
CONDA_ENV_NAME="yolo_env"

# The module-2 directory on the HPC filesystem
# (upload the full NLP-Research folder to your HPC home directory)
MODULE2_DIR="$HOME/NLP-Research/module-2"

# Batch size: scale up vs local 3050
#   A100 (40GB): 32
#   V100 (32GB): 16
#   V100 (16GB): 8
HPC_BATCH=32

# Epochs: same as local
HPC_EPOCHS=30

# Model to fine-tune
HPC_MODEL="yolov8n.pt"

# If A100: you can upgrade to yolov8s.pt here for better mAP at no extra cost
# HPC_MODEL="yolov8s.pt"

# ─────────────────────────────────────────────────────────────
# RUNTIME
# ─────────────────────────────────────────────────────────────

echo "======================================================="
echo " SLURM YOLOv8 Fine-Tuning Job"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $SLURMD_NODENAME"
echo " Started  : $(date)"
echo "======================================================="

# Create logs directory if it doesn't exist
mkdir -p "$MODULE2_DIR/logs"

# Print GPU info
nvidia-smi
echo ""

# Load conda (paths vary by HPC — ask lab admin if this fails)
# Try common module names:
module load anaconda 2>/dev/null || \
module load miniconda 2>/dev/null || \
module load conda 2>/dev/null || \
source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null

# Activate the conda environment
conda activate "$CONDA_ENV_NAME"

if [ $? -ne 0 ]; then
    echo "[ERROR] Could not activate conda environment: $CONDA_ENV_NAME"
    echo "Create it with:"
    echo "  conda create -n $CONDA_ENV_NAME python=3.10"
    echo "  conda activate $CONDA_ENV_NAME"
    echo "  pip install ultralytics pandas scikit-learn tqdm"
    exit 1
fi

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check ultralytics is installed
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')" || {
    echo "[ERROR] ultralytics not installed in $CONDA_ENV_NAME"
    echo "Run: pip install ultralytics"
    exit 1
}

# Move to module-2 directory
cd "$MODULE2_DIR" || { echo "[ERROR] Cannot cd to $MODULE2_DIR"; exit 1; }

echo "Working directory: $(pwd)"
echo ""
echo "Starting training..."
echo ""

# Run training
# --device 0         : use GPU 0 (first GPU assigned by SLURM)
# --batch $HPC_BATCH : larger batch for HPC GPU
# --workers 8        : more DataLoader workers (matches --cpus-per-task)
python train.py \
    --model "$HPC_MODEL" \
    --epochs "$HPC_EPOCHS" \
    --batch "$HPC_BATCH" \
    --imgsz 640 \
    --device 0 \
    --workers 8

EXIT_CODE=$?

echo ""
echo "======================================================="
echo " Job finished at: $(date)"
echo " Exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo " ✓ Training SUCCESS"
    echo " Best weights: $MODULE2_DIR/runs/train/weights/best.pt"
    echo " Summary     : $MODULE2_DIR/results_summary.txt"
else
    echo " ✗ Training FAILED — check logs/train_${SLURM_JOB_ID}.err"
fi
echo "======================================================="

exit $EXIT_CODE
