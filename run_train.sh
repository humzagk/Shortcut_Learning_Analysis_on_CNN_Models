# slurm_scripts/run_train.sh


#!/bin/bash
#SBATCH -J cnn_exp_full
#SBATCH -p akya-cuda         # Primary GPU partition
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10                # CPU cores for data loading
#SBATCH --gres=gpu:1         # Request 1 GPU
#SBATCH --time=24:00:00      # 24 hours to cover all experiments
#SBATCH --output=/arf/scratch/%u/shortcut_data/results/train_%j.log

echo "Job started on $(hostname)"

# 1. Load Environment
module purge
export PATH=$HOME/.local/bin:$PATH

# 2. Define Variables
USER_NAME=$(whoami)
# Data and Results in Scratch
DATA_ROOT="/arf/scratch/$USER_NAME/shortcut_data"
OUTPUT_DIR="/arf/scratch/$USER_NAME/shortcut_data/results"
SCRIPT_PATH="$HOME/CNN_Shortcut_Project/src/train.py"

mkdir -p $OUTPUT_DIR

# Define Lists
MODELS=("resnet18" "resnet101")
MODIFICATIONS=("edges" "segmentation" "grayscale" "occlusion")

# MAIN EXPERIMENT LOOP

for MODEL in "${MODELS[@]}"; do
    echo "STARTING BATCH FOR MODEL: $MODEL"



    # BLOCK A: BASELINE & EXPERIMENT 1 (Bias Check)
    # Goal: Train on Original, Test on Everything (Original + Modified)

    echo "[Exp 1] Training on ORIGINAL Data "

    # 1. Baseline: Train Original -> Test Original
    echo "Running: Train Original -> Test Original"
    python $SCRIPT_PATH \
        --data_root $DATA_ROOT \
        --output_dir $OUTPUT_DIR \
        --model $MODEL \
        --epochs 15 \
        --train_mode original \
        --test_mode original

    # 2. Bias Checks: Train Original -> Test Modified
    for MOD in "${MODIFICATIONS[@]}"; do
        echo "Running: Train Original -> Test $MOD"
        python $SCRIPT_PATH \
            --data_root $DATA_ROOT \
            --output_dir $OUTPUT_DIR \
            --model $MODEL \
            --epochs 15 \
            --train_mode original \
            --test_mode $MOD
    done


    # BLOCK B: EXPERIMENT 2 & 3 (Shape Adaptation & Domain Mastery)
    # Goal: Train on Modified, Test on Original & Same Modified

    echo "[Exp 2 & 3] Training on MODIFIED Data "

    for MOD in "${MODIFICATIONS[@]}"; do

        # Exp 2: Train Modified -> Test Original (Does the model learn shape?)
        echo "Running: Train $MOD -> Test Original"
        python $SCRIPT_PATH \
            --data_root $DATA_ROOT \
            --output_dir $OUTPUT_DIR \
            --model $MODEL \
            --epochs 15 \
            --train_mode $MOD \
            --test_mode original

        # Exp 3: Train Modified -> Test Modified (Theoretical limit)
        echo "Running: Train $MOD -> Test $MOD"
        python $SCRIPT_PATH \
            --data_root $DATA_ROOT \
            --output_dir $OUTPUT_DIR \
            --model $MODEL \
            --epochs 15 \
            --train_mode $MOD \
            --test_mode $MOD
    done

done

echo "All training experiments finished."