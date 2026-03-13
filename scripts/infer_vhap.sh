#!/bin/bash
# Inference script using VHAP motion format with pre-tracked source directory

TRACKED_DIR="output/tracking/export/000000"
MOTION_DIR="/home/r/rongfan/VHAP/tmp/test_run_2_lightweight"
MODEL_NAME="model_zoo/lam_models/releases/lam/lam-20k/step_045500/"
INFER_CONFIG="configs/inference/lam-20k-8gpu.yaml"

device=0

export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=$device python infer_vhap.py \
    --tracked_dir $TRACKED_DIR \
    --motion $MOTION_DIR \
    --model_name $MODEL_NAME \
    --infer_config $INFER_CONFIG
