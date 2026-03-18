# Driving LAM with Predicted FLAME Parameters

This document describes how to use predicted FLAME parameters from the `head_avatar` project to drive a source image in LAM.

## Overview

The pipeline has two stages:
1. **Convert** predicted FLAME params into LAM's motion format
2. **Run** LAM inference to generate the driven video

## Prerequisites

- Micromamba `lam` environment activated
- GCC >= 9 and CUDA available (needed for nvdiffrast JIT compilation during flame tracking)

```bash
module load GCC/11.2.0 CUDA/12.0.0
# Avoid CUDA library conflicts with pip-installed nvidia packages
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v '/sulis/easybuild/software/CUDA' | tr '\n' ':')
```

## Step 1: Convert Predicted FLAME Params to LAM Motion Format

LAM expects motion data in a specific directory structure:

```
assets/sample_motion/export/<name>/
├── transforms.json              # camera params + frame list
├── canonical_flame_param.npz    # canonical (neutral) FLAME params
└── flame_param/
    ├── 00000.npz                # per-frame FLAME params
    ├── 00001.npz
    └── ...
```

The predicted npz from `head_avatar` contains only animation parameters (`expr`, `jaw_pose`, `rotation`, `neck_pose`, `eyes_pose`). The conversion script supplements these with identity/camera parameters (`translation`, `shape`, `static_offset`, `focal_length`) from the tracked FLAME params.

### Input Files

| File | Source | Contents |
|------|--------|----------|
| `2920_flame_pred.npz` | `head_avatar` prediction | expr (N,100), jaw_pose (N,3), rotation (N,3), neck_pose (N,3), eyes_pose (N,6) |
| `tracked_flame_params_30.npz` | `head_avatar` tracking | translation (N,3), shape (300,), static_offset (1,5143,3), focal_length (1,), image_size (2,) |

### Run Conversion

```bash
python tools/convert_pred_to_motion.py \
    --pred_npz ../head_avatar/output/predicted/2920_flame_pred.npz \
    --tracked_npz ../head_avatar/output/monocular/2920_right_lmkOnly/2026-03-07_19-33-03/tracked_flame_params_30.npz \
    --output_dir assets/sample_motion/export/2920_flame_pred
```

The script performs the same FLAME relocation as LAM's `TrackedFLAMEDatasetWriter`:
- Subtracts mean translation to center the head at the origin
- Places the camera at (0, 0, 1) in world coordinates
- Applies the relocation matrix to the camera pose

## Step 2: Run LAM Inference

```bash
bash scripts/inference_2920.sh
```

This script runs `lam.launch infer.lam` which:

1. **Flame tracking on source image** (`assets/sample_input/000000.jpg`)
   - Detects face, crops to 1024x1024, applies matting
   - Optimizes FLAME parameters to fit the source face (gets identity shape)
   - Exports preprocessed image + mask + FLAME params to `tracking_output/export/000000/`

2. **Prepares motion sequence** from `assets/sample_motion/export/2920_flame_pred/`
   - Reads `transforms.json` for camera parameters
   - Reads per-frame FLAME params from `flame_param/`
   - Overrides shape (`betas`) with source identity's shape

3. **LAM inference** generates Gaussian splat avatar and renders driven video
   - Output: `exps/videos/lam/lam_20k/000000.mp4`

## Output

| File | Description |
|------|-------------|
| `exps/videos/lam/lam_20k/000000.mp4` | Driven video (176 frames, 30 FPS, ~5.87s) |

## Notes

- The audio attachment step will fail if no `.wav` file exists at `assets/sample_motion/export/2920_flame_pred/2920_flame_pred.wav`. The video (without audio) is still saved successfully.
- To use a different source image, change `IMAGE_INPUT` in the inference script.
- To use different predicted params, re-run `convert_pred_to_motion.py` with new `--pred_npz` and `--tracked_npz` paths.

## Adapting for Other Sequences

```bash
# Convert
python tools/convert_pred_to_motion.py \
    --pred_npz <path_to_predicted.npz> \
    --tracked_npz <path_to_tracked_flame_params.npz> \
    --output_dir assets/sample_motion/export/<sequence_name>

# Update inference script or run directly
CUDA_VISIBLE_DEVICES=0 python -m lam.launch infer.lam \
    --config configs/inference/lam-20k-8gpu.yaml \
    model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
    image_input=<source_image> \
    export_video=true export_mesh=false \
    motion_seqs_dir=assets/sample_motion/export/<sequence_name>/ \
    motion_img_dir=null render_fps=30 motion_video_read_fps=30 \
    vis_motion=false motion_img_need_mask=true \
    save_ply=false save_img=false cross_id=false test_sample=false \
    gaga_track_type="" rank=0 nodes=0
```
