# LAM CLI Inference

`inference.py` provides a command-line interface for running LAM inference without Gradio. It is designed for headless environments such as HPC clusters where a GUI is not available.

## Prerequisites

- Python 3.10 with the `lam` conda/micromamba environment activated
- PyTorch 2.3.0 + CUDA 12.1
- All dependencies installed (`pip install -r requirements.txt --no-build-isolation`)
- PyTorch3D installed (build from source: `pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation`)
- Model weights downloaded to `model_zoo/`
- FaceBoxesV2 compiled (`cd external/landmark_detection/FaceBoxesV2/utils && sh make.sh`)

## Quick Start

```bash
python inference.py --image assets/sample_input/messi.png --motion Anti_Drugs
```

Output is saved to `output/videos/messi_Anti_Drugs.mp4` by default.

## Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--image` | Yes | - | Path to input face image |
| `--motion` | Yes | - | Motion name or path (see below) |
| `--output` | No | `output/videos/<image>_<motion>.mp4` | Output video path |
| `--model_name` | No | `./model_zoo/lam_models/releases/lam/lam-20k/step_045500/` | Path to model checkpoint directory |
| `--infer_config` | No | `./configs/inference/lam-20k-8gpu.yaml` | Inference config YAML |

## Specifying Motion

The `--motion` argument accepts either:

1. **A motion name** from `assets/sample_motion/export/`:

   ```bash
   python inference.py --image input.png --motion Anti_Drugs
   ```

2. **A direct path** to a motion directory containing a `flame_param/` subdirectory:

   ```bash
   python inference.py --image input.png --motion assets/sample_motion/export/2966_right_whiteBg_staticOffset_maskBelowLine
   ```

## Available Sample Motions

| Motion Name |
|---|
| Anti_Drugs |
| D_ANgelo_Dinero |
| Donald_Trump |
| GEM |
| I_Am_Iron_Man |
| Joe_Biden |
| Look_In_My_Eyes |
| Michael_Wayne_Rosen |
| Pen_Pineapple_Apple_Pen |
| Speeding_Scandal |
| Taylor_Swift |
| The_Shawshank_Redemption |

## Available Sample Input Images

Located in `assets/sample_input/`:

`barbara.jpg`, `cluo.jpg`, `dufu.jpg`, `james.png`, `libai.jpg`, `messi.png`, `musk.jpg`, `pop.png`, `speed.jpg`, `status.png`, `zhouxingchi.jpg`

## Examples

```bash
# Basic usage with default output path
python inference.py --image assets/sample_input/messi.png --motion Anti_Drugs

# Custom output path
python inference.py --image assets/sample_input/messi.png --motion Anti_Drugs --output output/messi_test.mp4

# Using a direct path to a custom motion directory
python inference.py --image assets/sample_input/messi.png --motion /path/to/custom_motion_dir

# Custom model checkpoint
python inference.py --image assets/sample_input/messi.png --motion Taylor_Swift \
    --model_name ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
```

## Pipeline Overview

The script runs the following steps:

1. **Load config and model** - Loads the LAM model from the checkpoint
2. **Initialize FLAME tracking** - Loads face detection, landmark, and matting models
3. **Preprocess input image** - Face detection, cropping, segmentation, and landmark extraction
4. **FLAME tracking** - Optimizes FLAME parameters for the input face
5. **Prepare motion sequence** - Loads driving motion FLAME parameters
6. **LAM inference** - Generates animated Gaussian head frames
7. **Save video** - Composites frames and adds audio (if available from the motion source)

## Notes

- The driving motion must be pre-processed FLAME parameters (`.npz` files in a `flame_param/` directory). Raw MP4 videos cannot be used directly as driving motion.
- Audio from the motion source (e.g., `Anti_Drugs.wav`) is automatically added to the output video if present.
- Front-facing input images or face orientations close to the driving signal produce better results.
