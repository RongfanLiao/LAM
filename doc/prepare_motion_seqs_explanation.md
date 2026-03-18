# `prepare_motion_seqs` Function Explanation

## Location

- **Definition**: [head_utils.py:329-422](lam/runners/infer/head_utils.py#L329-L422)
- **Call site**: [inference.py:175-182](inference.py#L175-L182)

## Call Signature (at the call site)

```python
motion_seq = prepare_motion_seqs(
    flame_params_dir, None, save_root=tmpdir, fps=render_fps,
    bg_color=1., aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
    render_image_res=render_size, multiply=16,
    need_mask=motion_img_need_mask, vis_motion=vis_motion,
    shape_param=shape_param, test_sample=False, cross_id=False,
    src_driven=src_driven,
)

# Override betas with the source identity
motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
```

## Purpose

This function **loads and assembles a complete motion sequence** from pre-tracked FLAME parameters. It reads per-frame FLAME params (expression, jaw pose, eye gaze, head rotation, translation), camera poses, and intrinsics from the driving motion directory, then packages them into a single dictionary ready for the LAM model to render animated frames.

---

## Arguments Explained

| Argument | Value at call site | Meaning |
|----------|-------------------|---------|
| `motion_seqs_dir` | `flame_params_dir` | Path to the directory containing per-frame `.npz` FLAME param files (e.g., `assets/sample_motion/export/Anti_Drugs/flame_param/`) |
| `image_folder` | `None` | Optional path to driving video frames — `None` because we only need FLAME params, not source images |
| `save_root` | `tmpdir` | Temporary directory for intermediate files (used if motion needs to be predicted from images) |
| `fps` | `render_fps` (30) | Target frame rate for the output video |
| `bg_color` | `1.0` | Background color (white) for rendering |
| `aspect_standard` | `1.0` | Target aspect ratio (square) |
| `enlarge_ratio` | `[1.0, 1.0]` | No random crop enlargement (no augmentation at inference) |
| `render_image_res` | `render_size` | Output rendering resolution from config |
| `multiply` | `16` | Snap resolution to multiples of 16 (for the rendering network) |
| `need_mask` | `motion_img_need_mask` (typically `False`) | Whether to load masks for driving frames |
| `vis_motion` | `vis_motion` (typically `False`) | Whether to render FLAME mesh wireframes for visualization |
| `shape_param` | `shape_param` | Source identity's FLAME shape coefficients (300-dim tensor from `preprocess_image`) |
| `test_sample` | `False` | If `True`, subsamples to 50 frames for quick testing |
| `cross_id` | `False` | Flag for cross-identity motion transfer (unused in this code path) |
| `src_driven` | `[src, driven]` | Names of source and driving identities for logging/bookkeeping |

---

## Step-by-Step Breakdown

### 1. Resolve Motion Source (lines 333-335)

```python
if motion_seqs_dir is None:
    assert image_folder is not None
    motion_seqs_dir, image_folder = predict_motion_seqs_from_images(image_folder, save_root, fps)
```

If no pre-tracked FLAME params directory is provided, it runs FLAME tracking on raw video frames using multi-HMR. **At the call site, `motion_seqs_dir` is provided, so this is skipped.**

### 2. Resolve Shape Parameters (lines 341-346)

```python
if shape_param is None:
    cor_flame_path = os.path.join(os.path.dirname(motion_seqs_dir), 'canonical_flame_param.npz')
    flame_p = np.load(cor_flame_path)
    shape_param = torch.FloatTensor(flame_p['shape'])
```

If `shape_param` is not passed in, loads it from the **driving** identity's canonical FLAME params. **At the call site, `shape_param` is provided (from the source identity), so the driving identity's shape is NOT used.** This is critical for cross-identity transfer — the source person's face shape is preserved.

### 3. Load `transforms.json` (lines 348-354)

```python
transforms_json = os.path.join(os.path.dirname(motion_seqs_dir), "transforms.json")
with open(transforms_json) as fp:
    data = json.load(fp)
all_frames = data["frames"]
all_frames = sorted(all_frames, key=lambda x: x["flame_param_path"])
```

Reads the motion sequence manifest file. Each frame entry in `transforms.json` contains:

```json
{
    "cx": 512.0, "cy": 512.0,             // principal point
    "fl_x": 2604.94, "fl_y": 2604.94,     // focal lengths
    "h": 1024, "w": 1024,                 // image dimensions
    "transform_matrix": [[...], ...],       // 4×4 camera-to-world matrix
    "flame_param_path": "flame_param/00000.npz",  // per-frame FLAME params
    "file_path": "images/00000_00.png",    // corresponding image
    "fg_mask_path": "fg_masks/00000_00.png" // foreground mask
}
```

Frames are sorted by `flame_param_path` to ensure chronological order.

### 4. Optional Test Subsampling (lines 355-359)

```python
if test_sample:
    sample_num = 50
    frame_ids = frame_ids[np.linspace(0, frame_ids.shape[0]-1, sample_num).astype(np.int32)]
```

Uniformly subsamples to 50 frames for quick testing. **Skipped at inference (`test_sample=False`).**

### 5. Load Optional Teeth Blendshapes (lines 361-365)

```python
teeth_bs_pth = os.path.join(os.path.dirname(motion_seqs_dir), "tracked_teeth_bs.npz")
if os.path.exists(teeth_bs_pth):
    teeth_bs_lst = np.load(teeth_bs_pth)['expr_teeth']
else:
    teeth_bs_lst = None
```

If available, loads per-frame teeth blendshape weights for more realistic mouth interior rendering. Not all motion sequences have this file.

### 6. Per-Frame Loading Loop (lines 367-386)

```python
for idx, frame_id in enumerate(frame_ids):
    frame_info = all_frames[frame_id]
    flame_path = os.path.join(os.path.dirname(motion_seqs_dir), frame_info["flame_param_path"])

    teeth_bs = teeth_bs_lst[frame_id] if (...) else None
    flame_param = load_flame_params(flame_path, teeth_bs)

    c2w, intrinsic = _load_pose(frame_info)
    intrinsic = scale_intrs(intrinsic, 0.5, 0.5)

    c2ws.append(c2w)
    bg_colors.append(bg_color)
    intrs.append(intrinsic)
    flame_params.append(flame_param)
```

For each frame:

#### 6a. `load_flame_params` ([head_utils.py:314-326](lam/runners/infer/head_utils.py#L314-L326))

Reads one `.npz` file and extracts per-frame FLAME parameters as tensors:

| Key | Shape | Description |
|-----|-------|-------------|
| `expr` | `(100,)` | Expression blendshape coefficients |
| `rotation` | `(3,)` | Global head rotation (axis-angle) |
| `neck_pose` | `(3,)` | Neck joint rotation |
| `jaw_pose` | `(3,)` | Jaw opening rotation |
| `eyes_pose` | `(6,)` | Left + right eye gaze rotation |
| `translation` | `(3,)` | Global head translation |
| `teeth_bs` | `(4,)` | Teeth blendshape weights (optional) |

Each uses index `[0]` because the `.npz` stores shape `(1, N)` — this squeezes to `(N,)`.

#### 6b. `_load_pose` ([head_utils.py:298-311](lam/runners/infer/head_utils.py#L298-L311))

Extracts camera parameters from the frame's JSON entry:

- **c2w** (camera-to-world): The 4×4 transform matrix from `transforms.json`. The Y and Z axes are flipped (`c2w[:3, 1:3] *= -1`) to convert from the NeRF/OpenGL convention (Y-up, Z-backward) to the OpenCV convention (Y-down, Z-forward).
- **intrinsic**: A 4×4 matrix with focal lengths (`fl_x`, `fl_y`) and principal point (`cx`, `cy`).

#### 6c. `scale_intrs(intrinsic, 0.5, 0.5)`

Scales the intrinsics by 0.5× — the original tracking was done at full resolution (e.g., 1024×1024), but rendering happens at half resolution (e.g., 512×512). This halves focal lengths and principal point coordinates.

### 7. Stack into Tensors (lines 388-398)

```python
c2ws = torch.stack(c2ws, dim=0)         # [N, 4, 4]
intrs = torch.stack(intrs, dim=0)       # [N, 4, 4]
bg_colors = torch.tensor(bg_colors, ...).unsqueeze(-1).repeat(1, 3)  # [N, 3]

# Merge per-frame flame_params dicts into {key: [N, ...]} tensors
flame_params_tmp = defaultdict(list)
for flame in flame_params:
    for k, v in flame.items():
        flame_params_tmp[k].append(v)
for k, v in flame_params_tmp.items():
    flame_params_tmp[k] = torch.stack(v)
flame_params = flame_params_tmp
```

Converts the list of per-frame dictionaries into a single dictionary of stacked tensors:

| Key | Shape | Description |
|-----|-------|-------------|
| `expr` | `[N, 100]` | Expression coefficients for all frames |
| `rotation` | `[N, 3]` | Head rotations for all frames |
| `neck_pose` | `[N, 3]` | Neck poses |
| `jaw_pose` | `[N, 3]` | Jaw poses |
| `eyes_pose` | `[N, 6]` | Eye gaze |
| `translation` | `[N, 3]` | Head translations |

### 8. Set Identity Shape (line 400)

```python
flame_params["betas"] = shape_param
```

Sets the identity shape coefficients. Note: this is the **source identity's** shape (passed in from `preprocess_image`), NOT the driving person's shape. This ensures the rendered mesh has the source person's face geometry.

### 9. Optional Motion Visualization (lines 402-405)

```python
if vis_motion:
    motion_render = render_flame_mesh(flame_params, intrs, c2ws)
else:
    motion_render = None
```

If `vis_motion=True`, renders the FLAME mesh wireframe for each frame — useful for debugging to see the driving motion as a mesh overlay. Typically `False` at inference.

### 10. Add Batch Dimension (lines 407-413)

```python
for k, v in flame_params.items():
    flame_params[k] = v.unsqueeze(0)    # [N, ...] → [1, N, ...]
c2ws = c2ws.unsqueeze(0)                # [N, 4, 4] → [1, N, 4, 4]
intrs = intrs.unsqueeze(0)              # [N, 4, 4] → [1, N, 4, 4]
bg_colors = bg_colors.unsqueeze(0)      # [N, 3] → [1, N, 3]
```

Adds a batch dimension (B=1) to all tensors for compatibility with the model's batched interface.

### 11. Package and Return (lines 415-422)

```python
motion_seqs = {
    "render_c2ws":        c2ws,           # [1, N, 4, 4] - camera poses
    "render_intrs":       intrs,          # [1, N, 4, 4] - camera intrinsics
    "render_bg_colors":   bg_colors,      # [1, N, 3]    - background colors
    "flame_params":       flame_params,   # dict of [1, N, ...] tensors
    "vis_motion_render":  motion_render,  # numpy array or None
}
return motion_seqs
```

---

## The `betas` Override Statement

```python
motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
```

After `prepare_motion_seqs` returns, this line at [inference.py:185](inference.py#L185) **explicitly overrides** the betas with the source identity's shape, adding a batch dimension: `(300,)` → `(1, 300)`.

### Why is this necessary?

Inside `prepare_motion_seqs`, betas are set at line 400 as:

```python
flame_params["betas"] = shape_param      # shape: (300,)
```

But then at line 408-409, the batch dimension is added:

```python
for k, v in flame_params.items():
    flame_params[k] = v.unsqueeze(0)     # becomes (1, 300)
```

So after the function returns, `flame_params["betas"]` is already `(1, 300)`. The override at line 185 does `shape_param.unsqueeze(0)` which also produces `(1, 300)` — **the same value and shape**.

This override is **redundant but intentional as a safety guarantee**. It ensures that regardless of any internal logic in `prepare_motion_seqs` (e.g., if `shape_param` were `None` and the function fell back to the driving identity's shape), the source identity's betas are always used. The LAM model expects `betas` shape `[B, 300]` (validated at [modeling_lam.py:258](lam/models/modeling_lam.py#L258): `assert len(flame_params["betas"].shape) == 2`).

### The bigger picture

```
Source Image ──→ FLAME Tracking ──→ shape_param (300,) ──→ betas
                                         │
                                    "WHO the face belongs to"
                                         │
Driving Motion ──→ FLAME params ──→ expr, rotation, jaw, eyes, neck, translation
                                         │
                                    "HOW the face moves"
```

The `betas` override enforces a clean separation:
- **Identity** (shape/structure) comes from the **source image**
- **Motion** (expression/pose) comes from the **driving sequence**

This is the core of LAM's cross-identity motion transfer — the source person's 3D face shape is preserved while being animated by another person's facial performance.

---

## Output Structure Summary

The returned `motion_seq` dictionary provides everything the LAM model needs to render N animated frames:

```
motion_seq
├── "render_c2ws"       [1, N, 4, 4]   Camera-to-world matrices (viewpoints)
├── "render_intrs"      [1, N, 4, 4]   Camera intrinsics (focal length, principal point)
├── "render_bg_colors"  [1, N, 3]      Per-frame background color (white)
├── "flame_params"                      FLAME parameter dictionary:
│   ├── "betas"         [1, 300]        Identity shape (SOURCE person) — shared across all frames
│   ├── "expr"          [1, N, 100]     Per-frame expression blendshapes
│   ├── "rotation"      [1, N, 3]       Per-frame global head rotation
│   ├── "neck_pose"     [1, N, 3]       Per-frame neck rotation
│   ├── "jaw_pose"      [1, N, 3]       Per-frame jaw opening
│   ├── "eyes_pose"     [1, N, 6]       Per-frame eye gaze
│   └── "translation"   [1, N, 3]       Per-frame head translation
└── "vis_motion_render" [N, H, W, 3]   Optional FLAME mesh renders (or None)
```

Note that `betas` is `[1, 300]` (shared across all frames) while all other FLAME params are `[1, N, ...]` (per-frame). This reflects the fact that identity is static but motion varies frame by frame.
