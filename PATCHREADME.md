# Local patches / setup notes

Not upstream. RTX 5070 Ti (Blackwell `sm_120`) on native Linux.

## New PC (after `./setup.sh`)

Upstream pins `torch==2.3.1` (CUDA 12.1, max `sm_90`). Replace with CUDA 12.8 wheels. Driver is enough; no system CUDA toolkit.

```bash
conda activate vggt-slam
pip uninstall -y torch torchvision torchaudio triton
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

- Expect `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`, arch list includes `sm_120`
- Do not re-run `./setup.sh` / `pip install -r requirements.txt` or torch rolls back to `2.3.1`

## Code patches (already in this tree)

- `third_party/sam3/sam3/model/vitdet.py` — SAM3 fused MLP mixed `bfloat16` activations with `float32` `fc2` weights
- `third_party/sam3/sam3/model/sam3_image_processor.py` — `bfloat16` autocast for SAM3; cast scores/boxes/mask logits to `float32` (numpy cannot convert `bfloat16`)
- `vggt_slam/slam_utils.py` — `overlay_masks` converts masks to `float32` before `.numpy()`

## Apartment + `--run_os`

Point at `apartment/images/`, not `apartment/`. Viser: `http://localhost:8080`.

```bash
conda activate vggt-slam
cd /path/to/VGGT-SLAM
python3 main.py --image_folder apartment/images --max_loops 1 --vis_map --run_os
```

## Live webcam

Upstream realtime demo is RealSense-only. This tree adds `--camera webcam` (OpenCV / `/dev/video*`).

```bash
conda activate vggt-slam
cd /path/to/VGGT-SLAM
python3 main_realtime.py --camera webcam --vis_map --submap_size 8 --max_loops 0
```

- Refresh `http://localhost:8080` after each new process (old tab stays on the previous session).
- Watch the OpenCV window: `KFs: x/9`. A new submap starts only after enough camera motion. Holding still = no new map.
- `--max_loops 0` first. Indoor webcam views look similar; false loop closures warp the map.
- `--vis_voxel_size 0.01` is the realtime default so viser does not freeze on dense clouds.
- `--min_disparity 25` if keyframes are too sparse. `--camera_id 1` if the wrong camera opens.
- Open-set: add `--run_os` (no OpenCV preview). RealSense: omit `--camera webcam`.
