"""Live webcam SAM3 mask test (not wired into SLAM).

Shows the raw frame with a red overlay, and the same frame with masked
pixels blacked out. Uses matplotlib because OpenCV windows hang after SAM3 loads.

Example:
  python3 sam3_webcam_mask.py
  python3 sam3_webcam_mask.py --prompt "person's hands and whatever they are holding"
  python3 sam3_webcam_mask.py --prompt hand --camera_id 0 --every 2
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from vggt_slam.cameras import WebcamCamera


def masks_to_union(masks: torch.Tensor, height: int, width: int) -> np.ndarray:
    """Union of SAM3 instance masks -> bool array (H, W)."""
    if masks is None or masks.numel() == 0 or masks.shape[0] == 0:
        return np.zeros((height, width), dtype=bool)
    m = masks.detach()
    if m.ndim == 4:
        m = m[:, 0]
    union = m.any(dim=0).float().cpu().numpy() > 0.5
    if union.shape != (height, width):
        union = np.array(
            Image.fromarray(union.astype(np.uint8) * 255).resize((width, height), Image.NEAREST)
        ).astype(bool)
    return union


def compose_views(rgb: np.ndarray, union: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    overlay[union] = (0.55 * overlay[union] + np.array([255, 0, 0]) * 0.45).astype(np.uint8)
    remaining = rgb.copy()
    remaining[union] = 0
    return np.concatenate([overlay, remaining], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Live SAM3 human/hand mask preview")
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--prompt", type=str, default="person",
                        help='SAM3 text prompt, e.g. "person", "hand", "person\'s hands and whatever they are holding"')
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=1008,
                        help="SAM3 internal resolution. Must stay 1008 (model RoPE / patch size).")
    parser.add_argument("--every", type=int, default=1, help="Run SAM3 every N frames; reuse last mask in between")
    args = parser.parse_args()
    if args.resolution != 1008:
        print(f"Warning: SAM3 backbone is built for 1008x1008; overriding --resolution {args.resolution} -> 1008")
        args.resolution = 1008

    print("Loading SAM3 (first time downloads weights)...")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_model = build_sam3_image_model()
    processor = Sam3Processor(
        sam3_model, resolution=args.resolution, confidence_threshold=args.confidence
    )
    print(f"SAM3 ready. Prompt: {args.prompt!r}")

    camera = WebcamCamera(
        device=args.camera_id,
        width=args.camera_width,
        height=args.camera_height,
    )
    camera.start()
    print("Waiting for camera...")
    frame = None
    while frame is None:
        frame = camera.capture()
    print("Close the matplotlib window or Ctrl+C to quit.")

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_axis_off()
    union = np.zeros(frame.shape[:2], dtype=bool)
    im = None
    frame_i = 0
    t0 = time.time()

    try:
        while plt.fignum_exists(fig.number):
            bgr = camera.capture()
            if bgr is None:
                continue
            rgb = bgr[:, :, ::-1].copy()
            h, w = rgb.shape[:2]
            frame_i += 1

            if (frame_i - 1) % max(args.every, 1) == 0:
                pil = Image.fromarray(rgb)
                t1 = time.time()
                with torch.no_grad():
                    state = processor.set_image(pil)
                    out = processor.set_text_prompt(state=state, prompt=args.prompt)
                dt = time.time() - t1
                union = masks_to_union(out["masks"], h, w)
                n = int(out["masks"].shape[0]) if out["masks"] is not None else 0
                elapsed = time.time() - t0
                fps = frame_i / max(elapsed, 1e-6)
                print(f"frame={frame_i}  instances={n}  sam3={dt:.2f}s  display_fps={fps:.1f}")

            vis = compose_views(rgb, union)
            title = f"left: {args.prompt!r} overlay   right: masked out"
            if im is None:
                im = ax.imshow(vis)
                ax.set_title(title)
                fig.tight_layout()
            else:
                im.set_data(vis)
                ax.set_title(title)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        camera.stop()
        plt.close("all")


if __name__ == "__main__":
    main()
