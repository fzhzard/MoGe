import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

import json
from typing import 

import click
import numpy as np
import cv2
import torch
from tqdm import tqdm

from ultralytics import YOLO

from moge.model import import_model_class_by_version
from moge.utils.vis import colorize_depth
import utils3d


def get_rect_dimensions_3d(pts: np.ndarray) -> Tuple[float, float]:
    """
    Compute the dimensions (length, width) of the minimum area rectangle
    enclosing the 3D points projected onto their best-fit plane.
    """
    if pts.shape[0] < 3:
        return 0.0, 0.0
    
    # 1. Centering
    center = pts.mean(axis=0)
    pts_c = pts - center
    
    # 2. PCA via SVD on covariance matrix
    # pts_c: (N, 3)
    # cov: (3, 3)
    cov = np.cov(pts_c, rowvar=False)
    # Handle edge case where points are collinear or N is small
    if not np.all(np.isfinite(cov)):
        return 0.0, 0.0

    U, S, Vt = np.linalg.svd(cov)
    # U columns are principal components. 
    # U[:, 0] is the direction of max variance (length-ish)
    # U[:, 1] is the direction of 2nd max variance (width-ish)
    # U[:, 2] is the normal direction
    
    # 3. Project to 2D plane defined by first two principal components
    basis = U[:, :2]  # (3, 2)
    pts_2d = pts_c @ basis  # (N, 2)
    
    # 4. Min Area Rect
    # cv2.minAreaRect requires float32
    rect = cv2.minAreaRect(pts_2d.astype(np.float32))
    (center_2d, (w, h), angle) = rect
    
    return max(w, h), min(w, h)


@click.command(help="Fused inference: YOLOv8-seg + MoGe for PV distance/size")
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True), required=True,
              help='Input image or folder path. jpg/png supported.')
@click.option('--output', '-o', 'output_path', default='./output_fused', type=click.Path(),
              help='Output folder path')
@click.option('--yolo-weights', type=click.Path(exists=True),
              default='runs/segment/yolov8n-seg-pv3/weights/best.pt',
              help='Path to trained YOLOv8-seg weights.')
@click.option('--moge-pretrained', 'moge_pretrained', type=str, default=None,
              help='MoGe pretrained name or path (default: moge v2 normal).')
@click.option('--device', 'device_name', type=str, default='cuda',
              help='Device, e.g. "cuda", "cuda:0", "cpu".')
@click.option('--fov_x', 'fov_x_', type=float, default=None,
              help='Horizontal FOV in degrees if known, else MoGe estimates.')
@click.option('--resolution_level', type=int, default=9,
              help='MoGe resolution_level (token count).')
@click.option('--num_tokens', type=int, default=None,
              help='MoGe num_tokens; overrides resolution_level when set.')
@click.option('--use-fp16', 'use_fp16', is_flag=True,
              help='Use fp16 for MoGe inference.')
@click.option('--depth-threshold', type=float, default=0.04,
              help='Relative edge threshold when cleaning mesh (same as infer.py).')
@click.option('--save-vis', is_flag=True,
              help='Save visualization images (depth + YOLO masks).')
@click.option('--save-json', is_flag=True,
              help='Save per-panel distance/size info to JSON.')
def main(
    input_path: str,
    output_path: str,
    yolo_weights: str,
    moge_pretrained: str,
    device_name: str,
    fov_x_: float,
    resolution_level: int,
    num_tokens: int,
    use_fp16: bool,
    depth_threshold: float,
    save_vis: bool,
    save_json: bool,
):
    device = torch.device(device_name)

    # Collect images
    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    input_path = Path(input_path)
    if input_path.is_dir():
        image_paths = []
        for suf in include_suffices:
            image_paths.extend(input_path.rglob(f'*.{suf}'))
        image_paths = sorted(image_paths)
    else:
        image_paths = [input_path]

    if len(image_paths) == 0:
        raise FileNotFoundError(f'No image files found in {input_path}')

    # Load YOLO
    yolo_model = YOLO(str(yolo_weights))

    # Load MoGe
    if moge_pretrained is None:
        DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION = {
            "v2": "Ruicheng/moge-2-vits-normal",
        }
        moge_pretrained = DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION["v2"]

    moge_model = import_model_class_by_version('v2').from_pretrained(moge_pretrained).to(device).eval()
    if use_fp16:
        moge_model.half()

    output_root = Path(output_path)

    for img_path in tqdm(image_paths, desc='Fused inference'):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # MoGe inference
        img_tensor = torch.from_numpy(image_rgb / 255.0).float().permute(2, 0, 1).to(device)
        with torch.no_grad():
            out_moge = moge_model.infer(
                img_tensor,
                fov_x=fov_x_,
                resolution_level=resolution_level,
                num_tokens=num_tokens,
                use_fp16=use_fp16,
            )

        # MoGe outputs: depth [H, W], points [3, H, W], mask [H, W]
        depth = out_moge['depth'].squeeze().cpu().numpy()  # [H, W]
        H, W = depth.shape

        points_t = out_moge['points'].cpu()  # expect [3, H, W]
        # 强制 reshape 成 [H, W, 3]，避免维度顺序不一致
        points = points_t.view(3, -1).permute(1, 0).reshape(H, W, 3).numpy()

        mask_valid = out_moge['mask'].squeeze().cpu().numpy().astype(bool)
        intrinsics = out_moge['intrinsics'].cpu().numpy()
        fov_x, fov_y = utils3d.np.intrinsics_to_fov(intrinsics)

        # YOLO inference (segmentation)
        # ultralytics 接受 numpy(HWC BGR/RGB)，这里用 RGB
        results = yolo_model(image_rgb)
        r = results[0]

        panel_infos: List[Dict[str, Any]] = []
        inst_masks_np: Optional[np.ndarray] = None

        if r.masks is not None:
            # masks.data: [N, h_m, w_m]，需要缩放回原图大小
            masks_data = r.masks.data.cpu().numpy()  # [N, Hm, Wm]
            # 统一缩放到 MoGe 输出分辨率 [H, W]
            inst_masks_np = np.zeros((masks_data.shape[0], H, W), dtype=bool)
            for i, m in enumerate(masks_data):
                m_resized = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                inst_masks_np[i] = m_resized > 0.5

            boxes_xyxy = r.boxes.xyxy.cpu().numpy()  # [N, 4]
            clses = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else np.zeros(len(boxes_xyxy))
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.ones(len(boxes_xyxy))

            for i in range(inst_masks_np.shape[0]):
                inst_mask = inst_masks_np[i] & mask_valid
                pts_panel = points[inst_mask]
                
                # Filter out invalid points (inf/nan) just in case
                valid_pts = np.isfinite(pts_panel).all(axis=1)
                pts_panel = pts_panel[valid_pts]

                if pts_panel.shape[0] < 10:
                    continue

                dists = np.linalg.norm(pts_panel, axis=1)
                dist_median = float(np.median(dists))

                # Use PCA to find dimensions in 3D (handles slope and rotation)
                length_m, width_m = get_rect_dimensions_3d(pts_panel)

                panel_infos.append({
                    'box_xyxy': boxes_xyxy[i].tolist(),
                    'cls': int(clses[i]),
                    'conf': float(confs[i]),
                    'distance_median': dist_median,
                    'width_m': width_m,   # This is now the shorter dimension
                    'height_m': length_m, # This is now the longer dimension
                })

        # Save outputs
        rel = img_path.name if not input_path.is_dir() else img_path.relative_to(input_path)
        save_dir = output_root / rel.parent / rel.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        # 原图
        cv2.imwrite(str(save_dir / 'image.jpg'), image_bgr)

        # 深度可视化
        if save_vis:
            depth_vis = colorize_depth(depth)
            depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_dir / 'depth_vis.png'), depth_vis_bgr)

            # 画 YOLO 框
            overlay = image_bgr.copy()
            if panel_infos:
                for p in panel_infos:
                    x1, y1, x2, y2 = map(int, p['box_xyxy'])
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    txt = f"d={p['distance_median']:.1f}m {p['height_m']:.2f}x{p['width_m']:.2f}m"
                    cv2.putText(overlay, txt, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(str(save_dir / 'yolo_depth_overlay.jpg'), overlay)

        if save_json:
            with open(save_dir / 'panels.json', 'w') as f:
                json.dump({
                    'image': str(img_path),
                    'fov_x_deg': float(np.rad2deg(fov_x)),
                    'fov_y_deg': float(np.rad2deg(fov_y)),
                    'panels': panel_infos,
                }, f, indent=2)


if __name__ == '__main__':
    main()
