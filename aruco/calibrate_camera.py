import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}


def iter_input_images(input_path: str, sample_step: int, max_samples: int | None):
    path = Path(input_path)
    if path.exists() and path.is_file() and path.suffix in VIDEO_EXTENSIONS:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        frame_index = 0
        yielded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % max(sample_step, 1) == 0:
                yield f"frame_{frame_index:06d}", frame
                yielded += 1
                if max_samples is not None and yielded >= max_samples:
                    break
            frame_index += 1
        capture.release()
        return

    if path.exists() and path.is_dir():
        candidates = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            candidates.extend(sorted(path.glob(pattern)))
    else:
        candidates = [Path(item) for item in sorted(glob.glob(input_path))]

    if not candidates:
        raise ValueError(f"No calibration images found for: {input_path}")

    for index, image_path in enumerate(candidates):
        if max_samples is not None and index >= max_samples:
            break
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        yield image_path.name, image


def detect_corners(gray: np.ndarray, pattern_size: tuple[int, int]):
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def build_object_points(pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    object_points = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points[:, :2] = grid * square_size
    return object_points


def save_calibration(output_path: Path, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, image_size: tuple[int, int], rms: float):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "rms_reprojection_error": float(rms),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def main(args: argparse.Namespace) -> None:
    pattern_size = (args.pattern_cols, args.pattern_rows)
    template_object_points = build_object_points(pattern_size, args.square_size)

    object_points = []
    image_points = []
    image_size = None
    valid_samples = 0

    for sample_name, image in iter_input_images(args.input, args.sample_step, args.max_samples):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = detect_corners(gray, pattern_size)
        if not found:
            continue

        object_points.append(template_object_points.copy())
        image_points.append(np.asarray(corners, dtype=np.float32))
        image_size = (gray.shape[1], gray.shape[0])
        valid_samples += 1

        if args.preview:
            preview = image.copy()
            cv2.drawChessboardCorners(preview, pattern_size, corners, found)
            cv2.imshow("Calibration Preview", preview)
            if cv2.waitKey(200) & 0xFF == 27:
                break

        print(f"[Calibration] accepted: {sample_name}")

    if args.preview:
        cv2.destroyAllWindows()

    if valid_samples < 10:
        raise ValueError(
            f"Only {valid_samples} valid calibration views found. Collect at least 10 to 20 views with wide angle variation."
        )

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    save_calibration(Path(args.output), camera_matrix, dist_coeffs, image_size, rms)

    print("[Calibration] complete")
    print(f"[Calibration] valid views: {valid_samples}")
    print(f"[Calibration] RMS reprojection error: {rms:.6f}")
    print("[Calibration] camera_matrix:")
    print(camera_matrix)
    print("[Calibration] dist_coeffs:")
    print(dist_coeffs.reshape(-1))
    print(f"[Calibration] saved to: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics from chessboard images or video")
    parser.add_argument("--input", required=True, help="Image directory, image glob, or calibration video path")
    parser.add_argument("--output", default="aruco/camera_calibration.json", help="Output JSON file")
    parser.add_argument("--pattern_cols", type=int, required=True, help="Number of inner corners along board width")
    parser.add_argument("--pattern_rows", type=int, required=True, help="Number of inner corners along board height")
    parser.add_argument("--square_size", type=float, required=True, help="Chessboard square size in meters")
    parser.add_argument("--sample_step", type=int, default=15, help="For video input, sample one frame every N frames")
    parser.add_argument("--max_samples", type=int, default=80, help="Maximum number of images or sampled frames to use")
    parser.add_argument("--preview", action="store_true", help="Show accepted calibration frames")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())