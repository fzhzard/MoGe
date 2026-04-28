import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


ARUCO_DICTIONARIES = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}


def parse_marker_ids(text: str | None) -> set[int] | None:
    if not text:
        return None
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def load_camera_parameters(path: str) -> tuple[np.ndarray, np.ndarray]:
    calibration_path = Path(path)
    suffix = calibration_path.suffix.lower()

    if suffix == ".json":
        with calibration_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(
            data.get("dist_coeffs", data.get("distortion_coefficients", [0, 0, 0, 0, 0])),
            dtype=np.float32,
        ).reshape(-1, 1)
        return camera_matrix, dist_coeffs

    if suffix == ".npz":
        data = np.load(calibration_path)
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
        if "dist_coeffs" in data:
            dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float32).reshape(-1, 1)
        elif "distortion_coefficients" in data:
            dist_coeffs = np.array(data["distortion_coefficients"], dtype=np.float32).reshape(-1, 1)
        else:
            dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        return camera_matrix, dist_coeffs

    if suffix in {".yml", ".yaml", ".xml"}:
        storage = cv2.FileStorage(str(calibration_path), cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise ValueError(f"Cannot open calibration file: {path}")
        camera_matrix = storage.getNode("camera_matrix").mat()
        dist_coeffs = storage.getNode("dist_coeffs").mat()
        if dist_coeffs is None:
            dist_coeffs = storage.getNode("distortion_coefficients").mat()
        storage.release()
        if camera_matrix is None:
            raise ValueError(f"Missing camera_matrix in calibration file: {path}")
        if dist_coeffs is None:
            dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        return np.array(camera_matrix, dtype=np.float32), np.array(dist_coeffs, dtype=np.float32).reshape(-1, 1)

    raise ValueError("Calibration file must be .json, .npz, .yml, .yaml, or .xml")


def build_camera_matrix(
    width: int,
    height: int,
    fov_x_deg: float | None,
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
) -> np.ndarray:
    if fx is None and fy is None:
        if fov_x_deg is None:
            raise ValueError("Provide either --camera_calibration, explicit fx/fy, or --fov_x")
        fx = width / (2.0 * math.tan(math.radians(fov_x_deg) / 2.0))
        fy = fx
    elif fx is None:
        fx = fy
    elif fy is None:
        fy = fx

    if fx is None or fy is None:
        raise ValueError("Unable to determine focal length")

    cx = width / 2.0 if cx is None else cx
    cy = height / 2.0 if cy is None else cy

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def create_detector(dictionary_name: str):
    if dictionary_name not in ARUCO_DICTIONARIES:
        supported = ", ".join(sorted(ARUCO_DICTIONARIES))
        raise ValueError(f"Unsupported dictionary: {dictionary_name}. Available: {supported}")

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_name])
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        def detect(gray_frame):
            return detector.detectMarkers(gray_frame)

        return dictionary, detect

    def detect(gray_frame):
        return cv2.aruco.detectMarkers(gray_frame, dictionary, parameters=parameters)

    return dictionary, detect


def estimate_marker_poses(
    corners: list[np.ndarray],
    marker_length: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            marker_length,
            camera_matrix,
            dist_coeffs,
        )
        return np.asarray(rvecs), np.asarray(tvecs)

    half_size = marker_length / 2.0
    object_points = np.array(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float32,
    )

    rvecs = []
    tvecs = []
    for marker_corners in corners:
        image_points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            continue
        rvecs.append(rvec.reshape(1, 3))
        tvecs.append(tvec.reshape(1, 3))

    if not rvecs:
        return np.empty((0, 1, 3), dtype=np.float32), np.empty((0, 1, 3), dtype=np.float32)

    return np.asarray(rvecs, dtype=np.float32), np.asarray(tvecs, dtype=np.float32)


def draw_marker_measurement(
    frame: np.ndarray,
    marker_corners: np.ndarray,
    marker_id: int,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length: float,
) -> dict:
    translation = np.asarray(tvec, dtype=np.float32).reshape(3)
    euclidean_distance = float(np.linalg.norm(translation))
    forward_distance = float(translation[2])
    lateral_distance = float(translation[0])
    vertical_distance = float(translation[1])

    corners_2d = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
    center = np.mean(corners_2d, axis=0).astype(int)

    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5, 2)
    cv2.circle(frame, tuple(center), 4, (0, 255, 255), -1)

    label_lines = [
        f"ID {marker_id}",
        f"dist={euclidean_distance:.3f} m",
        f"z={forward_distance:.3f} m",
        f"x={lateral_distance:.3f} m y={vertical_distance:.3f} m",
    ]
    base_x = int(np.min(corners_2d[:, 0]))
    base_y = max(30, int(np.min(corners_2d[:, 1])) - 10)
    for index, label in enumerate(label_lines):
        text_origin = (base_x, base_y + index * 20)
        cv2.putText(frame, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    return {
        "marker_id": marker_id,
        "distance_m": euclidean_distance,
        "forward_m": forward_distance,
        "x_m": lateral_distance,
        "y_m": vertical_distance,
    }


def process_video(args: argparse.Namespace) -> None:
    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {args.input}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps):
        fps = 30.0

    if args.camera_calibration:
        camera_matrix, dist_coeffs = load_camera_parameters(args.camera_calibration)
    else:
        camera_matrix = build_camera_matrix(width, height, args.fov_x, args.fx, args.fy, args.cx, args.cy)
        dist_coeffs = np.array(args.dist_coeffs if args.dist_coeffs else [0, 0, 0, 0, 0], dtype=np.float32).reshape(-1, 1)

    dictionary, detect_markers = create_detector(args.dictionary)
    selected_ids = parse_marker_ids(args.marker_ids)

    writer = None
    if args.output_video:
        output_video_path = Path(args.output_video)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    rows = []
    frame_index = 0
    detections = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        annotated_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect_markers(gray)

        frame_measurements = []
        if ids is not None and len(ids) > 0:
            ids = ids.flatten()
            if selected_ids is not None:
                filtered = [
                    (marker_corners, marker_id)
                    for marker_corners, marker_id in zip(corners, ids)
                    if int(marker_id) in selected_ids
                ]
                corners = [item[0] for item in filtered]
                ids = np.array([item[1] for item in filtered], dtype=np.int32)

            if len(corners) > 0:
                cv2.aruco.drawDetectedMarkers(annotated_frame, corners, ids)
                rvecs, tvecs = estimate_marker_poses(corners, args.marker_length, camera_matrix, dist_coeffs)
                for marker_corners, marker_id, rvec, tvec in zip(corners, ids, rvecs, tvecs):
                    measurement = draw_marker_measurement(
                        annotated_frame,
                        marker_corners,
                        int(marker_id),
                        rvec,
                        tvec,
                        camera_matrix,
                        dist_coeffs,
                        args.marker_length,
                    )
                    measurement["frame"] = frame_index
                    measurement["timestamp_sec"] = frame_index / fps
                    rows.append(measurement)
                    frame_measurements.append(measurement)
                    detections += 1

        if frame_measurements:
            nearest = min(frame_measurements, key=lambda item: item["distance_m"])
            status = (
                f"nearest id={nearest['marker_id']} dist={nearest['distance_m']:.3f} m "
                f"z={nearest['forward_m']:.3f} m"
            )
            status_color = (0, 255, 0)
        else:
            status = "No selected ArUco marker detected"
            status_color = (0, 0, 255)

        cv2.putText(annotated_frame, status, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)
        cv2.putText(
            annotated_frame,
            f"dict={args.dictionary} marker={args.marker_length:.3f} m frame={frame_index}",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(annotated_frame)
        if args.display:
            cv2.imshow("ArUco Range Demo", annotated_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_index += 1

    capture.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["frame", "timestamp_sec", "marker_id", "distance_m", "forward_m", "x_m", "y_m"]
            writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(rows)

    print("[ArUco] Processing complete")
    print(f"[ArUco] Frames processed: {frame_index}")
    print(f"[ArUco] Marker detections: {detections}")
    if rows:
        nearest = min(rows, key=lambda item: item["distance_m"])
        print(
            "[ArUco] Minimum distance: "
            f"{nearest['distance_m']:.3f} m at frame {nearest['frame']} (marker {nearest['marker_id']})"
        )
    else:
        print("[ArUco] No valid marker pose was estimated")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect ArUco markers in video and estimate UAV-to-marker metric distance"
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output_video", default="output/aruco_detection.mp4", help="Annotated output video path")
    parser.add_argument("--output_csv", default="output/aruco_distances.csv", help="Per-frame distance CSV path")
    parser.add_argument("--marker_length", required=True, type=float, help="Physical marker side length in meters")
    parser.add_argument("--dictionary", default="DICT_4X4_50", help="ArUco dictionary name")
    parser.add_argument("--marker_ids", default=None, help="Comma-separated marker IDs to keep, e.g. 0,1,2")
    parser.add_argument("--camera_calibration", default=None, help="Camera calibration file: json, npz, yml, yaml, or xml")
    parser.add_argument("--fov_x", type=float, default=None, help="Horizontal FOV in degrees, used if no calibration file is provided")
    parser.add_argument("--fx", type=float, default=None, help="Focal length fx in pixels")
    parser.add_argument("--fy", type=float, default=None, help="Focal length fy in pixels")
    parser.add_argument("--cx", type=float, default=None, help="Principal point cx in pixels")
    parser.add_argument("--cy", type=float, default=None, help="Principal point cy in pixels")
    parser.add_argument(
        "--dist_coeffs",
        type=float,
        nargs="+",
        default=None,
        help="Optional distortion coefficients k1 k2 p1 p2 [k3 ...] when no calibration file is provided",
    )
    parser.add_argument("--display", action="store_true", help="Show a live preview window")
    return parser


if __name__ == "__main__":
    process_video(build_parser().parse_args())