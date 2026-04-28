import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ARUCO_DICTIONARIES = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}


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


def infer_scale_mm_per_px(
    layout_width_px: float,
    layout_height_px: float,
    paper_width_mm: float,
    paper_height_mm: float,
    fit_axis: str,
) -> float:
    if layout_width_px <= 0 or layout_height_px <= 0:
        raise ValueError("Layout size in pixels must be positive")
    if paper_width_mm <= 0 or paper_height_mm <= 0:
        raise ValueError("Paper size in millimeters must be positive")

    if fit_axis == "width":
        return paper_width_mm / layout_width_px
    if fit_axis == "height":
        return paper_height_mm / layout_height_px
    if fit_axis == "best":
        return min(paper_width_mm / layout_width_px, paper_height_mm / layout_height_px)

    raise ValueError(f"Unsupported fit mode: {fit_axis}")


def resolve_board_geometry(args: argparse.Namespace) -> dict[str, float]:
    marker_length = args.marker_length
    marker_separation = args.marker_separation
    square_length = args.square_length

    if marker_length is None or marker_separation is None or square_length is None:
        layout_width_px = args.board_cols * args.marker_px + (args.board_cols - 1) * args.gap_px
        layout_height_px = args.board_rows * args.marker_px + (args.board_rows - 1) * args.gap_px
        mm_per_px = infer_scale_mm_per_px(
            layout_width_px=layout_width_px,
            layout_height_px=layout_height_px,
            paper_width_mm=args.paper_width_mm,
            paper_height_mm=args.paper_height_mm,
            fit_axis=args.fit_axis,
        )

        if marker_length is None:
            marker_length = (args.marker_px * mm_per_px) / 1000.0
        if marker_separation is None:
            marker_separation = (args.gap_px * mm_per_px) / 1000.0
        if square_length is None:
            square_length = ((args.marker_px + args.gap_px) * mm_per_px) / 1000.0

    if marker_length is None or marker_length <= 0:
        raise ValueError("Invalid marker length")
    if marker_separation is None or marker_separation < 0:
        raise ValueError("Invalid marker separation")
    if square_length is None or square_length <= 0:
        raise ValueError("Invalid ChArUco square length")
    if marker_length >= square_length:
        raise ValueError("marker_length must be smaller than square_length for ChArUco mode")

    return {
        "marker_length": float(marker_length),
        "marker_separation": float(marker_separation),
        "square_length": float(square_length),
    }


def create_aruco_detector(dictionary_name: str):
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

        def detect(gray_frame: np.ndarray):
            return detector.detectMarkers(gray_frame)

        return dictionary, detect

    def detect(gray_frame: np.ndarray):
        return cv2.aruco.detectMarkers(gray_frame, dictionary, parameters=parameters)

    return dictionary, detect


def create_charuco_board(dictionary: Any, squares_x: int, squares_y: int, square_length: float, marker_length: float):
    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            return cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
        except TypeError:
            pass

    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, dictionary)

    raise ValueError("Your OpenCV build does not support ChArUco boards")


def create_grid_board(dictionary: Any, markers_x: int, markers_y: int, marker_length: float, marker_separation: float):
    if hasattr(cv2.aruco, "GridBoard"):
        try:
            return cv2.aruco.GridBoard((markers_x, markers_y), marker_length, marker_separation, dictionary)
        except TypeError:
            pass

    if hasattr(cv2.aruco, "GridBoard_create"):
        return cv2.aruco.GridBoard_create(markers_x, markers_y, marker_length, marker_separation, dictionary)

    raise ValueError("Your OpenCV build does not support GridBoard")


def get_board_chessboard_corners(board: Any) -> np.ndarray:
    if hasattr(board, "getChessboardCorners"):
        corners = board.getChessboardCorners()
    elif hasattr(board, "chessboardCorners"):
        corners = board.chessboardCorners
    else:
        raise ValueError("Cannot read ChArUco board chessboard corners")

    return np.asarray(corners, dtype=np.float32).reshape(-1, 3)


def get_board_object_points(board: Any) -> np.ndarray:
    if hasattr(board, "getObjPoints"):
        points = board.getObjPoints()
    elif hasattr(board, "objPoints"):
        points = board.objPoints
    else:
        raise ValueError("Cannot read board object points")

    if isinstance(points, (list, tuple)):
        chunks = [np.asarray(item, dtype=np.float32).reshape(-1, 3) for item in points]
        return np.concatenate(chunks, axis=0)

    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


def interpolate_charuco_corners(
    gray: np.ndarray,
    marker_corners: list[np.ndarray],
    marker_ids: np.ndarray,
    board: Any,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    try:
        corner_count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=marker_corners,
            markerIds=marker_ids,
            image=gray,
            board=board,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
        )
    except TypeError:
        corner_count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
            camera_matrix,
            dist_coeffs,
        )

    if charuco_corners is None or charuco_ids is None:
        return 0, None, None

    return int(corner_count), np.asarray(charuco_corners, dtype=np.float32), np.asarray(charuco_ids, dtype=np.int32)


def estimate_charuco_pose(
    board: Any,
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    ids_flat = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    if ids_flat.size < 4:
        return False, None, None

    board_corners = get_board_chessboard_corners(board)
    if np.any(ids_flat < 0) or np.any(ids_flat >= board_corners.shape[0]):
        return False, None, None

    object_points = board_corners[ids_flat]
    image_points = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
    solve_flag = cv2.SOLVEPNP_IPPE if object_points.shape[0] < 6 else cv2.SOLVEPNP_ITERATIVE
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=solve_flag,
        )
    except cv2.error:
        return False, None, None
    if not ok:
        return False, None, None

    return True, np.asarray(rvec, dtype=np.float32), np.asarray(tvec, dtype=np.float32)


def estimate_grid_pose(
    board: Any,
    marker_corners: list[np.ndarray],
    marker_ids: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, int]:
    ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1, 1)
    if ids.size == 0:
        return False, None, None, 0

    if hasattr(cv2.aruco, "estimatePoseBoard"):
        try:
            valid_markers, rvec, tvec = cv2.aruco.estimatePoseBoard(
                marker_corners,
                ids,
                board,
                camera_matrix,
                dist_coeffs,
                None,
                None,
            )
        except TypeError:
            valid_markers, rvec, tvec = cv2.aruco.estimatePoseBoard(
                marker_corners,
                ids,
                board,
                camera_matrix,
                dist_coeffs,
            )
        if int(valid_markers) > 0 and rvec is not None and tvec is not None:
            return True, np.asarray(rvec, dtype=np.float32), np.asarray(tvec, dtype=np.float32), int(valid_markers)

    if hasattr(board, "matchImagePoints"):
        object_points, image_points = board.matchImagePoints(marker_corners, ids)
        if object_points is not None and image_points is not None:
            object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
            image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
            if object_points.shape[0] >= 4 and image_points.shape[0] >= 4:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok:
                    return True, np.asarray(rvec, dtype=np.float32), np.asarray(tvec, dtype=np.float32), int(len(ids))

    return False, None, None, 0


def board_center_in_camera(rvec: np.ndarray, tvec: np.ndarray, board_center: np.ndarray) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32).reshape(3, 1))
    center_cam = rotation_matrix @ board_center.reshape(3, 1) + np.asarray(tvec, dtype=np.float32).reshape(3, 1)
    return center_cam.reshape(3)


def draw_pose_measurement(
    frame: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length: float,
    board_center: np.ndarray,
) -> dict[str, float]:
    origin_cam = np.asarray(tvec, dtype=np.float32).reshape(3)
    center_cam = board_center_in_camera(rvec, tvec, board_center)

    distance_origin = float(np.linalg.norm(origin_cam))
    distance_center = float(np.linalg.norm(center_cam))
    forward_distance = float(center_cam[2])
    lateral_distance = float(center_cam[0])
    vertical_distance = float(center_cam[1])

    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length, 2)

    return {
        "distance_center_m": distance_center,
        "distance_origin_m": distance_origin,
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

    board_cols = args.squares_x if args.squares_x is not None else args.board_cols
    board_rows = args.squares_y if args.squares_y is not None else args.board_rows
    if board_cols < 2 or board_rows < 2:
        raise ValueError("Board columns and rows must be >= 2")

    geometry = resolve_board_geometry(args)
    dictionary, detect_markers = create_aruco_detector(args.dictionary)

    charuco_board = None
    charuco_center = None
    if args.board_mode in {"charuco", "auto"}:
        charuco_board = create_charuco_board(
            dictionary=dictionary,
            squares_x=board_cols,
            squares_y=board_rows,
            square_length=geometry["square_length"],
            marker_length=geometry["marker_length"],
        )
        charuco_center = np.mean(get_board_chessboard_corners(charuco_board), axis=0).astype(np.float32)

    grid_board = None
    grid_center = None
    if args.board_mode in {"grid", "auto"}:
        grid_board = create_grid_board(
            dictionary=dictionary,
            markers_x=board_cols,
            markers_y=board_rows,
            marker_length=geometry["marker_length"],
            marker_separation=geometry["marker_separation"],
        )
        grid_center = np.mean(get_board_object_points(grid_board), axis=0).astype(np.float32)

    writer = None
    if args.output_video:
        output_video_path = Path(args.output_video)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    axis_length = args.axis_length if args.axis_length is not None else geometry["marker_length"] * 0.8
    rows = []
    frame_index = 0
    pose_detections = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        annotated_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = detect_markers(gray)

        status = "No board detected"
        status_color = (0, 0, 255)

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated_frame, marker_corners, marker_ids)

            pose_ok = False
            mode_used = ""
            rvec = None
            tvec = None
            support_count = 0
            board_center = None

            if charuco_board is not None:
                corner_count, charuco_corners, charuco_ids = interpolate_charuco_corners(
                    gray,
                    marker_corners,
                    marker_ids,
                    charuco_board,
                    camera_matrix,
                    dist_coeffs,
                )
                if charuco_corners is not None and charuco_ids is not None:
                    cv2.aruco.drawDetectedCornersCharuco(annotated_frame, charuco_corners, charuco_ids, (255, 255, 0))
                    if corner_count >= args.min_corners:
                        pose_ok, rvec, tvec = estimate_charuco_pose(
                            charuco_board,
                            charuco_corners,
                            charuco_ids,
                            camera_matrix,
                            dist_coeffs,
                        )
                        if pose_ok:
                            mode_used = "charuco"
                            support_count = int(corner_count)
                            board_center = charuco_center

            if not pose_ok and grid_board is not None:
                pose_ok, rvec, tvec, valid_markers = estimate_grid_pose(
                    grid_board,
                    marker_corners,
                    marker_ids,
                    camera_matrix,
                    dist_coeffs,
                )
                if pose_ok:
                    mode_used = "grid"
                    support_count = int(valid_markers)
                    board_center = grid_center

            if pose_ok and rvec is not None and tvec is not None and board_center is not None:
                measurement = draw_pose_measurement(
                    annotated_frame,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                    axis_length,
                    board_center,
                )
                measurement["frame"] = frame_index
                measurement["timestamp_sec"] = frame_index / fps
                measurement["mode"] = mode_used
                measurement["support_count"] = support_count
                rows.append(measurement)
                pose_detections += 1

                status = (
                    f"mode={mode_used} dist={measurement['distance_center_m']:.3f} m "
                    f"z={measurement['forward_m']:.3f} m support={support_count}"
                )
                status_color = (0, 255, 0)
            else:
                status = f"Markers={len(marker_ids)} pose solve failed"

        cv2.putText(annotated_frame, status, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.66, status_color, 2, cv2.LINE_AA)
        cv2.putText(
            annotated_frame,
            (
                f"dict={args.dictionary} mode={args.board_mode} board={board_cols}x{board_rows} "
                f"marker={geometry['marker_length']:.4f}m sep={geometry['marker_separation']:.4f}m frame={frame_index}"
            ),
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(annotated_frame)
        if args.display:
            cv2.imshow("ChArUco/GridBoard Range Demo", annotated_frame)
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
            fieldnames = [
                "frame",
                "timestamp_sec",
                "mode",
                "support_count",
                "distance_center_m",
                "distance_origin_m",
                "forward_m",
                "x_m",
                "y_m",
            ]
            writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(rows)

    print("[BoardPose] Processing complete")
    print(f"[BoardPose] Frames processed: {frame_index}")
    print(f"[BoardPose] Pose detections: {pose_detections}")
    print(
        "[BoardPose] Geometry used: "
        f"marker_length={geometry['marker_length']:.6f} m, "
        f"marker_separation={geometry['marker_separation']:.6f} m, "
        f"charuco_square_length={geometry['square_length']:.6f} m"
    )
    if rows:
        nearest = min(rows, key=lambda item: item["distance_center_m"])
        print(
            "[BoardPose] Minimum center distance: "
            f"{nearest['distance_center_m']:.3f} m at frame {nearest['frame']} "
            f"(mode={nearest['mode']}, support={nearest['support_count']})"
        )
    else:
        print("[BoardPose] No valid board pose was estimated")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect ChArUco/GridBoard in video and estimate camera-to-board metric distance"
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output_video", default="output/charuco_detection.mp4", help="Annotated output video path")
    parser.add_argument("--output_csv", default="output/charuco_distances.csv", help="Per-frame distance CSV path")

    parser.add_argument(
        "--camera_calibration",
        default=None,
        help="Camera calibration file: json, npz, yml, yaml, or xml",
    )
    parser.add_argument(
        "--fov_x",
        type=float,
        default=None,
        help="Horizontal FOV in degrees, used if no calibration file is provided",
    )
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

    parser.add_argument("--dictionary", default="DICT_4X4_50", help="ArUco dictionary name")
    parser.add_argument("--board_mode", choices=["auto", "charuco", "grid"], default="grid", help="Pose solver mode")

    parser.add_argument("--board_cols", type=int, default=4, help="Board columns (your board: 4)")
    parser.add_argument("--board_rows", type=int, default=5, help="Board rows (your board: 5)")
    parser.add_argument("--marker_px", type=float, default=100.0, help="Marker size in the design image (px)")
    parser.add_argument("--gap_px", type=float, default=10.0, help="Gap between markers in the design image (px)")
    parser.add_argument("--paper_width_mm", type=float, default=210.0, help="Printed paper width in mm (A4=210)")
    parser.add_argument("--paper_height_mm", type=float, default=297.0, help="Printed paper height in mm (A4=297)")
    parser.add_argument("--fit_axis", choices=["width", "height", "best"], default="width", help="How the board image is fitted on paper")

    parser.add_argument(
        "--marker_length",
        type=float,
        default=None,
        help="Marker side length in meters. If omitted, infer from marker_px/gap_px/paper size",
    )
    parser.add_argument(
        "--marker_separation",
        type=float,
        default=None,
        help="GridBoard marker separation in meters. If omitted, infer from marker_px/gap_px/paper size",
    )
    parser.add_argument(
        "--square_length",
        type=float,
        default=None,
        help="ChArUco square side length in meters. If omitted, infer from marker_px+gap_px",
    )

    parser.add_argument("--squares_x", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--squares_y", type=int, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--min_corners", type=int, default=6, help="Minimum ChArUco corners required to solve pose")
    parser.add_argument("--axis_length", type=float, default=None, help="Axis length for visualization (meters)")
    parser.add_argument("--display", action="store_true", help="Show a live preview window")
    return parser


if __name__ == "__main__":
    process_video(build_parser().parse_args())