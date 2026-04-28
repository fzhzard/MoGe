import argparse
import cv2
import numpy as np
import torch
import sys
from pathlib import Path
from tqdm import tqdm

# Ensure moge path inclusion
if (_package_root := str(Path(__file__).absolute().parents[1])) not in sys.path:
    sys.path.insert(0, _package_root)

from moge.model import import_model_class_by_version
from moge.utils.vis import colorize_depth_affine

# =========================================================================
# GBD-Sense: Geometric-Bayesian-Dynamic Proximity Perception Framework
# =========================================================================

class GBDSensor:
    """
    Implements the 3-stage GBD methodology for UAV Safety:
    1. Geometric Manifold Decoupling
    2. Spherical Bayesian Fusion
    3. Kinematic-Aware Risk Assessment
    """
    def __init__(self, safe_radius=2.0, decay_factor=0.85, prob_threshold=3.0, scale_factor=1.0, shift_factor=0.0):
        # Apply scaling to safety radius logic if needed, 
        # but usually we scale the POINT CLOUD, not the threshold.
        # So we keep safe_radius as the target physical distance (e.g. 2.0m real world)
        self.safe_radius = safe_radius 
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor
        
        # [Stage II] Spherical Spatiotemporal Bayesian Grid
        # Dimensions: Azimuth (36), Elevation (18), Radius (20 bins)
        self.grid_dim = (36, 18, 20)
        self.grid_map = np.zeros(self.grid_dim, dtype=np.float32)
        self.decay_factor = decay_factor
        self.prob_threshold = prob_threshold # Log-odds threshold for "Verified"
        
        # Resolution: ~10cm per radial bin
        self.r_step = safe_radius / self.grid_dim[2]
        
    def process(self, points, normals, uav_velocity=np.array([0,0,1.0])):
        """
        Process a single frame through the GBD pipeline.
        
        Args:
            points: (H, W, 3) metric point cloud in Camera Frame
            normals: (H, W, 3) normal map
            uav_velocity: (3,) velocity vector in Camera Frame (m/s)
        """
        # --- 0. Affine Calibration (Data Alignment) ---
        # SCI Interpretation: Bridging the domain gap with Affine Transformation
        # P_calib = scale * P_pred + shift * (P_pred / |P_pred|)
        # This shifts points radially.
        
        # Note: We create a copy implicitly by operation to avoid modifying original array if needed,
        # but for efficiency we can modify.
        
        # Apply Scale
        if self.scale_factor != 1.0:
            points = points * self.scale_factor
            
        # Apply Shift (Radial Bias Correction)
        # Some sensors have a constant offset error.
        if self.shift_factor != 0.0:
            # We assume shift is along the ray (distance bias)
            norms = np.linalg.norm(points, axis=-1, keepdims=True)
            # Avoid div by zero
            norms = np.maximum(norms, 1e-6) 
            # Direction vectors
            dirs = points / norms
            # Add shift
            points = points + dirs * self.shift_factor

        if points.ndim == 3: points = points.reshape(-1, 3)
        if normals.ndim == 3: normals = normals.reshape(-1, 3)
        
        # =========================================================
        # Stage I: Geometric Manifold Decoupling
        # =========================================================
        
        # 1.1 Spatial Filter: Only consider points within Safety Sphere
        dists = np.linalg.norm(points, axis=-1)
        sphere_mask = dists < self.safe_radius
        
        if not np.any(sphere_mask):
            self.grid_map *= self.decay_factor 
            return [], 0.0
            
        valid_points = points[sphere_mask]
        valid_normals = normals[sphere_mask]
        
        # 1.2 Manifold Fitting (Robust Plane Fitting)
        # Use Median of Normals for robustness (Simplified RANSAC)
        if len(valid_normals) > 50:
            plane_n = np.median(valid_normals, axis=0) # (3,)
            plane_n /= (np.linalg.norm(plane_n) + 1e-6)
            
            # Plane distance d: n.p + d = 0  =>  d = -median(n.p)
            plane_d = -np.median(np.dot(valid_points, plane_n))
        else:
            plane_n = np.array([0, 0, 1.0])
            plane_d = -1.0

        # 1.3 Anomaly Score Calculation
        # Spatial Residual
        dist_resid = np.abs(np.dot(valid_points, plane_n) + plane_d)
        # Manifold Deviation: 1 - |n_i . n_plane|
        angle_resid = 1.0 - np.abs(np.dot(valid_normals, plane_n))
        
        # 1.4 Candidate Extraction
        # Logic: 
        # 1. Anomaly: Always candidate (e.g. bird poop, power lines)
        # 2. Extreme Proximity: If any surface (even smooth PV panel) is < 1.0m, it's a candidate (Danger)
        # 3. Safe Working Zone (1.0m - safe_radius): Smooth panels are filtered out, only anomalies are kept.
        
        # Re-eval distances for valid subset
        valid_dists = np.linalg.norm(valid_points, axis=-1)
        
        # Define absolute danger zone (e.g. 1.0m crash limit)
        CRITICAL_CRASH_DIST = 1.0 
        is_extreme_proximity = valid_dists < CRITICAL_CRASH_DIST
        
        # Geometric anomaly
        is_anomaly = (dist_resid > 0.10) | (angle_resid > 0.10) 
        
        is_candidate = is_anomaly | is_extreme_proximity
        candidate_points = valid_points[is_candidate]
        
        # =========================================================
        # Stage II: Spherical Bayesian Fusion
        # =========================================================
        self.grid_map *= self.decay_factor # Temporal Decay
        
        if len(candidate_points) > 0:
            # Convert to Spherical Coords
            X, Y, Z = candidate_points[:, 0], candidate_points[:, 1], candidate_points[:, 2]
            R = np.sqrt(X**2 + Y**2 + Z**2)
            Phi = np.arctan2(Y, X) # Azimuth: -pi to pi
            Theta = np.arccos(np.clip(Z / (R + 1e-6), -1, 1)) # Elevation: 0 to pi
            
            # Bins
            r_idx = np.clip((R / self.r_step).astype(int), 0, self.grid_dim[2]-1)
            phi_idx = np.clip(((Phi + np.pi) / (2*np.pi) * self.grid_dim[0]).astype(int), 0, self.grid_dim[0]-1)
            theta_idx = np.clip((Theta / np.pi * self.grid_dim[1]).astype(int), 0, self.grid_dim[1]-1)
            
            # Bayesian Update (Log-Odds increment)
            np.add.at(self.grid_map, (phi_idx, theta_idx, r_idx), 0.8) 
            
        self.grid_map = np.clip(self.grid_map, 0, 10.0)
        
        # Extract Verified Obstacles
        active_indices = np.where(self.grid_map > self.prob_threshold)
        
        verified_points = []
        risk_score = 0.0
        
        # =========================================================
        # Stage III: Kinematic-Aware Risk Assessment
        # =========================================================
        
        if len(active_indices[0]) > 0:
            idx_phi, idx_theta, idx_r = active_indices
            
            phis = (idx_phi + 0.5) / self.grid_dim[0] * 2 * np.pi - np.pi
            thetas = (idx_theta + 0.5) / self.grid_dim[1] * np.pi
            rs = (idx_r + 0.5) * self.r_step
            
            # Map back to cartesian for distance calculation
            zs = rs * np.cos(thetas)
            xs = rs * np.sin(thetas) * np.cos(phis)
            ys = rs * np.sin(thetas) * np.sin(phis)
            
            verified_points = np.stack([xs, ys, zs], axis=1)
            
            # Dynamic Potential Field
            # Stretch distance metric along velocity vector (+Z assumed)
            v_mag = np.linalg.norm(uav_velocity)
            stretch_z = 1.0 + v_mag 
            
            d_dyn = np.sqrt(xs**2 + ys**2 + (zs / stretch_z)**2)
            
            risk_mask = d_dyn < self.safe_radius
            if np.any(risk_mask):
                d_eff = d_dyn[risk_mask]
                # Risk Potential: 0.5 * (1/d - 1/R)^2
                potentials = 0.5 * (1.0/(d_eff+0.1) - 1.0/self.safe_radius)**2
                risk_score = np.sum(potentials)
                
        return verified_points, risk_score


class VideoSafetySystem:
    def __init__(self, model_path="Ruicheng/moge-2-vitl", device="cuda"):
        print(f"[INFO] Initializing Video Safety System with model: {model_path}")
        self.device = torch.device(device)
        
        # Determine model type
        if "moge-2" in model_path:
            ModelClass = import_model_class_by_version("v2")
        else:
            ModelClass = import_model_class_by_version("v1")
            
        self.model = ModelClass.from_pretrained(model_path).to(self.device).eval()
        if self.device.type == 'cuda':
            self.model.half()
            
        # Initialize GBD Sensor
        self.gbd_sensor = GBDSensor(safe_radius=1.5)
        
    def process_video(self, video_path, output_path="output.mp4", safe_radius=1.5, fov_x=None, scale_factor=1.0, shift_factor=0.0):
        self.gbd_sensor.safe_radius = safe_radius
        self.gbd_sensor.scale_factor = scale_factor
        self.gbd_sensor.shift_factor = shift_factor
        
        # Open Video
        try:
            val = int(video_path)
            cap = cv2.VideoCapture(val)
        except:
            cap = cv2.VideoCapture(video_path)
            
        if not cap.isOpened():
            raise ValueError(f"Cannot open {video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Writer setup
        out_dims = (width, height * 2) 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, out_dims)
        
        # Inference Size (multiple of 14)
        infer_w = 640
        infer_h = int(infer_w * height / width)
        infer_h = (infer_h // 14) * 14
        infer_w = (infer_w // 14) * 14
        
        pbar = tqdm()
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Inference
            img_in = cv2.resize(frame, (infer_w, infer_h))
            img_tensor = torch.tensor(img_in[...,::-1]/255.0, dtype=torch.float32, device=self.device).permute(2,0,1).unsqueeze(0)
            
            with torch.inference_mode():
                out = self.model.infer(img_tensor, fov_x=fov_x, use_fp16=(self.device.type=='cuda'))
                
            points = out['points'].squeeze(0).cpu().numpy() # (H, W, 3)
            if 'normals' in out:
                normals = out['normals'].squeeze(0).cpu().numpy()
            elif 'normal' in out:
                normals = out['normal'].squeeze(0).cpu().numpy()
            else:
                normals = np.zeros_like(points); normals[..., 2] = 1.0
            
            # 2. Apply manual scale and shift to the raw point cloud immediately
            # so that visualization (Depth Map) AND GBD processing use the same data.
            if scale_factor != 1.0:
                points = points * scale_factor

            if shift_factor != 0.0:
                norms = np.linalg.norm(points, axis=-1, keepdims=True)
                norms = np.maximum(norms, 1e-6)
                points = points + (points / norms) * shift_factor

            # 2. GBD Pipeline
            # Note: We set scale_factor=1.0 in GBD sensor to avoid double scaling
            # verified_obs, risk_score = self.gbd_sensor.process(points, normals)
            # But wait, self.gbd_sensor.scale_factor was set in process_video magnitude
            # We must temporarily disable it or change how we call it.

            # Let's rely on the global scaling above and force sensor's internal scale to 1.0
            # effectively bypassing its internal calibration block.
            old_sensor_scale = self.gbd_sensor.scale_factor
            old_sensor_shift = self.gbd_sensor.shift_factor
            self.gbd_sensor.scale_factor = 1.0
            self.gbd_sensor.shift_factor = 0.0

            verified_obs, risk_score = self.gbd_sensor.process(points, normals)

            # Restore just in case (though we overwrite it every frame in process_video start currently)
            self.gbd_sensor.scale_factor = old_sensor_scale
            self.gbd_sensor.shift_factor = old_sensor_shift
            
            # 3. Visualization
            vis_img = frame.copy()
            cx, cy = width//2, height//2
            
            # Draw Spherical Grid (HUD) - "Wireframe Sphere" effect
            # We project a virtual sphere wireframe onto the image
            f_est = 0.8 * width
            
            # 1. Draw concentric circles (Longitude lines)
            radii_steps = [0.5, 1.0, 1.5, 2.0] # meters
            # Approximate projection: r_px = f * R_real / Z_virtual (assuming Z ~ R for periphery)
            # A better HUD approx for a sphere around camera is simply concentric circles 
            # whose radius corresponds to FOV angles, but here we want "distance shells".
            # For a forward-looking camera, a sphere of radius R is hard to visualize "around" the drone 
            # on a 2D image without 3rd person view.
            # Instead, we visualize the "Cross-section" of the safety sphere at different depths
            
            # Let's draw a "Tunnel" grid effect to simulate 3D space
            grid_color = (0, 255, 0) # Safe color
            alert_color = (0, 0, 255) # Warning color
            
            current_color = grid_color
            if risk_score > 1.0: current_color = (0, 165, 255) # Orange
            if risk_score > 5.0: current_color = alert_color # Red

            # Draw "Radar" circles
            cv2.circle(vis_img, (cx, cy), 50, current_color, 1)
            cv2.circle(vis_img, (cx, cy), 150, current_color, 1)
            cv2.circle(vis_img, (cx, cy), 250, current_color, 2)
            
            # Draw diagonal crosshairs
            cv2.line(vis_img, (cx-300, cy), (cx+300, cy), current_color, 1)
            cv2.line(vis_img, (cx, cy-200), (cx, cy+200), current_color, 1)
            
            # HUD Text Status
            if risk_score > 5.0:
                text = f"CRITICAL DIST: {min([np.linalg.norm(p) for p in verified_obs]) if len(verified_obs)>0 else 0:.1f}m"
                bg_color = (0, 0, 255)
            elif risk_score > 1.0:
                text = f"WARNING ({risk_score:.1f})"
                bg_color = (0, 165, 255)
            else:
                text = "SAFE SPHERE"
                bg_color = (0, 200, 0)
                
            # Draw text with background box
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(vis_img, (20, 20), (20+tw+20, 20+th+20), bg_color, -1)
            cv2.putText(vis_img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Draw Verified Obstacles (Projected) with "Enveloping" effect
            if len(verified_obs) > 0:
                # Find bounding box of obstacles to draw an "envelope"
                u_coords, v_coords = [], []
                min_dist_obs = 999.0
                
                for (ox, oy, oz) in verified_obs:
                     dist = np.sqrt(ox**2 + oy**2 + oz**2)
                     if dist < min_dist_obs: min_dist_obs = dist
                     
                     if oz > 0.1:
                         u = int(cx + f_est * ox / oz)
                         v = int(cy + f_est * oy / oz)
                         if 0 <= u < width and 0 <= v < height:
                             # Draw individual points as small nodes
                             cv2.circle(vis_img, (u, v), 3, (0, 0, 255), -1)
                             u_coords.append(u)
                             v_coords.append(v)
                
                # Draw Convex Hull (The "Envelope")
                if len(u_coords) >= 3:
                    pts = np.vstack((u_coords, v_coords)).T
                    hull = cv2.convexHull(pts)
                    # Draw transparent overlay
                    overlay = vis_img.copy()
                    cv2.fillPoly(overlay, [hull], (0, 0, 255))
                    alpha = 0.4
                    vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)
                    # Draw contour
                    cv2.polylines(vis_img, [hull], True, (0, 0, 255), 2)
                    
                    # Label distance on the centroid
                    cen_u = int(np.mean(u_coords))
                    cen_v = int(np.mean(v_coords))
                    cv2.putText(vis_img, f"{min_dist_obs:.2f}m", (cen_u, cen_v), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                elif len(u_coords) > 0:
                    cen_u = int(np.mean(u_coords))
                    cen_v = int(np.mean(v_coords))
                    cv2.circle(vis_img, (cen_u, cen_v), 20, (0, 0, 255), 2)
                    cv2.putText(vis_img, f"{min_dist_obs:.2f}m", (cen_u+25, cen_v), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Depth
            # Use original points for visualization to avoid confusion with scaled ones?
            # Or use scaled ones to show consistency? Let's use points which are SCALED now in GBDSensor but...
            # Wait, points were passed by value or reference?
            # In Python, numpy arrays are passed by reference but operations like `points = points * scale` create new array
            # inside the function scope unless `points[:] = ...` is used.
            # So `points` in `process_video` (the caller) remains UNSCALED.
            # This is the BUG! The depth visualization and HUD distance were using different scales.
            # But wait, verified_obs ARE scaled because they come from `gbd_sensor.process`.

            depth_vis = colorize_depth_affine(points[..., 2], cmap='Spectral')
            depth_vis = cv2.resize(depth_vis, (width, height))
            
            final = np.vstack([vis_img, depth_vis])
            writer.write(final)
            pbar.update(1)

            
        cap.release()
        writer.release()
        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Video path or camera index")
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--model", default="Ruicheng/moge-2-vitl-normal.pt")
    parser.add_argument("--fov", type=float, default=None, help="Horizontal Field of View (degrees)")
    parser.add_argument("--scale", type=float, default=1.0, help="Linear calibration factor (Multiplier)")
    parser.add_argument("--shift", type=float, default=0.0, help="Affine calibration shift (Bias in meters)")
    parser.add_argument("--safe_radius", type=float, default=1.5, help="Safety radius for obstacle avoidance")
    args = parser.parse_args()

    system = VideoSafetySystem(model_path=args.model)
    system.process_video(args.input, output_path=args.output, fov_x=args.fov, scale_factor=args.scale, shift_factor=args.shift, safe_radius=args.safe_radius)
