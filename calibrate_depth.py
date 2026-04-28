import argparse
import numpy as np
import glob
import os
import torch
import cv2
import sys
from pathlib import Path

# Add project root to path
if (_package_root := str(Path(__file__).absolute().parents[1])) not in sys.path:
    sys.path.insert(0, _package_root)

from depth_calibrator import ProbabilisticDepthCalibrator
# Assuming we can import the MoGe model to run references if needed
# But for calibration, we might just need pairs of (predicted, real) 
# which might come from a text file or manual measurements.

def generate_dummy_data():
    """
    Generate synthetic calibration data for demonstration.
    Simulate the 'Scale Drift' problem:
    Real = 1.4m, Predicted = 0.8m
    """
    print("[Info] Generating synthetic calibration data (Simulating Scale Drift)...")
    
    # Real distances (e.g., from LiDAR or Measure Tape)
    real_depths = np.linspace(0.5, 5.0, 20)
    
    # Predicted depths (Simulated logic: MoGe underestimates significantly)
    # y = x * 0.6 + noise
    pred_depths = real_depths * 0.6 + np.random.normal(0, 0.05, size=len(real_depths))
    
    # Add some non-linearity for GPR to learn
    # e.g., at very close range, the error is different
    pred_depths[real_depths < 1.0] = real_depths[real_depths < 1.0] * 0.8

    return pred_depths, real_depths

def main():
    parser = argparse.ArgumentParser(description="Train Depth Calibration Model (GPR)")
    parser.add_argument("--save_path", type=str, default="gpr_calibration_model.pkl", help="Where to save the model")
    parser.add_argument("--data_file", type=str, default=None, help="Path to a .npz or .txt file with 'pred' and 'real' arrays")
    args = parser.parse_args()

    calibrator = ProbabilisticDepthCalibrator()

    if args.data_file and os.path.exists(args.data_file):
        print(f"[Data] Loading data from {args.data_file}")
        data = np.load(args.data_file)
        if 'pred' in data and 'real' in data:
            pred_depths = data['pred']
            real_depths = data['real']
        else:
            print("[Error] .npz file must contain 'pred' and 'real' arrays.")
            return
    else:
        # Use dummy data
        pred_depths, real_depths = generate_dummy_data()

    # Fit the model
    # Note: depth_calibrator.py has fit(predicted_depths, real_depths) 
    # OR fit_from_pairs(list, list). 
    # Let's use fit_from_pairs as it handles list conversion.
    
    calibrator.fit_from_pairs(pred_depths, real_depths)
    
    # Save
    calibrator.save(args.save_path)
    
    # Visualize
    vis_path = args.save_path.replace(".pkl", ".png")
    calibrator.visualize_calibration(save_path=vis_path)
    
    print("\n[Summary]")
    print(f"Calibration Model Saved: {args.save_path}")
    print(f"Visualization Saved:     {vis_path}")
    print("\nHow to use in inference:")
    print(f"python uav_video_demo.py --calibration_path {args.save_path} ...")

if __name__ == "__main__":
    main()
