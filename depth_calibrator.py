import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

class ProbabilisticDepthCalibrator:
    """
    Non-parametric Probabilistic Depth Calibrator
    基于高斯过程回归 (GPR) 进行不确定性量化的深度修正。
    
    Paper Terminology: "Non-parametric Probabilistic Depth Calibration"
    """
    def __init__(self, kernel=None):
        # 定义核函数: 
        # C(1.0) * RBF(length_scale): 模拟深度之间的平滑相关性
        # WhiteKernel: 模拟观测噪声 (Noise Level)
        if kernel is None:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-1, 1e3)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1e1))
        
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        self.is_fitted = False
        
        self.X_train = [] 
        self.y_train = [] 

    def fit_from_pairs(self, raw_depths, true_depths):
        """
        一次性利用多个数据对进行训练
        raw_depths: list of floats from MoGe
        true_depths: list of floats from Ground Truth (Sensor)
        """
        self.X_train = list(raw_depths)
        self.y_train = list(true_depths)
        self.fit()
        
    def fit(self):
        if len(self.X_train) < 2:
            return
            
        X = np.array(self.X_train).reshape(-1, 1)
        y = np.array(self.y_train)
        
        self.gpr.fit(X, y)
        self.is_fitted = True
        print(f"[GPR] Fitted. Kernel: {self.gpr.kernel_}")

    def predict(self, raw_depth_map, return_std=True):
        """
        利用 LUT (查找表) 加速全图预测
        """
        if not self.is_fitted:
            # Default scale if not fitted (or use simple linear factor if known)
            return raw_depth_map, np.zeros_like(raw_depth_map)

        # 1. Build LUT for speed
        try:
            # Robustly find max value, ignoring NaN and Inf
            valid_mask = np.isfinite(raw_depth_map)
            if not np.any(valid_mask):
                max_val = 0.0
            else:
                max_val = np.max(raw_depth_map[valid_mask])
        except Exception:
            max_val = 0.0

        if not np.isfinite(max_val) or max_val <= 0: 
            return raw_depth_map, np.zeros_like(raw_depth_map)
        
        # 创建查询向量：从 0 到 当前最大深度+1米，采样500个点
        # Ensure query_X is strictly finite
        query_X = np.linspace(0, float(max_val) + 1.0, 500).reshape(-1, 1) 
        if not np.all(np.isfinite(query_X)):
             # Fallback if linspace fails somehow
             return raw_depth_map, np.zeros_like(raw_depth_map)

        # GPR Inference (只对这500个点做，非常快)
        pred_y, pred_std = self.gpr.predict(query_X, return_std=True)
        
        # 2. Interpolate (Vectorized)
        # 将查表结果映射回全分辨率图像
        flat_depth = raw_depth_map.flatten()
        
        # Handle NaNs in input for interp: replace with 0 for query, then restore mask?
        # np.interp propagates NaNs if x is NaN. That's fine for output map.
        # But we need to ensure we don't crash.
        
        calib_flat = np.interp(flat_depth, query_X.flatten(), pred_y)
        std_flat = np.interp(flat_depth, query_X.flatten(), pred_std)
        
        return calib_flat.reshape(raw_depth_map.shape), std_flat.reshape(raw_depth_map.shape)

    def save(self, filepath="gpr_calibration_model.pkl"):
        joblib.dump(self.gpr, filepath)
        print(f"[Calibration] Model saved to {filepath}")

    def load(self, filepath="gpr_calibration_model.pkl"):
        try:
            self.gpr = joblib.load(filepath)
            self.is_fitted = True
            print(f"[Calibration] Model loaded from {filepath}")
        except FileNotFoundError:
            print(f"[Error] Calibration model file not found: {filepath}")

    def visualize_calibration(self, save_path="calibration_curve.png"):
        """Generate calibration curve for the paper"""
        if not self.is_fitted:
            print("[Vis] Model not fitted, skipping visualization.")
            return

        # Use the stored training data if available, or just the GPR's training data
        if hasattr(self.gpr, "X_train_"):
             X_train = self.gpr.X_train_
             y_train = self.gpr.y_train_
        else:
             print("[Vis] No training data found in GPR model.")
             return

        x_min, x_max = X_train.min(), X_train.max()
        # Extend range slightly
        x_plot = np.linspace(max(0, x_min * 0.5), x_max * 1.5, 200).reshape(-1, 1)
        
        y_mean, y_std = self.gpr.predict(x_plot, return_std=True)
        
        plt.figure(figsize=(10, 6))
        # Plot training data
        plt.scatter(X_train, y_train, c='r', label='Calibration Pairs', zorder=10, s=20)
        
        # Plot predictive mean
        plt.plot(x_plot, y_mean, 'k-', label='GPR Mean Correction')
        
        # Plot uncertainty (95% confidence interval = 1.96 * std)
        plt.fill_between(x_plot[:, 0], 
                         y_mean - 1.96 * y_std, 
                         y_mean + 1.96 * y_std, 
                         alpha=0.2, color='b', label='95% Uncertainty (2$\sigma$)')
        
        # Plot identity line (Ideal uncalibrated scenario)
        plt.plot(x_plot, x_plot, 'g--', alpha=0.5, label='Ideal Identity')

        plt.xlabel("MoGe Predicted Depth (m)")
        plt.ylabel("Physical Depth (m)")
        plt.title("Constraint-Aware Probabilistic Depth Calibration")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        print(f"[Vis] Calibration plot saved to {save_path}")
