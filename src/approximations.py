import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time
from scipy.stats import norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RandomFourierFeatures:
    """Random Fourier Features approximation with uncertainty"""
    
    def __init__(self, n_components=1000, lengthscale=1.0, sigma_noise=0.1, random_state=42):
        """
        Args:
            n_components: Number of random features (D)
            lengthscale: RBF kernel lengthscale
            sigma_noise: Observation noise
            random_state: Random seed
        """
        self.n_components = n_components
        self.lengthscale = lengthscale
        self.sigma_noise = sigma_noise
        self.random_state = random_state
        
        self.weights = None  # Random projection weights
        self.bias = None     # Random bias
        self.alpha = None    # Linear regression weights
        self.covariance = None  # Posterior covariance for uncertainty
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.training_time = 0
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    def _transform(self, X):
        """Transform input to random feature space"""
        # X @ W + b
        projection = X @ self.weights + self.bias
        # sqrt(2/D) * cos(projection)
        Z = np.sqrt(2.0 / self.n_components) * np.cos(projection)
        return Z
    
    def fit(self, X_train, y_train):
        """Train RFF model"""
        start_time = time.time()
        
        X_scaled = self.scaler_X.fit_transform(X_train)
        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        n_features = X_scaled.shape[1]
        
        self.weights = np.random.randn(n_features, self.n_components) / self.lengthscale
        self.bias = np.random.uniform(0, 2 * np.pi, self.n_components)
        
        Z = self._transform(X_scaled)
        
        ZTZ = Z.T @ Z
        lambda_reg = self.sigma_noise ** 2
        ZTZ_reg = ZTZ + lambda_reg * np.eye(self.n_components)
        
        self.alpha = np.linalg.solve(ZTZ_reg, Z.T @ y_scaled)
        self.covariance = lambda_reg * np.linalg.inv(ZTZ_reg)
        
        self.training_time = time.time() - start_time
        
        return self
    
    def predict(self, X_test, return_std=True):
        """Make predictions with uncertainty"""
        X_scaled = self.scaler_X.transform(X_test)
        Z_test = self._transform(X_scaled)
        
        # Mean prediction
        y_pred_scaled = Z_test @ self.alpha
        
        # Transform back to original scale
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        if return_std:
            pred_var_scaled = np.array([
                self.sigma_noise**2 + Z_test[i] @ self.covariance @ Z_test[i]
                for i in range(len(Z_test))
            ])
            pred_std = np.sqrt(pred_var_scaled) * self.scaler_y.scale_
            return y_pred, pred_std
        
        return y_pred
    
    def evaluate(self, X_test, y_test, verbose=False):
        """Comprehensive evaluation"""
        start_time = time.time()
        
        y_pred, y_std = self.predict(X_test, return_std=True)
        
        inference_time = time.time() - start_time
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nll = self._negative_log_likelihood(y_test, y_pred, y_std)
        ece = self._expected_calibration_error(y_test, y_pred, y_std)
        
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)
        
        metrics = {
            'rmse': rmse,
            'nll': nll,
            'ece': ece,
            'mae': mae,
            'r2': r2,
            'training_time': self.training_time,
            'inference_time': inference_time,
            'inference_time_per_sample': inference_time / len(X_test),
            'mean_std': y_std.mean(),
            'std_std': y_std.std()
        }
        
        return metrics
    
    def _negative_log_likelihood(self, y_true, y_pred, y_std):
        """Compute NLL assuming Gaussian likelihood"""
        var = y_std ** 2
        nll = 0.5 * np.log(2 * np.pi * var) + (y_true - y_pred)**2 / (2 * var)
        return np.mean(nll)
    
    def _expected_calibration_error(self, y_true, y_pred, y_std, n_bins=10):
        """Compute ECE for regression"""
        errors = np.abs(y_true - y_pred)
        confidence_levels = np.linspace(0.1, 0.9, n_bins)
        
        ece = 0.0
        for conf in confidence_levels:
            z_score = norm.ppf((1 + conf) / 2)
            predicted_interval = y_std * z_score
            within_interval = (errors <= predicted_interval).astype(float)
            observed_conf = within_interval.mean()
            ece += np.abs(observed_conf - conf) / n_bins
        
        return ece


class NystromApproximation:
    """Nyström approximation with uncertainty"""
    
    def __init__(self, n_landmarks=500, lengthscale=1.0, sigma_noise=0.1, random_state=42):
        """
        Args:
            n_landmarks: Number of landmark/inducing points (m)
            lengthscale: RBF kernel lengthscale
            sigma_noise: Observation noise
            random_state: Random seed
        """
        self.n_landmarks = n_landmarks
        self.lengthscale = lengthscale
        self.sigma_noise = sigma_noise
        self.random_state = random_state
        
        self.landmarks = None    # Inducing points
        self.alpha = None        # Weights
        self.K_mm_inv = None     # Inverse of K(landmarks, landmarks)
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.training_time = 0
        
        np.random.seed(random_state)
    
    def _rbf_kernel(self, X1, X2):
        """RBF kernel: k(x, x') = exp(-||x - x'||^2 / (2 * lengthscale^2))"""
        # Compute pairwise squared distances
        X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
        distances_sq = X1_norm + X2_norm - 2 * X1 @ X2.T
        
        # RBF kernel
        return np.exp(-distances_sq / (2 * self.lengthscale**2))
    
    def fit(self, X_train, y_train):
        """Train Nyström model"""
        start_time = time.time()
        
        X_scaled = self.scaler_X.fit_transform(X_train)
        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        n_samples = X_scaled.shape[0]
        
        if self.n_landmarks < n_samples:
            landmark_indices = np.random.choice(n_samples, self.n_landmarks, replace=False)
            self.landmarks = X_scaled[landmark_indices]
        else:
            self.landmarks = X_scaled.copy()
            self.n_landmarks = n_samples
        
        K_mm = self._rbf_kernel(self.landmarks, self.landmarks)
        K_nm = self._rbf_kernel(X_scaled, self.landmarks)
        
        K_mm_stable = K_mm + 1e-5 * np.eye(self.n_landmarks)
        self.K_mm_inv = np.linalg.inv(K_mm_stable)
        
        A = K_mm_stable + (1.0 / self.sigma_noise**2) * (K_nm.T @ K_nm)
        A_inv = np.linalg.inv(A)
        self.alpha = (1.0 / self.sigma_noise**2) * (A_inv @ K_nm.T @ y_scaled)
        
        self.sigma_y = y_scaled.std()
        
        self.training_time = time.time() - start_time
        
        return self
    
    def predict(self, X_test, return_std=True):
        """Make predictions with uncertainty"""
        X_scaled = self.scaler_X.transform(X_test)
        
        # Compute K(X_test, landmarks)
        K_tm = self._rbf_kernel(X_scaled, self.landmarks)  # n_test x m
        
        # Mean prediction: f* = K_tm alpha
        y_pred_scaled = K_tm @ self.alpha
        
        # Transform back
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        if return_std:
            k_star_star = 1.0
            variance_reduction = np.array([
                K_tm[i] @ self.K_mm_inv @ K_tm[i]
                for i in range(len(K_tm))
            ])
            pred_var_scaled = k_star_star - variance_reduction + self.sigma_noise**2
            min_var = self.sigma_noise**2
            max_var = 3.0
            pred_var_scaled = np.clip(pred_var_scaled, min_var, max_var)
            pred_std = np.sqrt(pred_var_scaled) * self.scaler_y.scale_
            return y_pred, pred_std
        
        return y_pred
    
    def evaluate(self, X_test, y_test, verbose=False):
        """Comprehensive evaluation"""
        start_time = time.time()
        
        y_pred, y_std = self.predict(X_test, return_std=True)
        
        inference_time = time.time() - start_time
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nll = self._negative_log_likelihood(y_test, y_pred, y_std)
        ece = self._expected_calibration_error(y_test, y_pred, y_std)
        
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)
        
        metrics = {
            'rmse': rmse,
            'nll': nll,
            'ece': ece,
            'mae': mae,
            'r2': r2,
            'training_time': self.training_time,
            'inference_time': inference_time,
            'inference_time_per_sample': inference_time / len(X_test),
            'mean_std': y_std.mean(),
            'std_std': y_std.std()
        }
        
        return metrics
    
    def _negative_log_likelihood(self, y_true, y_pred, y_std):
        """Compute NLL"""
        var = y_std ** 2
        nll = 0.5 * np.log(2 * np.pi * var) + (y_true - y_pred)**2 / (2 * var)
        return np.mean(nll)
    
    def _expected_calibration_error(self, y_true, y_pred, y_std, n_bins=10):
        """Compute ECE"""
        errors = np.abs(y_true - y_pred)
        confidence_levels = np.linspace(0.1, 0.9, n_bins)
        ece = 0.0
        for conf in confidence_levels:
            z_score = norm.ppf((1 + conf) / 2)
            predicted_interval = y_std * z_score
            within_interval = (errors <= predicted_interval).astype(float)
            observed_conf = within_interval.mean()
            ece += np.abs(observed_conf - conf) / n_bins
        return ece