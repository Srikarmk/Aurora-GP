"""
AURORA - Kernel Approximation Methods
Random Fourier Features and Nyström with proper uncertainty estimates
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time
from scipy.stats import norm
from scipy.spatial.distance import pdist


class RandomFourierFeatures:
    """Random Fourier Features approximation with uncertainty"""
    
    def __init__(self, n_components=1000, lengthscale=1.0, sigma_noise=0.1, random_state=42):
        self.n_components = n_components
        self.lengthscale = lengthscale
        self.sigma_noise = sigma_noise
        self.random_state = random_state
        
        self.weights = None
        self.bias = None
        self.alpha = None
        self.covariance = None
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.training_time = 0
        
        np.random.seed(random_state)
    
    def _transform(self, X):
        """Transform input to random feature space"""
        projection = X @ self.weights + self.bias
        Z = np.sqrt(2.0 / self.n_components) * np.cos(projection)
        return Z
    
    def fit(self, X_train, y_train):
        """Train RFF model"""
        start_time = time.time()
        
        # Standardize
        X_scaled = self.scaler_X.fit_transform(X_train)
        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        n_features = X_scaled.shape[1]
        
        # Sample random weights
        self.weights = np.random.randn(n_features, self.n_components) / self.lengthscale
        self.bias = np.random.uniform(0, 2 * np.pi, self.n_components)
        
        # Transform to feature space
        Z = self._transform(X_scaled)
        
        # Bayesian linear regression
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
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        if return_std:
            # Predictive variance
            pred_var_scaled = np.array([
                self.sigma_noise**2 + Z_test[i] @ self.covariance @ Z_test[i]
                for i in range(len(Z_test))
            ])
            
            pred_std = np.sqrt(pred_var_scaled) * self.scaler_y.scale_
            
            return y_pred, pred_std
        
        return y_pred
    
    def evaluate(self, X_test, y_test, verbose=True):
        """Comprehensive evaluation"""
        start_time = time.time()
        
        y_pred, y_std = self.predict(X_test, return_std=True)
        
        inference_time = time.time() - start_time
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nll = self._negative_log_likelihood(y_test, y_pred, y_std)
        ece = self._expected_calibration_error(y_test, y_pred, y_std)
        
        metrics = {
            'rmse': rmse,
            'nll': nll,
            'ece': ece,
            'training_time': self.training_time,
            'inference_time': inference_time
        }
        
        if verbose:
            print(f"\nRFF EVALUATION (D={self.n_components})")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  NLL:  {nll:.4f}")
            print(f"  ECE:  {ece:.4f}")
        
        return metrics
    
    def _negative_log_likelihood(self, y_true, y_pred, y_std):
        var = y_std ** 2
        nll = 0.5 * np.log(2 * np.pi * var) + (y_true - y_pred)**2 / (2 * var)
        return np.mean(nll)
    
    def _expected_calibration_error(self, y_true, y_pred, y_std, n_bins=10):
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
        self.n_landmarks = n_landmarks
        self.lengthscale = lengthscale
        self.sigma_noise = sigma_noise
        self.random_state = random_state
        
        self.landmarks = None
        self.alpha = None
        self.K_mm_inv = None
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.training_time = 0
        
        np.random.seed(random_state)
    
    def _rbf_kernel(self, X1, X2):
        """RBF kernel"""
        X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
        distances_sq = X1_norm + X2_norm - 2 * X1 @ X2.T
        return np.exp(-distances_sq / (2 * self.lengthscale**2))
    
    def fit(self, X_train, y_train):
        """Train Nyström model"""
        start_time = time.time()
        
        X_scaled = self.scaler_X.fit_transform(X_train)
        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        n_samples = X_scaled.shape[0]
        
        # Select landmarks
        if self.n_landmarks < n_samples:
            landmark_indices = np.random.choice(n_samples, self.n_landmarks, replace=False)
            self.landmarks = X_scaled[landmark_indices]
        else:
            self.landmarks = X_scaled.copy()
            self.n_landmarks = n_samples
        
        # Compute kernel matrices
        K_mm = self._rbf_kernel(self.landmarks, self.landmarks)
        K_nm = self._rbf_kernel(X_scaled, self.landmarks)
        
        K_mm_stable = K_mm + 1e-5 * np.eye(self.n_landmarks)
        
        self.K_mm_inv = np.linalg.inv(K_mm_stable)
        
        # Solve for weights
        A = K_mm_stable + (1.0 / self.sigma_noise**2) * (K_nm.T @ K_nm)
        A_inv = np.linalg.inv(A)
        self.alpha = (1.0 / self.sigma_noise**2) * (A_inv @ K_nm.T @ y_scaled)
        
        self.training_time = time.time() - start_time
        
        return self
    
    def predict(self, X_test, return_std=True):
        """Make predictions with uncertainty"""
        X_scaled = self.scaler_X.transform(X_test)
        
        K_tm = self._rbf_kernel(X_scaled, self.landmarks)
        
        y_pred_scaled = K_tm @ self.alpha
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        if return_std:
            k_star_star = 1.0
            
            variance_reduction = np.array([
                K_tm[i] @ self.K_mm_inv @ K_tm[i]
                for i in range(len(K_tm))
            ])
            
            pred_var_scaled = k_star_star - variance_reduction + self.sigma_noise**2
            pred_var_scaled = np.clip(pred_var_scaled, self.sigma_noise**2, 3.0)
            
            pred_std = np.sqrt(pred_var_scaled) * self.scaler_y.scale_
            
            return y_pred, pred_std
        
        return y_pred
    
    def evaluate(self, X_test, y_test, verbose=True):
        """Comprehensive evaluation"""
        start_time = time.time()
        
        y_pred, y_std = self.predict(X_test, return_std=True)
        
        inference_time = time.time() - start_time
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nll = self._negative_log_likelihood(y_test, y_pred, y_std)
        ece = self._expected_calibration_error(y_test, y_pred, y_std)
        
        metrics = {
            'rmse': rmse,
            'nll': nll,
            'ece': ece,
            'training_time': self.training_time,
            'inference_time': inference_time
        }
        
        
        return metrics
    
    def _negative_log_likelihood(self, y_true, y_pred, y_std):
        var = y_std ** 2
        nll = 0.5 * np.log(2 * np.pi * var) + (y_true - y_pred)**2 / (2 * var)
        return np.mean(nll)
    
    def _expected_calibration_error(self, y_true, y_pred, y_std, n_bins=10):
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