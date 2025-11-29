import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
from scipy.stats import norm
from pathlib import Path
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).parent.parent


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


def load_baseline_hyperparameters(dataset_name):
    """Load hyperparameters from baseline GP results if available"""
    baseline_path = PROJECT_ROOT / 'results' / 'baseline_gp' / 'summary.json'
    
    if not baseline_path.exists():
        return {'lengthscale': 1.0, 'sigma_noise': 0.1}
    
    try:
        with open(baseline_path, 'r') as f:
            baseline_results = json.load(f)
        
        if dataset_name in baseline_results:
            metrics = baseline_results[dataset_name]
            # Try to extract hyperparameters if available
            # Default values if not found
            lengthscale = metrics.get('lengthscale', 1.0)
            sigma_noise = metrics.get('sigma_noise', 0.1)
            return {'lengthscale': lengthscale, 'sigma_noise': sigma_noise}
    except Exception:
        pass
    
    return {'lengthscale': 1.0, 'sigma_noise': 0.1}


def run_approximation_experiments(data_dir=None, results_dir=None):
    """Run RFF and Nyström approximations on all datasets"""
    if data_dir is None:
        data_dir = PROJECT_ROOT / 'data'
    if results_dir is None:
        results_dir = PROJECT_ROOT / 'results' / 'approximations'
    
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("APPROXIMATION METHODS BENCHMARK")
    print("="*70)
    print("\nMethods:")
    print("  - Random Fourier Features (RFF)")
    print("  - Nyström Approximation")
    print("="*70)
    
    datasets = list(data_dir.glob('*.npz'))
    
    if not datasets:
        print(f"\n[ERROR] No datasets found in {data_dir}")
        return None
    
    all_results = {}
    
    for dataset_file in datasets:
        dataset_name = dataset_file.stem
        
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        try:
            # Load data
            data = np.load(dataset_file, allow_pickle=True)
            X = data['X']
            y = data['y']
            
            # Subsample large datasets
            if len(X) > 10000:
                print(f"  Subsampling from {len(X)} to 10000")
                idx = np.random.RandomState(42).choice(len(X), 10000, replace=False)
                X = X[idx]
                y = y[idx]
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Load hyperparameters
            hyperparams = load_baseline_hyperparameters(dataset_name)
            print(f"  Lengthscale: {hyperparams['lengthscale']:.4f}")
            print(f"  Sigma noise: {hyperparams['sigma_noise']:.4f}")
            
            dataset_results = {}
            
            # Run Random Fourier Features
            print("\n  Training Random Fourier Features...")
            rff = RandomFourierFeatures(
                n_components=1000,
                lengthscale=hyperparams['lengthscale'],
                sigma_noise=hyperparams['sigma_noise'],
                random_state=42
            )
            rff.fit(X_train, y_train)
            rff_metrics = rff.evaluate(X_test, y_test, verbose=True)
            rff_metrics['lengthscale'] = hyperparams['lengthscale']
            rff_metrics['sigma_noise'] = hyperparams['sigma_noise']
            dataset_results['rff'] = rff_metrics
            
            # Run Nyström Approximation
            print("\n  Training Nyström Approximation...")
            nystrom = NystromApproximation(
                n_landmarks=500,
                lengthscale=hyperparams['lengthscale'],
                sigma_noise=hyperparams['sigma_noise'],
                random_state=42
            )
            nystrom.fit(X_train, y_train)
            nystrom_metrics = nystrom.evaluate(X_test, y_test, verbose=True)
            nystrom_metrics['lengthscale'] = hyperparams['lengthscale']
            nystrom_metrics['sigma_noise'] = hyperparams['sigma_noise']
            dataset_results['nystrom'] = nystrom_metrics
            
            # Save results
            all_results[dataset_name] = dataset_results
            
            # Save per-dataset results
            dataset_dir = results_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            rff_dir = dataset_dir / 'rff'
            rff_dir.mkdir(parents=True, exist_ok=True)
            with open(rff_dir / 'metrics.json', 'w') as f:
                json.dump(rff_metrics, f, indent=2)
            
            nystrom_dir = dataset_dir / 'nystrom'
            nystrom_dir.mkdir(parents=True, exist_ok=True)
            with open(nystrom_dir / 'metrics.json', 'w') as f:
                json.dump(nystrom_metrics, f, indent=2)
            
            print(f"\n✓ {dataset_name} complete")
            
        except Exception as e:
            print(f"\n[ERROR] {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {'error': str(e)}
    
    # Save summary
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("APPROXIMATION METHODS SUMMARY")
    print("="*70)
    print(f"{'Dataset':<20} {'Method':<15} {'RMSE':<10} {'NLL':<10} {'ECE':<10} {'Time(s)':<10}")
    print("-"*70)
    
    for name, results in all_results.items():
        if 'error' not in results:
            if 'rff' in results:
                rff = results['rff']
                print(f"{name:<20} {'RFF':<15} {rff['rmse']:<10.4f} {rff['nll']:<10.4f} "
                      f"{rff['ece']:<10.4f} {rff['training_time']:<10.2f}")
            if 'nystrom' in results:
                nys = results['nystrom']
                print(f"{name:<20} {'Nyström':<15} {nys['rmse']:<10.4f} {nys['nll']:<10.4f} "
                      f"{nys['ece']:<10.4f} {nys['training_time']:<10.2f}")
    
    print("="*70)
    print(f"\n✓ All results saved to {results_dir}")
    
    return all_results


if __name__ == "__main__":
    results = run_approximation_experiments()