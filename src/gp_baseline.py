import numpy as np
import torch
import gpytorch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time
import warnings
from pathlib import Path
import json
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
warnings.filterwarnings('ignore')


class ExactGPModel(gpytorch.models.ExactGP):
    """GPyTorch Exact GP with RBF kernel"""
    
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessBaseline:
    """Exact Gaussian Process baseline for regression
    
    Uses GPyTorch for scalable GP implementation with GPU support.
    Handles data preprocessing, training, prediction, and evaluation.
    """
    
    def __init__(self, max_train_size=5000, random_state=42, use_gpu=True):
        """
        Args:
            max_train_size: Maximum training samples (for computational feasibility)
            random_state: Random seed for reproducibility
            use_gpu: Whether to use GPU if available
        """
        self.max_train_size = max_train_size
        self.random_state = random_state
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.likelihood = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.train_x = None
        self.train_y = None
        
        self.training_time = 0
        self.n_train = 0
        
        print(f"Using device: {self.device}")
    
    def _prepare_data(self, X, y, subset=True):
        """Preprocess and optionally subsample data"""
        # Subsample if needed
        if subset and len(X) > self.max_train_size:
            print(f"WARNING: Subsampling from {len(X)} to {self.max_train_size} points")
            idx = np.random.RandomState(self.random_state).choice(
                len(X), self.max_train_size, replace=False
            )
            X = X[idx]
            y = y[idx]
        
        # Standardize
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Convert to torch tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)
        
        return X_tensor, y_tensor
    
    def fit(self, X_train, y_train, n_iter=50):
        """Train the GP model
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples,)
            n_iter: Number of optimization iterations
        """
        print("\n" + "="*60)
        print("TRAINING EXACT GP")
        print("="*60)
        
        start_time = time.time()
        
        # Prepare data
        self.train_x, self.train_y = self._prepare_data(X_train, y_train, subset=True)
        self.n_train = len(self.train_x)
        
        print(f"Training samples: {self.n_train}")
        print(f"Features: {self.train_x.shape[1]}")
        
        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood).to(self.device)
        
        # Training mode
        self.model.train()
        self.likelihood.train()
        
        # Use Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        
        # Loss function - marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        print(f"\nOptimizing hyperparameters ({n_iter} iterations)...")
        
        # Training loop with progress
        losses = []
        for i in range(n_iter):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if (i + 1) % 10 == 0:
                print(f"  Iter {i+1}/{n_iter} - Loss: {loss.item():.4f}")
        
        self.training_time = time.time() - start_time
        
        print(f"\n[OK] Training complete in {self.training_time:.2f}s")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Lengthscale: {self.model.covar_module.base_kernel.lengthscale.item():.4f}")
        print(f"  Noise: {self.likelihood.noise.item():.4f}")
        
        return self
    
    def predict(self, X_test, return_std=True):
        """Make predictions with uncertainty
        
        Args:
            X_test: Test features (n_samples, n_features)
            return_std: Whether to return predictive standard deviation
            
        Returns:
            mean: Predicted means
            std: Predictive standard deviations (if return_std=True)
        """
        # Prepare test data
        X_scaled = self.scaler_X.transform(X_test)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # Evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self.model(X_tensor))
            
            # Get mean and variance
            mean_scaled = pred_dist.mean.cpu().numpy()
            var_scaled = pred_dist.variance.cpu().numpy()
        
        # Inverse transform to original scale
        mean = self.scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
        
        if return_std:
            # Transform variance to original scale
            std = np.sqrt(var_scaled) * self.scaler_y.scale_
            return mean, std
        else:
            return mean
    
    def evaluate(self, X_test, y_test, verbose=True):
        """Comprehensive evaluation with multiple metrics
        
        Args:
            X_test: Test features
            y_test: Test targets
            verbose: Whether to print results
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        start_time = time.time()
        
        # Get predictions
        y_pred, y_std = self.predict(X_test, return_std=True)
        
        inference_time = time.time() - start_time
        
        # Compute metrics
        metrics = self._compute_metrics(y_test, y_pred, y_std)
        metrics['inference_time'] = inference_time
        metrics['inference_time_per_sample'] = inference_time / len(X_test)
        
        if verbose:
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            print(f"RMSE:  {metrics['rmse']:.4f}")
            print(f"NLL:   {metrics['nll']:.4f}")
            print(f"ECE:   {metrics['ece']:.4f}")
            print(f"Inference time: {inference_time:.3f}s ({metrics['inference_time_per_sample']*1000:.2f}ms/sample)")
            print("="*60)
        
        return metrics
    
    def _compute_metrics(self, y_true, y_pred, y_std):
        """Compute accuracy and calibration metrics"""
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Negative Log-Likelihood
        nll = self._negative_log_likelihood(y_true, y_pred, y_std)
        
        # Expected Calibration Error
        ece = self._expected_calibration_error(y_true, y_pred, y_std)
        
        # Additional metrics
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
        
        return {
            'rmse': rmse,
            'nll': nll,
            'ece': ece,
            'mae': mae,
            'r2': r2,
            'mean_std': y_std.mean(),
            'std_std': y_std.std()
        }
    
    def _negative_log_likelihood(self, y_true, y_pred, y_std):
        """Compute negative log-likelihood (Gaussian)"""
        # NLL = -log p(y | μ, σ²) = 0.5 * log(2πσ²) + (y - μ)² / (2σ²)
        var = y_std ** 2
        nll = 0.5 * np.log(2 * np.pi * var) + (y_true - y_pred)**2 / (2 * var)
        return np.mean(nll)
    
    def _expected_calibration_error(self, y_true, y_pred, y_std, n_bins=10):
        """Compute Expected Calibration Error for regression
        
        For each confidence level, check if true errors fall within predicted intervals.
        """
        errors = np.abs(y_true - y_pred)
        
        # Use different confidence levels (z-scores)
        confidence_levels = np.linspace(0.1, 0.9, n_bins)
        
        ece = 0.0
        for conf in confidence_levels:
            # Predicted interval based on confidence level
            z_score = np.abs(np.percentile(np.random.randn(10000), [(1-conf)*100/2, (1+conf)*100/2]))
            predicted_interval = y_std * z_score[1]
            
            # Fraction of points within predicted interval
            within_interval = (errors <= predicted_interval).astype(float)
            observed_conf = within_interval.mean()
            
            # Calibration error for this confidence level
            ece += np.abs(observed_conf - conf) / n_bins
        
        return ece
    
    def get_uncertainty_map(self, X):
        """Get uncertainty estimates across input space
        
        This is used in Stage 1 for region identification.
        
        Args:
            X: Points to evaluate (n_samples, n_features)
            
        Returns:
            uncertainties: Predictive standard deviations
        """
        _, std = self.predict(X, return_std=True)
        return std
    
    def save(self, path):
        """Save model to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model state (PyTorch tensors and state dicts)
        torch.save({
            'model_state': self.model.state_dict(),
            'likelihood_state': self.likelihood.state_dict(),
            'train_x': self.train_x,
            'train_y': self.train_y,
        }, path / 'gp_model.pth')
        
        # Save scalers separately using joblib (StandardScaler objects)
        joblib.dump(self.scaler_X, path / 'scaler_X.pkl')
        joblib.dump(self.scaler_y, path / 'scaler_y.pkl')
        
        print(f"[OK] Model saved to {path}")
    
    def load(self, path):
        """Load model from disk"""
        path = Path(path)
        checkpoint = torch.load(path / 'gp_model.pth')
        
        self.train_x = checkpoint['train_x']
        self.train_y = checkpoint['train_y']
        
        # Load scalers using joblib
        self.scaler_X = joblib.load(path / 'scaler_X.pkl')
        self.scaler_y = joblib.load(path / 'scaler_y.pkl')
        
        # Recreate model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state'])
        
        print(f"[OK] Model loaded from {path}")


def run_baseline_experiments(data_dir=None, results_dir=None):
    """Run GP baseline on all datasets"""
    if data_dir is None:
        data_dir = PROJECT_ROOT / 'data'
    if results_dir is None:
        results_dir = PROJECT_ROOT / 'results' / 'baseline_gp'
    
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("RUNNING EXACT GP BASELINE EXPERIMENTS")
    print("="*70)
    
    # Load all datasets
    datasets = {}
    npz_files = list(data_dir.glob('*.npz'))
    
    if not npz_files:
        print(f"\n[ERROR] No .npz files found in {data_dir.absolute()}")
        print(f"Please check that the data directory exists and contains .npz files.")
        return {}
    
    print(f"\nFound {len(npz_files)} dataset(s):")
    for npz_file in npz_files:
        print(f"  - {npz_file.name}")
    
    for npz_file in npz_files:
        name = npz_file.stem
        print(f"\nLoading {name}...")
        data = np.load(npz_file, allow_pickle=True)
        datasets[name] = {
            'X': data['X'],
            'y': data['y'],
            'feature_names': data['feature_names'].tolist()
        }
        print(f"  Loaded: X.shape={datasets[name]['X'].shape}, y.shape={datasets[name]['y'].shape}")
    
    if not datasets:
        print("\n[ERROR] No datasets were successfully loaded!")
        return {}
    
    all_results = {}
    
    for name, data in datasets.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {name.upper()}")
        print(f"{'='*70}")
        
        X, y = data['X'], data['y']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Total samples: {len(X)}")
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"Features: {X.shape[1]}")
        
        # Train GP
        gp = GaussianProcessBaseline(max_train_size=5000, random_state=42)
        gp.fit(X_train, y_train, n_iter=50)
        
        # Evaluate
        metrics = gp.evaluate(X_test, y_test, verbose=True)
        
        # Add training info
        metrics['training_time'] = gp.training_time
        metrics['n_train'] = gp.n_train
        metrics['n_test'] = len(X_test)
        
        # Save results
        all_results[name] = metrics
        
        # Save model
        model_dir = results_dir / name
        gp.save(model_dir)
        
        # Save metrics
        with open(model_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Save summary
    print("\n" + "="*70)
    print("BASELINE RESULTS SUMMARY")
    print("="*70)
    print(f"{'Dataset':<20} {'RMSE':<10} {'NLL':<10} {'ECE':<10} {'Time(s)':<10}")
    print("-"*70)
    
    for name, metrics in all_results.items():
        print(f"{name:<20} {metrics['rmse']:<10.4f} {metrics['nll']:<10.4f} "
              f"{metrics['ece']:<10.4f} {metrics['training_time']:<10.2f}")
    
    # Save summary to file
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[OK] All results saved to {results_dir}")
    
    return all_results


if __name__ == "__main__":
    # Run experiments
    results = run_baseline_experiments()
    
    print("\n[OK] Exact GP baseline experiments complete!")