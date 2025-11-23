import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import urllib.request
import os
from scipy.io import loadmat

class DatasetCollector:
    """Collect and prepare datasets for kernel approximation experiments"""
    
    def __init__(self, data_dir='./data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.datasets = {}
    
    # # def load_uci_energy(self):
    #     """Energy Efficiency Dataset - 768 samples, 8 features"""
    #     print("Loading Energy Efficiency dataset...")
    #     try:
    #         data = fetch_openml('energy-efficiency', version=1, parser='auto')
    #         X = pd.DataFrame(data.data, columns=data.feature_names)
    #         # Use heating load as target
    #         y = pd.Series(data.target[0] if isinstance(data.target, list) else data.target)
            
    #         self.datasets['energy'] = {
    #             'X': X.values,
    #             'y': y.values,
    #             'feature_names': list(X.columns),
    #             'description': 'Energy Efficiency - Building heating load prediction'
    #         }
    #         print(f"[OK] Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    #         return True
    #     except Exception as e:
    #         print(f"[ERROR] Error loading Energy dataset: {e}")
    #         return False
    
    def load_uci_concrete(self):
        """Concrete Compressive Strength - 1030 samples, 8 features"""
        print("\nLoading Concrete Strength dataset...")
        try:
            # Download directly from UCI repository (faster and more reliable)
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
            excel_path = self.data_dir / "Concrete_Data.xls"
            
            if not excel_path.exists():
                print("Downloading from UCI repository...")
                urllib.request.urlretrieve(url, excel_path)
            else:
                print("Using cached file...")
            
            df = pd.read_excel(excel_path)
            # Last column is target (Concrete compressive strength)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            # Ensure X and y are numeric
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            
            feature_names = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 
                           'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']
            
            self.datasets['concrete'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'description': 'Concrete Strength - Compressive strength prediction'
            }
            print(f"[OK] Loaded: {X.shape[0]} samples, {X.shape[1]} features")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading Concrete dataset: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_openml_robot_arm(self):
        """Robot Arm kin8nm - 8192 samples, 9 features"""
        print("\nLoading Robot Arm (kin8nm) dataset...")
        try:
            data = fetch_openml('kin8nm', version=1, parser='auto')
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = pd.Series(data.target)
            # Ensure y is numeric
            y = pd.to_numeric(y, errors='coerce')
            
            self.datasets['robot_arm'] = {
                'X': np.asarray(X.values, dtype=np.float64),
                'y': np.asarray(y.values, dtype=np.float64),
                'feature_names': list(X.columns),
                'description': 'Robot Arm kin8nm - Highly non-linear kinematics'
            }
            print(f"[OK] Loaded: {X.shape[0]} samples, {X.shape[1]} features")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading Robot Arm dataset: {e}")
            return False
    
    def load_protein_structure(self):
        """Protein Tertiary Structure - ~45730 samples, 9 features"""
        print("\nLoading Protein Structure dataset...")
        try:
            # Download from UCI repository
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"
            csv_path = self.data_dir / "CASP.csv"
            
            if not csv_path.exists():
                print("Downloading...")
                urllib.request.urlretrieve(url, csv_path)
            
            df = pd.read_csv(csv_path)
            X = df.iloc[:, 1:].values  # Skip first column (protein ID)
            y = df.iloc[:, 0].values   # RMSD is first column
            
            # Ensure X and y are numeric numpy arrays
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            
            feature_names = [f"F{i}" for i in range(1, X.shape[1]+1)]
            
            self.datasets['protein'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'description': 'Protein Structure - RMSD prediction from physicochemical properties'
            }
            print(f"[OK] Loaded: {X.shape[0]} samples, {X.shape[1]} features")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading Protein dataset: {e}")
            return False
    
    def load_sarcos_robot(self):
        """SARCOS Robot Arm - 44484 training samples, 21 features"""
        print("\nLoading SARCOS Robot dataset...")
        try:
            # Download from GPML website
            train_url = "http://gaussianprocess.org/gpml/data/sarcos_inv.mat"
            test_url = "http://gaussianprocess.org/gpml/data/sarcos_inv_test.mat"
            
            train_path = self.data_dir / "sarcos_inv.mat"
            test_path = self.data_dir / "sarcos_inv_test.mat"
            
            if not train_path.exists():
                print("Downloading training data...")
                urllib.request.urlretrieve(train_url, train_path)
            
            if not test_path.exists():
                print("Downloading test data...")
                urllib.request.urlretrieve(test_url, test_path)
            
            train_data = loadmat(train_path)['sarcos_inv']
            test_data = loadmat(test_path)['sarcos_inv_test']
            
            # Combine train and test
            all_data = np.vstack([train_data, test_data])
            X = all_data[:, :21]  # First 21 columns are inputs
            y = all_data[:, 21]   # 22nd column is first torque
            
            # Ensure X and y are numeric numpy arrays
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            
            feature_names = [f"joint_{i//3}_{['pos','vel','acc'][i%3]}" 
                           for i in range(21)]
            
            self.datasets['sarcos'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'description': 'SARCOS Robot - Inverse dynamics (largest dataset)'
            }
            print(f"[OK] Loaded: {X.shape[0]} samples, {X.shape[1]} features")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading SARCOS dataset: {e}")
            return False
    
    # # def generate_synthetic_nonuniform(self, n_samples=5000, noise_level=0.1):
    #     """Generate synthetic data with non-uniform distribution"""
    #     print("\nGenerating Synthetic Non-uniform dataset...")
        
    #     # Create clusters with different densities
    #     cluster1 = np.random.randn(int(n_samples * 0.5), 2) * 0.5 + np.array([0, 0])
    #     cluster2 = np.random.randn(int(n_samples * 0.3), 2) * 0.3 + np.array([3, 3])
    #     cluster3 = np.random.randn(int(n_samples * 0.2), 2) * 0.8 + np.array([-2, 2])
        
    #     X = np.vstack([cluster1, cluster2, cluster3])
        
    #     # Non-linear function with varying smoothness
    #     y = (np.sin(2 * X[:, 0]) + 
    #          0.5 * np.cos(3 * X[:, 1]) + 
    #          0.3 * X[:, 0] * X[:, 1] +
    #          np.random.randn(len(X)) * noise_level)
        
    #     self.datasets['synthetic_nonuniform'] = {
    #         'X': X,
    #         'y': y,
    #         'feature_names': ['X1', 'X2'],
    #         'description': 'Synthetic - Non-uniform data distribution with clusters'
    #     }
    #     print(f"[OK] Generated: {X.shape[0]} samples, {X.shape[1]} features")
    #     return True
    
    def generate_synthetic_heteroscedastic(self, n_samples=5000):
        """Generate synthetic data with heteroscedastic noise (varying noise levels)"""
        print("\nGenerating Synthetic Heteroscedastic dataset...")
        
        X = np.random.uniform(-5, 5, (n_samples, 2))
        
        # Function value
        f = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1])
        
        # Heteroscedastic noise: noise increases with distance from origin
        distance = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        noise_std = 0.05 + 0.15 * (distance / distance.max())
        noise = np.random.randn(n_samples) * noise_std
        
        y = f + noise
        
        # Ensure X and y are numeric numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        self.datasets['synthetic_heteroscedastic'] = {
            'X': X,
            'y': y,
            'feature_names': ['X1', 'X2'],
            'description': 'Synthetic - Heteroscedastic noise (varying uncertainty)',
            'noise_std': noise_std  # Store for analysis
        }
        print(f"[OK] Generated: {X.shape[0]} samples, {X.shape[1]} features")
        return True
    
    def collect_all(self, option='focused'):
        """Collect all datasets
        
        Args:
            option: 'focused' (5 datasets) or 'comprehensive' (7 datasets)
        """
        print("COLLECTING DATASETS FOR AURORA")
        self.generate_synthetic_heteroscedastic()
        self.load_uci_concrete()           # 1K
        self.load_openml_robot_arm()       # 8K
        self.load_protein_structure()      # 46K
        self.load_sarcos_robot()           # 44K
        
        return self.datasets
    
    def visualize_datasets(self):
        """Visualize the collected datasets"""
        n_datasets = len(self.datasets)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (name, data) in enumerate(self.datasets.items()):
            ax = axes[idx]
            X, y = data['X'], data['y']
            
            if X.shape[1] == 2:
                # 2D scatter plot
                scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                                   alpha=0.6, s=20)
                ax.set_xlabel(data['feature_names'][0])
                ax.set_ylabel(data['feature_names'][1])
                plt.colorbar(scatter, ax=ax, label='Target')
            else:
                # For higher dimensions, show first two features
                scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                                   alpha=0.6, s=20)
                ax.set_xlabel(data['feature_names'][0])
                ax.set_ylabel(data['feature_names'][1])
                plt.colorbar(scatter, ax=ax, label='Target')
            
            ax.set_title(f"{name}\n{data['description']}")
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_datasets, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'dataset_visualizations.png', dpi=150, bbox_inches='tight')
        print(f"\n[OK] Visualization saved to {self.data_dir / 'dataset_visualizations.png'}")
        return fig
    
    def get_dataset_statistics(self):
        """Print statistics for all datasets"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        stats = []
        for name, data in self.datasets.items():
            X, y = data['X'], data['y']
            # Ensure y is a numeric numpy array
            y = np.asarray(y, dtype=np.float64)
            stats.append({
                'Dataset': name,
                'Samples': X.shape[0],
                'Features': X.shape[1],
                'Target Mean': f"{y.mean():.2f}",
                'Target Std': f"{y.std():.2f}",
                'Target Range': f"[{y.min():.2f}, {y.max():.2f}]"
            })
        
        stats_df = pd.DataFrame(stats)
        print(stats_df.to_string(index=False))
        print("="*60)
        
        return stats_df
    
    def save_datasets(self):
        """Save all datasets to disk"""
        print("\nSaving datasets...")
        for name, data in self.datasets.items():
            np.savez(
                self.data_dir / f'{name}.npz',
                X=data['X'],
                y=data['y'],
                feature_names=data['feature_names']
            )
        print(f"[OK] All datasets saved to {self.data_dir}")


if __name__ == "__main__":
    # Collect datasets
    collector = DatasetCollector()
    
    # # Choose collection option
    # print("Choose dataset collection:")
    # print("1. FOCUSED (5 datasets) - Recommended for initial experiments")
    # print("2. COMPREHENSIVE (7 datasets) - More thorough coverage")
    
    # Default to focused
    option = 'focused'  # Change to 'comprehensive' if you want more datasets
    
    datasets = collector.collect_all(option=option)
    
    # Show statistics
    collector.get_dataset_statistics()
    
    # Visualize
    collector.visualize_datasets()
    
    # Save to disk
    collector.save_datasets()
    
    print("\n[OK] Dataset collection complete!")
    print(f"[OK] All files saved in '{collector.data_dir}' directory")
    print(f"\n[TIP] Next step: Run 'python dataset_exploration.py' to analyze characteristics")