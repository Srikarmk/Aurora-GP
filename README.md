# Aurora-GP

## Project Overview

Aurora-GP is a project focused on **region-aware Gaussian Process (GP) approximation**. The goal is to develop and evaluate GP methods that can adapt their approximation strategy based on different regions of the input space. This is particularly useful for datasets where:

- Data density varies across the input space (non-uniform distribution)
- Noise levels vary by region (heteroscedasticity)
- Function smoothness changes in different areas

## Datasets

The project includes several benchmark datasets commonly used in Gaussian Process regression:

### 1. **Protein (CASP)**

- **Source**: CASP (Critical Assessment of Structure Prediction) competition data
- **Description**: Protein structure prediction dataset. Contains features related to protein sequences and their predicted structural properties.
- **Use Case**: Tests GP performance on biological data with potentially complex, non-linear relationships.

### 2. **Concrete**

- **Source**: Concrete compressive strength dataset
- **Description**: Contains features such as cement content, water, fine aggregate, coarse aggregate, age, etc., with the target being concrete compressive strength.
- **Use Case**: Engineering/material science regression problem with multiple input features and real-world noise characteristics.

### 3. **Robot Arm**

- **Description**: Robot arm kinematics or dynamics dataset. Typically involves predicting joint positions, velocities, or torques based on arm configuration.
- **Use Case**: Tests GP on mechanical systems with potentially smooth but complex dynamics.

### 4. **SARCOS**

- **Source**: SARCOS inverse dynamics dataset
- **Description**: Well-known robotics benchmark dataset for learning inverse dynamics of a 7-DOF robot arm. Contains 21 input features (joint positions, velocities, accelerations) and predicts joint torques.
- **Use Case**: High-dimensional robotics problem that challenges GP scalability and accuracy.

### 5. **Synthetic Heteroscedastic**

- **Description**: Artificially generated dataset designed with intentionally varying noise levels (heteroscedasticity) across different regions of the input space.
- **Use Case**: Controlled test case specifically designed to evaluate how well region-aware methods can adapt to varying noise levels.

## Project Structure

```
Aurora-GP/
├── data/                          # Dataset files
│   ├── robot_arm.npz
│   ├── concrete.npz
│   ├── protein.npz
│   ├── sarcos.npz
│   └── synthetic_heteroscedastic.npz
├── src/                           # Source code
│   ├── gp_baseline.py            # Exact GP baseline implementation
│   ├── gp_visualization.py       # Visualization tools
│   ├── gp_test.py               # Test suite for GP implementation
│   └── dataset_exploration.py    # Dataset analysis and exploration
├── results/                       # Output directory (created automatically)
│   └── baseline_gp/              # Baseline GP results
├── notebooks/                     # Jupyter notebooks
│   └── analysis.ipynb           # Analysis notebook
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## Usage

### Running Dataset Exploration

To analyze all datasets and generate visualizations:

```bash
python src/dataset_exploration.py
```

This will:
1. Load all datasets from the `data/` directory
2. Run comprehensive analysis on each dataset
3. Generate recommendations for region-aware testing
4. Create detailed visualizations saved to `results/`

### Running GP Baseline Experiments

To run exact GP baseline experiments on all datasets:

```bash
python src/gp_baseline.py
```

### Running Tests

To verify the GP implementation:

```bash
python src/gp_test.py
```

### Generating Visualizations

To generate visualization reports for all datasets:

```bash
python src/gp_visualization.py
```

## Requirements

See `requirements.txt` for full dependency list. Key dependencies include:
- `numpy`, `scipy`, `pandas` - Core scientific computing
- `scikit-learn` - Machine learning utilities
- `gpytorch`, `torch` - Gaussian Process modeling
- `matplotlib`, `seaborn` - Visualization
