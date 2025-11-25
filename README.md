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
