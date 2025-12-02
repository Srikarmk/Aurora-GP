# diagnostic_routing.py
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from aurora_final import AURORA

# Get project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Load concrete
data = np.load(PROJECT_ROOT / 'data' / 'concrete.npz')
X, y = data['X'], data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train AURORA
aurora = AURORA(random_state=42)
aurora.fit(X_train, y_train, n_gp_iter=20, verbose=False)

# Get predictions WITH regions
y_pred, y_std, regions = aurora.predict(X_test, return_std=True, return_regions=True)

# Check distribution
print("\nTEST REGION DISTRIBUTION:")
print(f"  Low (0):    {np.sum(regions==0)} ({np.sum(regions==0)/len(X_test)*100:.1f}%)")
print(f"  Medium (1): {np.sum(regions==1)} ({np.sum(regions==1)/len(X_test)*100:.1f}%)")
print(f"  High (2):   {np.sum(regions==2)} ({np.sum(regions==2)/len(X_test)*100:.1f}%)")

# Check which model is being used
if np.sum(regions==0) == 0 and np.sum(regions==1) == 0:
    print("\n🚨 ALL POINTS ROUTED TO HIGH (Exact GP)!")
    print("   This explains too-perfect ECE")
elif np.sum(regions==2) == 0:
    print("\n🚨 NO POINTS ROUTED TO HIGH!")
    print("   This explains worse performance")
else:
    print("\n✓ Points distributed across regions")

# Verify model types
print("\nMODEL TYPES:")
print(f"  Model 0: {type(aurora.models[0]).__name__}")
print(f"  Model 1: {type(aurora.models[1]).__name__}")
print(f"  Model 2: {type(aurora.models[2]).__name__}")

print("\nExpected:")
print("  Model 0: RandomFourierFeatures")
print("  Model 1: NystromApproximation")
print("  Model 2: GaussianProcessBaseline")