# Data

Every `.npz` holds `X` (inputs) and `y` (target). All experiments standardise `X` and
`y` on the training split only and report metrics in original target units.

| file | source | shape | target | origin |
|---|---|---|---|---|
| `concrete.npz` | UCI Concrete Compressive Strength (Yeh 1998), from `Concrete_Data.xls` | 1030 x 8 | compressive strength (MPa) | measured |
| `protein.npz` | UCI Physicochemical Properties of Protein Tertiary Structure (Rana 2013), from `CASP.csv` | 45730 x 9 | RMSD | measured |
| `robot_arm.npz` | **kin8nm** (DELVE), via `fetch_openml('kin8nm', version=1)` | 8192 x 8 | end-effector distance | simulated |
| `sarcos.npz` | SARCOS inverse dynamics (Vijayakumar & Schaal 2000), GPML site | 48933 x 21 | torque, joint 1 (column 22) | measured |
| `synthetic_heteroscedastic.npz` | `src/datacollect.py`, unseeded | 5000 x 2 | sin x1 + 0.5 cos x2 + noise, noise sd 0.05 + 0.15 r/r_max | constructed |
| `real/airfoil.npz` | UCI Airfoil Self-Noise (Brooks et al. 1989) | 1503 x 5 | sound pressure (dB) | measured |
| `real/calhousing.npz` | California housing (Pace & Barry 1997), sklearn variant | 20640 x 8 | median value (x $100k) | measured |
| `real/energy.npz` | UCI Energy Efficiency (Tsanas & Xifara 2012) | 768 x 8 | heating load Y1 | simulated |
| `real/powerplant.npz` | UCI Combined Cycle Power Plant (Tufekci 2014) | 9568 x 4 | net electrical output (MW) | measured |
| `real/wine_white.npz` | UCI Wine Quality, white (Cortez et al. 2009) | 4898 x 11 | quality, integer 3-9 | measured |
| `real/yacht.npz` | UCI Yacht Hydrodynamics (Gerritsma et al. 1981) | 308 x 6 | residuary resistance | measured |

The constructed datasets `hetero_extreme`, `varying_smoothness`, `varying_density` and
`combined` are generated on the fly, seeded, by `src/structured_data.py`.

## Known issues

- **SARCOS test rows are also in the training file.** `sarcos.npz` stacks
  `sarcos_inv.mat` (44484) and `sarcos_inv_test.mat` (4449), and every test input also
  appears in the training file, so 4449 rows are duplicates. Under the paper's
  protocols the share of test points with an exact twin in training is 0.6%
  (n_train 1500) to about 2% (n_train 4800).
- **Duplicate inputs** in the canonical files: wine_white 937, protein 1756, powerplant 41,
  concrete 34. On wine_white about 13% of test points have an exact training twin.
- **calhousing is censored**: 4.7% of targets sit at the 5.00001 cap.
- **energy** is building-simulation output (192 designs x 4 orientations); orientation
  barely affects the target, so most test points have a near-twin in training.
- **wine_white** has an ordinal target; Gaussian calibration metrics treat it as continuous.
- **yacht** is small (184 / 62 / 62 train / val / test) and heavy-tailed, so ECE on it
  has a floor of about 0.04 even for a perfectly calibrated model.

## data/real provenance

No retrieval script exists for `data/real`; the committed files are canonical and
match the canonical shapes and first rows of the UCI files. Checksums below.

## Licences

The UCI datasets are CC BY 4.0. SARCOS (GPML site), California housing (StatLib) and
kin8nm (DELVE) carry no explicit licence and are redistributed here for research use
with attribution; cite the sources above.

## SHA-256

```
4b6764ebfad1f51d49220ac41b2d0fbffe2ce25a26663ab3777b9dfe91525acd  data/concrete.npz
3008d29963e7f18324ddb3c2afb338809757729d8779f17c8d0957904b7de762  data/protein.npz
9e7e24b27acde00d74dd68be4005995c8b754c57b074ccbe136d79d0b85731f8  data/robot_arm.npz
a371cde70d1c19cf2bcbae3c19f89c56fb1c86be74d784f140d834278082f8cb  data/sarcos.npz
a2924119799c39d78870ee3fcb5e2f67c2b08eed8f25b249d96d50c418b5b75a  data/synthetic_heteroscedastic.npz
4797d3577f90596470141a36bb783a3f66f754abdcdd60897243abb03eadfb52  data/real/airfoil.npz
b09ebc90c0ceac91bf21f1b0735e7073ae27ecdf83014cabdf897eb7aa2d71f8  data/real/calhousing.npz
4d0a6ba4fc716ab90fc491f0f030b6b359ef93cbabecb2fdc9a939434e1ad60a  data/real/energy.npz
ec5fc4c9b46ebe37fa6b7b1e7853e8fe6cd2c54485ac38dcfeeb1f21b8cf4ce9  data/real/powerplant.npz
04a17e87367af27c929f023f7c536615a6a2aadb52bae11768bd93da2de78863  data/real/wine_white.npz
7ad956fe27616f8b628712ee1aef3934bab6330c4250a50ae712b9482e33a32e  data/real/yacht.npz
4277cfcb4e91a181746cbc654f001b57951c9e6a80f4f795fdb5c807e0848f40  data/CASP.csv
710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4  data/Concrete_Data.xls
b8a249733253ba6097372fedee7696833fcf30de42037d5b4a7227f21a6d1d97  data/sarcos_inv.mat
161a59b5c3b4f4b404584323f181607b2acbe620eb134dc720760dc3f38f5cec  data/sarcos_inv_test.mat
```
