# HK-dEIT: Hellinger-Kantorovich regularized dynamic EIT for pendelluft quantification

MATLAB code for the manuscript

> H. N. Huynh, *Hellinger-Kantorovich regularized dynamic electrical impedance tomography for pendelluft quantification: transport-based reconstruction, identifiability analysis, and in vivo null validation*, submitted to Medical Image Analysis.

Author: **Hoang Nhut Huynh**, National Chung Hsing University, Taichung, Taiwan.

The code reconstructs time-difference EIT with an unbalanced optimal transport (Hellinger-Kantorovich) action on the dual graph of the finite-element mesh, estimates the interregional gas flux, removes the part of it that is in phase with global flow (quadrature estimator), and compares every result with nulls processed by the same pipeline. It also contains the image-based baselines (opposite flow, region-out, FRIC-vol, pixel amplitude, regional quadrature on Gauss-Newton images) and every script that produced a number, table, or figure of the paper.

---

## 1. Requirements

| Item | Version used | Notes |
|---|---|---|
| MATLAB | R2017b or later | core MATLAB only, no toolboxes, no `parfor` |
| EIDORS | v3.12-ng | forward models, Jacobians, Gauss-Newton reference, library thorax models (Netgen is needed once to build `adult_male_32el_lungs`, used only for an excluded recording) |
| Python 3 (optional) | numpy, matplotlib | only for `graphical_abstract/make_graphical_abstract.py` |

Start EIDORS before running any script:

```matlab
run('path/to/eidors-v3.12-ng/eidors/startup.m');
cd('path/to/this/repository');
```

Every script locates its data and writes its results relative to its own folder, so keep all `.m` files in the repository root and run them from there.

---

## 2. Data

All recordings are public (EIDORS contributed data). They are not redistributed here. Place them in a folder `Data/` in the repository root with this layout:

```
Data/
├── baby/if-neonate-spontaneous (1)/P04P-1016.get        neonate, spontaneous breathing (Heinrich et al., 2006)
├── child/cg_data_2012_p1/*.get                          pediatric ARDS, patient 1 (Gomez-Laberge et al., 2012; Wolf et al., 2012)
├── child/cg_data_2012_p7/*.get                          pediatric ARDS, patient 7 (excluded, Suppl. S5)
├── Pig_torso/g08-c003.get ... g10-c029.get              piglets, normal / lavaged / surfactant (Dargaville et al., 2010)
├── Draeger/ET_Test_File_Healthy_Lung_01.eit             healthy adult A, Draeger PulmoVista 500
├── Midas_Medical_Ventom/EITData2024_10_31_08_43_09__S01.vtm   healthy adult B
├── healthy_human/human-ventilation-2014.eit             32-electrode adult recording (excluded, Suppl. S5)
└── normal_human/cg_invivo_1pl, cg_invivo_2pl            adult one/two-plane recording (excluded, Suppl. S5)
```

Please cite the original publications of each data set (see the reference list of the paper).

Folders created by the scripts (all ignored by git): `E3_results/`, `E5_results/`, `E5b_results/`, `maps/`, `figs/`, `logs/`, and `*_results*.mat` files.

---

## 3. Code overview

### Core method

| File | Role | Paper |
|---|---|---|
| `prox_perspective.m` | proximal map of the perspective function a²/(2b) (closed-form cubic + monotone Newton) | Sec. 2.4 |
| `calc_dual_graph.m` | dual graph of the FEM mesh: incidence B, face areas, centroid distances, diamond volumes, volume-balanced clusters | Sec. 2.2 |
| `domain_graph.m` | cluster graph on the lungs or the whole thorax (`MASK = 'thorax'` in the paper) | Sec. 2.2 |
| `hk_K.m`, `hk_Kt.m` | linear operator K and its adjoint | Sec. 2.4 |
| `hk_core.m` | preconditioned Chambolle-Pock solver with ventilation mode `w·c(t)`, background content `rho0`, adaptive steps, Woodbury quadratic step, residual-based stopping | Sec. 2.3, 2.4 |
| `inv_solve_diff_hk.m` | EIDORS wrapper (`imdl.solve = @inv_solve_diff_hk`) | Sec. 2.4 |
| `hk_pendelluft.m` | interface flux Q_Γ(k) between anterior and posterior clusters | Sec. 2.5 |
| `pend_estimators.m` | baselines on GN images: opposite flow, GN quadrature (qGN), FRIC-vol, region-out, pixel amplitude, regional lag | Sec. 2.7 |
| `pend_maps.m` | cluster-level TIV, out-of-phase density, and regional curves for one window | Fig. 12 |
| `post_timing_shift.m` | timing contrast C and its shift-based z score (E3, E5) | Sec. 2.5, Suppl. S4 |
| `load_real_eit.m` | reader for `.get`, `.eit`, `.vtm` files with frame rate, bad-frame and jump checks | Sec. 3.3 |

### Verification

| File | Role | Paper |
|---|---|---|
| `test_hk_unit.m` | unit tests: proximal map vs brute force, B·1 = 0, adjoint of K, step-size condition, scale equivariance, Woodbury vs dense, affine shifts, nuisance channel. **Run first; all tests must pass.** | Suppl. S8 |
| `toy_hk_1d.m` | 1-D toy problem | development |
| `diag_cp_E4.m` | choice of the initial primal step `tau0` | development |

---

## 4. Reproducing the paper

Parameters are fixed in every script to the values of the paper (α = 1e-3, β = 1e-2, d_m = 4, κ = 3, whole-thorax domain). Run times are for one desktop PC.

### Step 0. Unit tests

```matlab
test_hk_unit
```

### Step 1. Simulations (Figs. 2, 3, 11; Tables 4, 5; Suppl. S7)

| Order | Script | Settings | Output | Paper | Time |
|---|---|---|---|---|---|
| 1 | `run_E4d_noncrime` | fine forward mesh, coarse inverse mesh, 1 % noise | `E4d_results.mat` | Fig. 2, Sec. 4.1 | |
| 2 | `run_E4c_dose` | identical meshes | `E4c_results.mat` | Fig. 3 | |
| 3 | `run_E6_mechanism` | `MODE = 'full'` (11 conditions, 2 seeds) | `E6_results.mat` | Fig. 11, Table 4, Suppl. Tables S5, S6, Sec. 4.4 | ~45 min |
| 4 | `run_E6_mechanism` | `MODE = 'noise'` (0.3, 3, 10 % noise, 3 seeds) | `E6_noise_results.mat` | Table 5, Suppl. Table S7 | ~1.2 h |

`run_E4_sim.m` and `run_E4b_sim.m` are earlier development versions (same mesh for simulation and reconstruction) and are not used in the paper.

### Step 2. Neonate (Figs. 1, 4 to 10; Tables 2, 3; Suppl. S1 to S3)

Run in this order with `MASK = 'thorax'` (default):

| Order | Script | What it does | Output | Paper |
|---|---|---|---|---|
| 1 | `run_E2_neonate` | GN and HK-dEIT reconstruction, regional curves, flux | `E2_results_thorax.mat` | Figs. 4, 5, 6; Sec. 4.2 |
| 2 | `run_E2b_floor` | raw-flux floors (noise and gradient nulls) and dose response | `E2b_results_thorax.mat` | Sec. 2.5 (raw flux) |
| 3 | `run_E2e_quadrature` | quadrature estimator, gradient null, dose response | `E2e_results_thorax.mat` | Figs. 7, 8; Sec. 2.5 |
| 4 | `run_E2f_floors` | quadrature floors per breath for noise and model-mismatch nulls (`BASE_ONLY = false`), GN baselines | `E2f_results_thorax.mat`, `E2f_baselines_thorax.mat` | Figs. 5, 7; Table 2; Suppl. Fig. S1 |
| 5 | `run_E2g_async` | regional lag, transfer-free delay nulls | `E2g_v2_results_thorax.mat` | Figs. 9, 10; Table 3; Sec. 4.3 |
| 6 | `run_E2h_rc` | transfer-free time-constant nulls | `E2h_results_thorax.mat` | Fig. 9, 10; Table 3 |

Supplementary analyses on the same recording:

| Script | Paper |
|---|---|
| any `run_E2*` with `MASK = 'lungs'` (writes files without the `_thorax` suffix) | Suppl. S1, Table S1 (lung-only column) |
| `run_E2i_mask` | Suppl. S1 (dilated masks and whole thorax) |
| `diag_E2_antiphase` | Suppl. S2 (opposite-sign component) |
| `run_E2d_mismatch` | Suppl. S2 (opposite-sign component under model mismatch, lung-only domain) |
| `run_E2c_nuisance` | Suppl. S2 (nuisance channel) |

### Step 3. Pediatric ARDS (Fig. 14; Suppl. S4, S5)

| Script | Settings | Output | Paper | Time |
|---|---|---|---|---|
| `run_E3_ards` | `MAPS_ONLY = {}`, `SUBJ_RUN = 1` (patient 1, 9 PEEP steps, last 40 s, 5 nulls) | `E3_results/*_thorax_v2.mat` | Fig. 14, Sec. 4.5.2 | ~5 h |
| `post_timing_shift` | timing contrast C and z for E3 | printed table | Suppl. Table S2 | minutes |
| `diag_E3_quality`, `diag_E3_p7`, `diag_E3_p7b` | data quality and model fit for patient 7 | printed tables | Suppl. S5 (exclusion of patient 7) | minutes |

### Step 4. Piglets (Fig. 13; Table 6; Suppl. S3, S4)

| Script | Settings | Output | Paper | Time |
|---|---|---|---|---|
| `run_E5_piglets` | whole 180-s maneuver in one run, one projection coefficient | `E5_results/` | Suppl. S3 (why the analysis is local) | |
| `run_E5b_local` | `MAPS_ONLY = {}`; HK-dEIT per PEEP level, 3 in-phase nulls per window, baselines | `E5b_results/*.mat`, summary tables | Fig. 13, Table 6, Sec. 4.5.1, Suppl. S4 | ~8 h (resumable) |

### Step 5. Healthy adults and breath holds (Fig. 15; Table 7; Suppl. S5, S6)

| Script | Settings | Output | Paper | Time |
|---|---|---|---|---|
| `run_E7_healthy` | `ORIENT = 'native'`, `MAPS_ONLY = {}` | `E7_results_draeger.mat`, `E7_results_midas.mat`, `E7_results_swisstom.mat` | Fig. 15, Table 7, Sec. 4.5.3 | ~5 h |
| `run_E7_healthy` | `ORIENT = 'cardiac'` (rename the previous result files first) | alternative registrations | Suppl. S6, Table S4 | ~5 h |

The `swisstom` (32-electrode) data set is run but excluded from quantification (Suppl. S5).

### Step 6. Per-dataset maps for Fig. 12

Set `MAPS_ONLY` and rerun three scripts. Each writes one file to `maps/` and does not touch the results of Steps 3 to 5.

```matlab
% run_E3_ards.m:     MAPS_ONLY = {'1001_b'};
% run_E5b_local.m:   MAPS_ONLY = {'g08-c003', 11; 'g08-c017', 11; 'g08-c035', 11};
% run_E7_healthy.m:  MAPS_ONLY = {'draeger', 2; 'midas', 2};
run_E3_ards; run_E5b_local; run_E7_healthy
```

The neonate row is computed from `E2_results_thorax.mat` inside `make_figs_HK`.

### Step 7. Figures

```matlab
make_figs_HK      % writes figs/Fig01 ... Fig16 (.fig, .pdf, .png)
```

Mapping between the files written by `make_figs_HK` and the figure numbers of the paper:

| Paper | File | Data from |
|---|---|---|
| Fig. 1 | `Fig01_geometry` | model of `run_E2_neonate` |
| Fig. 2 | `Fig10_E4d` | `run_E4d_noncrime` |
| Fig. 3 | `Fig11_E4c` | `run_E4c_dose` |
| Fig. 4 | `Fig02_images` | `run_E2_neonate` |
| Fig. 5 | `Fig04_timecourses` | `run_E2_neonate`, `run_E2f_floors` |
| Fig. 6 | `Fig03_flux` | `run_E2_neonate` |
| Fig. 7 | `Fig06_floors` | `run_E2e_quadrature`, `run_E2f_floors` |
| Fig. 8 | `Fig05_dose_response` | `run_E2e_quadrature` |
| Fig. 9 | `Fig08_asynchrony` | `run_E2g_async`, `run_E2h_rc` |
| Fig. 10 | `Fig09_timing` | `run_E2g_async`, `run_E2h_rc` |
| Fig. 11 | `Fig14_E6` | `run_E6_mechanism` |
| Fig. 12 | `Fig16_gallery` | Step 6 maps + `E2_results_thorax.mat` |
| Fig. 13 | `Fig12_E5b` | `run_E5b_local` |
| Fig. 14 | `Fig13_E3` | `run_E3_ards` |
| Fig. 15 | `Fig15_E7` | `run_E7_healthy` |
| Suppl. Fig. S1 | `Fig07_per_breath` | `run_E2f_floors` |

### Tables

| Paper | Printed by |
|---|---|
| Table 1 (data) | Sec. 2 of this README |
| Table 2 (neonate estimators) | `run_E2f_floors` |
| Table 3 (asynchrony nulls) | `run_E2g_async`, `run_E2h_rc` |
| Table 4 (mechanism test) | `run_E6_mechanism`, `MODE = 'full'` |
| Table 5 (noise) | `run_E6_mechanism`, `MODE = 'noise'` |
| Table 6 (piglets) | `run_E5b_local` (summary at the end of the run) |
| Table 7 (adults) | `run_E7_healthy` (summary at the end of the run) |

Each script prints its table to the MATLAB command window. Use `diary('logs/<name>.log')` before a run to keep a log.

### Excluded adult one/two-plane recording (Suppl. S5)

`prep_E1` → `run_E1_hk` → `run_E1_sens`, with `diag_E1_alpha` and `diag_E1_pos` for the parameter scan of Suppl. Table S3.

---

## 5. Notes

- Breaths whose windows touch a breath hold are excluded from tidal statistics; breath-hold windows have the length of one inspiration (`run_E7_healthy`).
- Phase-randomized in-phase nulls are unreliable for records with breath holds or baseline jumps; breath holds are the reference for the adults.
- Single clusters are not interpreted; all reported quantities are regional sums.

## 6. Citation

If you use this code, please cite the paper above (full reference to be added after publication).
