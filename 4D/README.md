# Study II: the two-stage mutation model (4-parameter DNN-ABC)

Parameters `(log10 p1, log10 p2, τ, log10 δ)`: the original paper's Simulation Study 2. Settings follow the original
paper (`t_p = 20`, `J = 100`, `p ∈ [1e-11, 1e-7]`, `τ ∈ [0.1, 19.9]`, `δ ∈ [0.5, 2]`); where paper and released code
differ we follow the paper (`../paper_code_discrepancy.md`).

```
sim/          simulators (ports of the authors' MATLAB), Latin hypercube, settings
datagen/      training data (CSV, same layout as Study I) and observed datasets (+ .mat export for MATLAB)
model/        zero-aware two-feature network, architecture search, selection + report
inference/    DNN-ABC sampler, NPE, step-width pilot, simulation-based calibration
gps_abc/      MATLAB wrapper that runs the authors' GPS-ABC on our observed data (patched/ = memory-safe copy of one function)
analysis/     recovery / coverage / fidelity tables, figures, LaTeX tables for the manuscript
server/       shell drivers for the multi-core server (dynamic core use)
tests/        simulator checks; comparison with the training sets the original authors released
data/         slow_data_4D.csv (rows: Z0,a,delta,p1,p2,tau,tp,J,design,rep,d_bar,d_1..d_100), observed *.npz
results/      search tables, trained surrogates, chains, posterior draws
```

## Pipeline

```bash
# 1. data (≈1 min on 32 cores)
python datagen/gen_training.py --model 4d --n 10000 --rep 5 --out data/slow_data_4D.csv
for c in 1 2; do python datagen/gen_observed.py --model 4d --case $c --nsim 100 --out data/obs_4d_case$c.npz; done
# 2. surrogate: 12 architectures x 5 seeds, selection on validation, final model + report
bash server/run_search_4d.sh
# 3. DNN-ABC, NPE, simulation-based calibration, and the matched-budget control (1,500 design points)
bash server/run_study2.sh 4d ; bash server/finish_study2.sh 4d      # DNN-ABC, NPE, calibration
bash server/run_matched.sh ; bash server/run_corr.sh                  # 1,500-point control; correlated-feature surrogate (DNN-ABC-rho)
python model/gp_baseline.py --data data/slow_data_4D.csv --out results/gp_baseline_4D.json
# 4. GPS-ABC (MATLAB, the authors' code; ~hours per replicate)
python datagen/export_mat.py data/obs_4d_case1.npz data/mat/obs_4d_case1   # and case2
(cd gps_abc && matlab -batch "make_init(12)")  &&  bash server/run_gps_case.sh 1 5 20000 5 & bash server/run_gps_case.sh 2 5 20000 5
# chains are checkpointed every 250 iterations; copy gps_abc/out/ckpt_*.mat to results/gps/ and analyse at a common length:
python analysis/study2_analysis.py --model 4d --gps-horizon 2000
# 5. tables and figures
python analysis/study2_analysis.py --model 4d && python analysis/make_tables_tex.py && python analysis/make_figures.py --model 4d
```

`tests/validate_vs_published.py` tests whether our generator is the original authors' generator: at each of their released design
points it draws from our simulator and applies a randomised probability-integral-transform Kolmogorov--Smirnov test to their stored
values (needs `../Original_Paper_Code/training/`, kept local). Result: `results/validation_vs_published.json`.
