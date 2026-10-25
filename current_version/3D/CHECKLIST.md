# Plan / status (notes_oct_23)

Python env used locally: ../current_work/Models/3D/.venv/bin/python (numpy, scipy, torch, pandas, matplotlib)

## Done (local, smoke-tested only)
- [x] sim/mbp.py: ports of mut_bMBP, mut_bMBP_rev, mut2stage_bMBP (exact, 3-param), mut2stage_bMBP_rev (fast, 4-param incl. delta). MATLAB geornd/round conventions kept.
- [x] sim/mbp.py: mut2stage_exact_delta -- NEW exact 4-param simulator (not in repo; see below).
- [x] sim/settings.py: all numbers from simulation2_server.m / simulation1_server.m (ranges, dt_range=[log10 .5, log10 2], tp, eps, ksi, steps, 4th-root statistic).
- [x] sim/lhs.py: LHS = lhsdesign equivalent + generic maximin. NOT a transcription of Liao et al. 2010 (paper not read).
- [x] datagen/gen_training.py: CSV in the slow_data_*.csv layout (1d/3d/4d, --exact option).
- [x] datagen/gen_observed.py: simulation2.m-style observed sets (100 seeds x 100 cultures, MOM).
- [x] model/train.py (DNN for mean/sd features), inference/mcmc.py (published proposal/prior structure; "G" = G fresh simulations kernel-ABC likelihood, "gauss" = old normal approx), figures/fig3d.py.

## Not done
- [ ] Full-size data (n=10000 designs) and ABC runs -- need stat86 (see message).
- [ ] 1D re-run, results cleanup, dropping surrogate_fit.png, manuscript update (need real results first).
- [ ] Validation of fast vs repo MATLAB: no MATLAB/Octave here.

## Findings to resolve
1. The published fast simulator (M=round(Z*p)) gives ~half the mean mutant count of the exact one (tests/test_sim.py). Ported faithfully.
2. Exact simulation at tp=20 is infeasible (~5e8 cells/culture). Measured 4D exact cost/culture: tp=10 2ms, 12 7ms, 14 61ms, 15 240ms (x~4 per +1 beyond). 
3. Repo's 1-D scripts use the 4th root (sqrt(sqrt(X/Z))), while the current 1D code uses root 2.

## Audit against the authors' shipped training sets (tests/validate_vs_published.py)
- 4D (training/init_2a.mat, 1500x4x5): column order (log10 delta, log10 p1, log10 p2, tau); ranges [-0.30,0.30],[-11,-7],[-11,-7],[4,18]; valid LHS (1500/1500 strata per column); 5 single-culture reps per point.
  Our fast simulator at the SAME thetas: zero-fraction 0.564 vs 0.558, mean Y 0.00751 vs 0.00752, rep-sd 0.00371 vs 0.00365.
- 1D (training/simu1/init1_p{4..8}.mat, 1500x1x10): boxes and tp=fzero(...) reproduce; zero-fraction, mean and rep-sd agree to ~1-3% for p=1e-8,1e-6,1e-4.
- NOT reproducible: the observed datasets Y2_<seed>.mat (MATLAB mt19937ar seeds 1-100, stored on the authors' server, not in the repo).

## Rule from user: paper wins over code (paper_code_discrepancy.md) -- applied
tau in [0.1,19.9]; xi=0.3; theta order (p1,p2,tau,delta); priors TN(log10 MOM,20,[-11,-7]) x2, TN(10,20,[0.1,19.9]), TN(0,20,[log10 .5,log10 2]).
Open: paper's step widths (0.8,3.5,3.5,4) are unlabelled -> assigned (delta,p1,p2,tau) as in the code (only sensible reading).
3-D = Algorithm 4 with delta=1 (paper fn.5: extends to piecewise constant; the composition in mut2stage_bMBP_rev.m is the only specification of that extension).
server/run_3d.sh generates the 3-D training CSV (10000 designs x 5 reps x J=100) and the two observed-data cases.

## STATUS at pause (Oct 23, afternoon) -- resume from here
Server stat86 (jobs run in background, survive laptop sleep): ~/new_version/3D
 - results/search_all.log : server/run_search.sh = arch search + final surrogate for 3D then 4D (two features, hurdle). Marker SEARCH_DONE.
 - gps_abc/make_init.log   : MATLAB make_init(12) -> gps_abc/init_2a_paperbox.mat (authors' GPS-ABC training set on the paper box)
 - ~/sbi_install.log       : pip install sbi into ~/abc_env (NPE)
 - data/ has slow_data_3D.csv, slow_data_4D.csv, obs_{3d,4d}_case{1,2}.npz; data/mat/ has the 4D observed .mat files for MATLAB
Local code written but NOT yet run on the server: inference/dnn_abc.py, inference/npe.py, inference/sbc.py, gps_abc/run_rep.m (needs init file; time one replicate first).
TODO next: (1) pilot step widths for DNN-ABC (paper widths give ~4% acceptance; paper target is 20-40%); (2) run DNN-ABC/NPE on 100 obs datasets x 2 cases x {3D,4D}; (3) GPS-ABC replicates in MATLAB (subset, measure time first; 3D needs a minimally patched ABC_mu2.m); (4) SBC; (5) analysis tables + few clear figures; (6) rewrite Study II + abstract/intro/discussion/conclusion/reproducibility in manuscript.tex and supplementary; (7) README/requirements/cleanup; (8) rebuild PDF and `open` a status PDF.
Local cleanup done: smoke files removed; lit_review PDFs and Original_Paper_Code/training kept locally but git-ignored; .gitignore updated (current_work/ and prof_version/ ignored). Nothing committed or pushed.
