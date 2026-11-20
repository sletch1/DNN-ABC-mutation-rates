# One-command reproduction of the paper, from the experiments up.
#
#   make paper        run EVERY experiment (Study I, Study II, Ricker), then rebuild all numbers, tables and figures and compile
#                     manuscript.pdf and the supplement. Each experiment step is skipped when its output file already exists,
#                     so on a fresh clone with no results it runs everything, and on this repository it only fills gaps.
#                     To redo a step, delete its output file (listed next to each rule below); `make rerun-all` deletes them all.
#   make paper-fast   old behaviour: rebuild numbers, tables, figures and PDFs from the committed results only (minutes)
#   make study1       Study I only (1D/): surrogate, estimator comparison, MCSE, figures, tables
#   make study2       Study II only (4D/): data, surrogate, DNN-ABC, NPE, calibration, controls, repair checks
#   make ricker       the Ricker benchmark only
#   make package      rebuild submission_package/ (Springer Nature source files + PDFs) from the current results and compile-test it
#   make check        fast test of the simulator (10 s)
#   make numbers      regenerate tables/revision_numbers.tex from the result files (standard library only)
#   make gpsnumbers   the two GPS-ABC chain analyses (need the 20,000-iteration chains in 4D/results/gps, which are not tracked)
#
# NOT run by `make paper`: the GPS-ABC chains themselves (MATLAB, the original authors' code, hours per chain: see 4D/HOW_TO_RUN.md
# step 6) and the comparison with the authors' released training sets (needs Original_Paper_Code/training, kept local). Both are
# used if their results are present, and skipped with a message if not.
#
# Settings:   make paper PY=/path/to/venv/bin/python CORES=8      (CORES defaults to all cores; PY to .venv/bin/python if it exists)
# Needs: python packages of requirements.txt (torch, sbi, ...), and latexmk. Expect days of compute for a from-scratch run.

PY    ?= $(shell [ -x .venv/bin/python ] && echo $(CURDIR)/.venv/bin/python || echo python3)
CORES ?= $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
SC    ?= 0.05
ARCH  ?= res_128x3_gelu

# SMOKE=1 shrinks every experiment to a few seconds-to-minutes so the WHOLE pipeline can be tested end to end (results are meaningless).
# Use it in a scratch copy or with `make rerun-all` first: it writes to the same result files as a real run.
ifdef SMOKE
  FULL1D   := --quick
  GEN_X    := --n 3000 --rep 2
  OBS_X    := 3
  ARCH_X   := --seeds 1 --archs res_128x3_gelu
  DNN_X    := --iters 600 --n 3
  SBC_X    := --iters 600
  KSBC     := 12
  NPE_X    := --n 3 --draws 300
  P4_X     := --iters 600
  P3_X     := --iters 16000
  GP_X     := --budgets 50,100
  P7_X     := --K 50 --seeds 1
  P9_X     := --iters 200 --warm 100
  P10N     := 2000
  P11N     := 20000
  ORACLE_X := --idx 0 1 2 --n_s 10 --iters 400 --adapt 100
  CAND_X   := --n_lp 20 --n_sbc 20 --L 50
  RICK_GEN := 600
  RICK_X   := --iters 600
  RICK_K   := 8
  VAL_X    := --m 200 --seeds 1
else
  FULL1D   := --full
  GEN_X    := --n 10000 --rep 5
  OBS_X    := 100
  ARCH_X   := --seeds 5
  DNN_X    :=
  SBC_X    :=
  KSBC     := 500
  NPE_X    :=
  P4_X     :=
  P3_X     :=
  GP_X     :=
  P7_X     :=
  P9_X     :=
  P10N     := 10000
  P11N     := 2000000
  ORACLE_X := $(ORACLE_X)
  CAND_X   :=
  RICK_GEN := 60000
  RICK_X   :=
  RICK_K   := 500
  VAL_X    := --m 4000 --seeds 5
endif
export PYTHONUNBUFFERED = 1
export KERAS_BACKEND = torch   # for the BayesFlow NPE candidates (needs Python <= 3.13: see README)

R4 := 4D/results
CANDS := nsf maf mdn fmpe npse bf_affine bf_spline

.PHONY: package paper paper-fast study1 study2 ricker check numbers gpsnumbers tables clean rerun-all \
        data4 surrogate4 inference4 controls4 repairs4 oracle4 npecands4 validation4

paper: study1 study2 ricker $(if $(SMOKE),,numbers tables)
	$(if $(SMOKE),@echo SMOKE: experiments done - skipping numbers/tables/PDFs for toy-size results,latexmk -pdf -interaction=nonstopmode manuscript.tex && latexmk -pdf -interaction=nonstopmode manuscript-supplementary.tex && $(MAKE) clean)

paper-fast: numbers tables
	latexmk -pdf -interaction=nonstopmode manuscript.tex
	latexmk -pdf -interaction=nonstopmode manuscript-supplementary.tex
	$(MAKE) clean

# ------------------------------------------------------------------ Study I (1D)
# run_all.py $(FULL1D): trains the surrogate, runs the estimator comparison (40+ replicates, hours), MCSE, figures.
study1: $(if $(SMOKE),1D/results/tables/table1_mse.csv,1D/results/tables/study1_summary.json)   # the quick 1D run lacks the J grid the summary needs

1D/results/tables/table1_mse.csv:
	cd 1D && $(PY) run_all.py $(FULL1D)

1D/results/tables/study1_summary.json: | 1D/results/tables/table1_mse.csv
	$(PY) 1D/analysis/make_study1_tex.py

# ------------------------------------------------------------------ Study II (4D)
study2: data4 surrogate4 inference4 controls4 repairs4 oracle4 npecands4 validation4

# Step 1: training data and the observed data sets
data4: 4D/data/slow_data_4D.csv 4D/data/obs_4d_case1.npz 4D/data/obs_4d_case2.npz
4D/data/slow_data_4D.csv:
	cd 4D && $(PY) datagen/gen_training.py --model 4d $(GEN_X) --out data/slow_data_4D.csv
4D/data/obs_4d_case%.npz:
	cd 4D && $(PY) datagen/gen_observed.py --model 4d --case $* --nsim $(OBS_X) --out data/obs_4d_case$*.npz

# Step 2: architecture search, then selection and the final surrogate (hours)
surrogate4: $(R4)/surrogate_4d.pt
$(R4)/arch_search_4D_raw.csv: | 4D/data/slow_data_4D.csv
	cd 4D && $(PY) model/arch_search.py --data data/slow_data_4D.csv $(ARCH_X) --out results/arch_search_4D --max-cores $(CORES)
$(R4)/surrogate_4d.pt: | $(R4)/arch_search_4D_raw.csv
	cd 4D && $(PY) model/select_and_report.py --data data/slow_data_4D.csv --search results/arch_search_4D --out results

# Step 3: DNN-ABC on both cases, NPE, simulation-based calibration
inference4: $(R4)/dnnabc_4d_case1.npz $(R4)/dnnabc_4d_case2.npz $(R4)/npe_4d_case1.npz $(R4)/npe_4d_case2.npz $(R4)/sbc_4d.npz
$(R4)/dnnabc_4d_case%.npz: | $(R4)/surrogate_4d.pt 4D/data/obs_4d_case%.npz
	cd 4D && $(PY) inference/dnn_abc.py --surrogate results/surrogate_4d.pt --obs data/obs_4d_case$*.npz --out results/dnnabc_4d_case$*.npz --step-scale $(SC) --cores $(CORES) $(DNN_X)
$(R4)/npe_4d_case1.npz: | 4D/data/slow_data_4D.csv 4D/data/obs_4d_case1.npz
	cd 4D && $(PY) inference/npe.py --data data/slow_data_4D.csv --obs data/obs_4d_case1.npz --out results/npe_4d_case1.npz $(NPE_X)
$(R4)/npe_4d_case2.npz: | $(R4)/npe_4d_case1.npz 4D/data/obs_4d_case2.npz   # reuses the case-1 posterior (same training data)
	cd 4D && $(PY) inference/npe.py --data data/slow_data_4D.csv --obs data/obs_4d_case2.npz --out results/npe_4d_case2.npz --reuse results/npe_4d_case1.pkl $(NPE_X)
$(R4)/sbc_4d.npz: | $(R4)/surrogate_4d.pt $(R4)/npe_4d_case1.npz
	cd 4D && $(PY) inference/sbc.py --model 4d --surrogate results/surrogate_4d.pt --npe results/npe_4d_case1.pkl --K $(KSBC) --step-scale $(SC) --cores $(CORES) $(SBC_X) --out results/sbc_4d.npz

# Step 4: controls (matched 1,500-point surrogate; correlation head; Gaussian-process baseline)
controls4: $(R4)/dnnabc1500_4d_case1.npz $(R4)/dnnabc1500_4d_case2.npz $(R4)/dnnabcrho_4d_case1.npz $(R4)/dnnabcrho_4d_case2.npz \
           $(R4)/dnnabcrho1500_4d_case1.npz $(R4)/dnnabcrho1500_4d_case2.npz $(R4)/sbc_4d_rho.npz $(R4)/gp_baseline_4D.json
$(R4)/surrogate_4d_n1500.pt: | 4D/data/slow_data_4D.csv
	cd 4D && $(PY) model/train_matched.py --data data/slow_data_4D.csv --arch $(ARCH) --n 1500 --out results/surrogate_4d_n1500.pt
$(R4)/surrogate_4d_rho.pt: | 4D/data/slow_data_4D.csv
	cd 4D && $(PY) model/train_matched.py --data data/slow_data_4D.csv --arch $(ARCH) --n 100000 --corr --out results/surrogate_4d_rho.pt
$(R4)/surrogate_4d_rho_n1500.pt: | 4D/data/slow_data_4D.csv
	cd 4D && $(PY) model/train_matched.py --data data/slow_data_4D.csv --arch $(ARCH) --n 1500 --corr --out results/surrogate_4d_rho_n1500.pt
$(R4)/dnnabc1500_4d_case%.npz: | $(R4)/surrogate_4d_n1500.pt 4D/data/obs_4d_case%.npz
	cd 4D && $(PY) inference/dnn_abc.py --surrogate results/surrogate_4d_n1500.pt --obs data/obs_4d_case$*.npz --out results/dnnabc1500_4d_case$*.npz --step-scale $(SC) --cores $(CORES) $(DNN_X)
$(R4)/dnnabcrho_4d_case%.npz: | $(R4)/surrogate_4d_rho.pt 4D/data/obs_4d_case%.npz
	cd 4D && $(PY) inference/dnn_abc.py --surrogate results/surrogate_4d_rho.pt --obs data/obs_4d_case$*.npz --out results/dnnabcrho_4d_case$*.npz --step-scale $(SC) --cores $(CORES) $(DNN_X)
$(R4)/dnnabcrho1500_4d_case%.npz: | $(R4)/surrogate_4d_rho_n1500.pt 4D/data/obs_4d_case%.npz
	cd 4D && $(PY) inference/dnn_abc.py --surrogate results/surrogate_4d_rho_n1500.pt --obs data/obs_4d_case$*.npz --out results/dnnabcrho1500_4d_case$*.npz --step-scale $(SC) --cores $(CORES) $(DNN_X)
$(R4)/sbc_4d_rho.npz: | $(R4)/surrogate_4d_rho.pt
	cd 4D && $(PY) inference/sbc.py --model 4d --surrogate results/surrogate_4d_rho.pt --K $(KSBC) --step-scale $(SC) --cores $(CORES) $(SBC_X) --out results/sbc_4d_rho.npz
$(R4)/gp_baseline_4D.json: | 4D/data/slow_data_4D.csv
	cd 4D && $(PY) model/gp_baseline.py --data data/slow_data_4D.csv --out results/gp_baseline_4D.json $(GP_X)

# Step 5: why the chains are overconfident and what repairs it (Problems 3, 4, 7, 9, 10, 11 of the revision)
repairs4: $(R4)/p3/recal.json $(R4)/p3/is_check.json $(R4)/p3/is_check_simulator.json $(R4)/p3/compare.json $(R4)/p4.json \
          $(R4)/p7_extra_features.json $(R4)/p9/hmc_check.json $(R4)/p10/npe_is.json $(R4)/p11/ensemble_is.json
$(R4)/p3/data.npz: | $(R4)/surrogate_4d.pt
	cd 4D && $(PY) inference/p3_overconfidence.py data --surrogate results/surrogate_4d.pt --K $(KSBC)
$(R4)/p3/feature_checks.json: | $(R4)/p3/data.npz
	cd 4D && $(PY) inference/p3_overconfidence.py feature --surrogate results/surrogate_4d.pt --K $(KSBC)
$(R4)/p3/sbc_selfconsistent.json: | $(R4)/p3/feature_checks.json
	cd 4D && $(PY) inference/p3_overconfidence.py selfsbc --surrogate results/surrogate_4d.pt --K $(KSBC) --step-scale $(SC) $(P3_X)
$(R4)/p3/sbc_base.json: | $(R4)/p3/sbc_selfconsistent.json
	cd 4D && $(PY) inference/p3_overconfidence.py variant --surrogate results/surrogate_4d.pt --K $(KSBC) --step-scale $(SC) $(P3_X)
$(R4)/p3/sbc_base_long.json: | $(R4)/p3/sbc_base.json
	cd 4D && $(PY) inference/p3_overconfidence.py mixing --surrogate results/surrogate_4d.pt --K $(KSBC) --step-scale $(SC) $(P3_X)
$(R4)/p3/recal.json: | $(R4)/p3/sbc_base_long.json
	cd 4D && $(PY) inference/p3_overconfidence.py recal --surrogate results/surrogate_4d.pt --K $(KSBC)
$(R4)/p3/is_check.json: | $(R4)/p3/data.npz
	cd 4D && $(PY) inference/p3_is_check.py --surrogate results/surrogate_4d.pt
$(R4)/p3/is_check_simulator.json: | $(R4)/p3/data.npz
	cd 4D && $(PY) inference/p3_is_check.py --surrogate results/surrogate_4d.pt --simulator
$(R4)/p3/compare.json: | $(R4)/p3/sbc_selfconsistent.json
	cd 4D && $(PY) inference/p3_compare.py 48
$(R4)/p4.json: | $(R4)/surrogate_4d.pt
	cd 4D && $(PY) inference/p4_npe_vs_surrogate.py --data data/slow_data_4D.csv --surrogate results/surrogate_4d.pt --out results/p4.json --K $(KSBC) $(P4_X)
$(R4)/p7_extra_features.json: | 4D/data/slow_data_4D.csv
	cd 4D && $(PY) inference/p7_extra_features.py --data data/slow_data_4D.csv --out results/p7_extra_features.json $(P7_X)
$(R4)/p9/hmc_check.json: | $(R4)/p3/data.npz
	cd 4D && $(PY) inference/p9_hmc_check.py --surrogate results/surrogate_4d.pt --K $(KSBC) --workers $(CORES) $(P9_X)
$(R4)/p10/npe_is.json: | $(R4)/p3/data.npz $(R4)/npe_4d_case1.npz
	cd 4D && $(PY) inference/p10_npe_is.py --N $(P10N) --K $(KSBC) --workers $(CORES)
$(R4)/ensemble/member_%.pt: | 4D/data/slow_data_4D.csv
	cd 4D && mkdir -p results/ensemble && $(PY) model/train_ensemble.py --data data/slow_data_4D.csv --arch $(ARCH) --member $* --out results/ensemble/member_$*.pt
$(R4)/p11/ensemble_is.json: | $(foreach m,0 1 2 3 4 5 6 7 8 9,$(R4)/ensemble/member_$(m).pt) $(R4)/p3/data.npz
	cd 4D && $(PY) inference/p11_ensemble_is.py --members "results/ensemble/member_*.pt" --N $(P11N) --K $(KSBC)

# Step 6: independent reference posterior (synthetic-likelihood MCMC, simulator in the loop)
oracle4: $(R4)/tables/p2_oracle.json
$(R4)/oracle/oracle_case1_0.npz: | $(R4)/dnnabc_4d_case1.npz $(R4)/dnnabc_4d_case2.npz
	cd 4D && $(PY) inference/p2_oracle.py --cases 1 2 $(ORACLE_X)
$(R4)/tables/p2_oracle.json: | $(R4)/oracle/oracle_case1_0.npz
	cd 4D && $(PY) analysis/p2_oracle_compare.py --results results --gps-horizon 2000

# Step 7: the seven NPE candidates, three training seeds each
npecands4: $(R4)/npe_candidates_summary.json
$(R4)/npe_candidates/%.json: | 4D/data/slow_data_4D.csv
	cd 4D && $(PY) inference/npe_candidates.py $(patsubst %_seed1,%,$(patsubst %_seed2,%,$*)) --threads 2 $(CAND_X) \
	    $(if $(findstring _seed1,$*),--seed 1,$(if $(findstring _seed2,$*),--seed 2,--seed 0))
$(R4)/npe_candidates_summary.json: | $(foreach c,$(CANDS),$(R4)/npe_candidates/$(c).json $(R4)/npe_candidates/$(c)_seed1.json $(R4)/npe_candidates/$(c)_seed2.json)
	$(PY) 4D/analysis/npe_candidates_summary.py

# Step 8: extended check of our generator against the authors' released training sets (needs the local, untracked Original_Paper_Code/training)
validation4: $(R4)/validation_extended.json
$(R4)/validation_extended.json:
	@if [ -d Original_Paper_Code/training ]; then cd 4D && $(PY) tests/validate_extended.py $(VAL_X); \
	else echo "SKIP validation_extended: Original_Paper_Code/training not present (needs the authors' released data)"; fi

# ------------------------------------------------------------------ Ricker benchmark
ricker: Ricker/results/sbc_ricker.json Ricker/results/npe_ricker.json
# the intermediate .npz files are not tracked; each output regenerates what it needs
Ricker/results/sbc_ricker.json:
	cd Ricker && ([ -f results/train.npz ] || $(PY) ricker.py gen --n $(RICK_GEN) --cores $(CORES)) && ([ -f results/surrogate.npz ] || $(PY) ricker.py train) && $(PY) ricker.py sbc --K $(RICK_K) --cores $(CORES) --sl $(RICK_X)
Ricker/results/npe_ricker.json:
	cd Ricker && ([ -f results/train.npz ] || $(PY) ricker.py gen --n $(RICK_GEN) --cores $(CORES)) && $(PY) ricker.py npe --K $(RICK_K) --cores $(CORES)

# ------------------------------------------------------------------ numbers, tables, figures
numbers:
	$(PY) 4D/analysis/p7_val_summary.py
	$(PY) 4D/analysis/npe_candidates_summary.py
	$(if $(SMOKE),@echo SMOKE: skipping make_study1_tex,$(PY) 1D/analysis/make_study1_tex.py)
	@if ls 4D/results/gps/*.mat >/dev/null 2>&1; then $(MAKE) gpsnumbers; \
	else echo "SKIP gpsnumbers: no GPS-ABC chains in 4D/results/gps; using the committed summaries in 4D/results/tables"; fi
	$(PY) 4D/analysis/revision_numbers.py

gpsnumbers:
	cd 4D && $(PY) analysis/gps_truncation.py && $(PY) analysis/gps_long_chain.py

tables:
	cd 4D && $(PY) analysis/study2_analysis.py --model 4d && $(PY) analysis/make_tables_tex.py && $(PY) analysis/make_repair_tables.py && $(PY) analysis/make_figures.py --model 4d

check:
	cd 4D && $(PY) tests/test_sim.py

# Journal submission folder (Statistics and Computing: LaTeX source with style files and figures, a compiled PDF, figures named FigN, the
# supplement as Online Resource 1). Rebuilt from the current files and compiled inside the folder to prove it is self-contained.
package: numbers tables
	rm -rf submission_package && mkdir -p submission_package/tables
	cp sn-jnl.cls sn-basic.bst manuscript.bib submission_package/
	cp tables/*.tex submission_package/tables/
	cp 1D/results/figures/surrogate_fit.pdf submission_package/Fig1.pdf
	cp 1D/results/figures/fig_uncertainty.pdf submission_package/Fig2.pdf
	cp 1D/results/figures/fig_posterior.pdf submission_package/Fig3.pdf
	cp 4D/figures/fig_recovery_4d.pdf submission_package/Fig4.pdf
	cp 4D/figures/fig_sbc_4d.pdf submission_package/FigS1.pdf
	sed -e 's#1D/results/figures/surrogate_fit.pdf#Fig1.pdf#' -e 's#1D/results/figures/fig_uncertainty.pdf#Fig2.pdf#' \
	    -e 's#1D/results/figures/fig_posterior.pdf#Fig3.pdf#' -e 's#4D/figures/fig_recovery_4d.pdf#Fig4.pdf#' manuscript.tex > submission_package/manuscript.tex
	sed -e 's#4D/figures/fig_sbc_4d.pdf#FigS1.pdf#' manuscript-supplementary.tex > submission_package/manuscript-supplementary.tex
	cd submission_package && latexmk -pdf -interaction=nonstopmode manuscript.tex && latexmk -pdf -interaction=nonstopmode manuscript-supplementary.tex
	cp submission_package/manuscript-supplementary.pdf submission_package/ESM_1.pdf
	cd submission_package && rm -f *.aux *.blg *.log *.out *.fls *.fdb_latexmk   # keeps manuscript.bbl, which the journal's system compiles from
	@grep -n "PLACEHOLDER" submission_package/manuscript.tex && echo "WARNING: placeholders remain in manuscript.tex (see lines above); fill them before submitting" || true

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk

# Delete every regenerable experiment output so the next `make paper` recomputes it all from scratch (data and observed sets included).
# Leaves GPS-ABC chains, the Original_Paper_Code folder and the manuscript alone. Asks first.
rerun-all:
	@printf "This deletes all 1D, 4D and Ricker experiment results (days to recompute). Type 'yes' to continue: "; read a; [ "$$a" = yes ]
	rm -rf 1D/results/tables 1D/results/model 1D/results/logs/raw_replicates.csv
	cd 4D && rm -rf data/slow_data_4D.csv data/obs_4d_case*.npz results/arch_search_4D_* results/surrogate_4d*.pt results/dnnabc*.npz results/npe_4d_* \
	    results/sbc_4d*.* results/gp_baseline_4D.json results/p3 results/p4.json results/p7_extra_features.json results/p9 results/p10 results/p11 \
	    results/ensemble results/oracle results/npe_candidates results/npe_candidates_summary.json results/validation_extended.json
	rm -f Ricker/results/*
