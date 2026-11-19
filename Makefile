# One-command reproduction of the paper from the committed results.
#   make paper     rebuild every number, table and figure the manuscript uses, then compile manuscript.pdf and the supplement
#   make gpsnumbers  the two GPS-ABC chain analyses (need the 20,000-iteration chains in 4D/results/gps, which are not tracked)
#   make check     fast test of the simulator (10 s)
#   make numbers   regenerate tables/revision_numbers.tex from the result files (standard library only)
# Heavy steps (data generation, surrogate training, DNN-ABC, NPE, GPS-ABC) are not run here: see 4D/HOW_TO_RUN.md.
# Use PY=/path/to/venv/bin/python if numpy/pandas/scipy/matplotlib are not in the default python3.
PY ?= python3

.PHONY: paper check numbers gpsnumbers tables clean

paper: numbers tables
	latexmk -pdf -interaction=nonstopmode manuscript.tex
	latexmk -pdf -interaction=nonstopmode manuscript-supplementary.tex
	$(MAKE) clean

numbers:
	$(PY) 4D/analysis/p7_val_summary.py
	$(PY) 4D/analysis/npe_candidates_summary.py
	$(PY) 4D/analysis/revision_numbers.py

gpsnumbers:
	cd 4D && $(PY) analysis/gps_truncation.py && $(PY) analysis/gps_long_chain.py

tables:
	cd 4D && $(PY) analysis/study2_analysis.py --model 4d && $(PY) analysis/make_tables_tex.py && $(PY) analysis/make_figures.py --model 4d

check:
	cd 4D && $(PY) tests/test_sim.py

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk
