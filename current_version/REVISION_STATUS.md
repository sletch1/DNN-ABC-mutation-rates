# Revision status: Statistics and Computing plan

Updated 2026-11-10. The manuscript, supplement and this file are kept in step with each change.

## Done (no compute needed)
* Action 0b, citation audit: 50 bib entries, every non-arXiv entry checked against Crossref or the PMLR page. BayesFlow now cites IEEE TNNLS 2022; liao2010 replaced by McKay et al. 1979; the arXiv-only hendrycks2016 dropped. `talts2018` stays arXiv-only (no journal version found) and is flagged in `manuscript.bib`. Added: Luria-Delbruck lineage, ABC regression/NN, surrogate MCMC (incl. Reiser et al. 2025, S&C), synthetic likelihood, neural likelihood and benchmarks, SBC, conformal, hurdle and heteroscedastic regression, recent S&C ABC work. Every key is cited and none is undefined.
* Dinh et al. 2025 has authors Dinh, Liu, Xiang, Liu, Tavare; the plan's author list was incomplete.
* Format: abstract rewritten (212 words); conformal removed from the abstract and keywords as a posterior-calibration claim; "Study I, Table n" caption prefixes removed; supplement renamed Online Resource 1; "Statements and Declarations" heading; limitations paragraph added.

## Server work (stat86 reachable; password login, key login is blocked by NFS home dirs)
Updated 2026-11-10 10:30. All heavy jobs run in the background with dynamic core use (`1D/abc/dyncores.py`: each worker waits for an idle core).
| Item | Status |
|---|---|
| P1 GPS-ABC data sets 6-20 (30 chains, 2,000 it.) | running via `4D/server/gps_queue.sh`; about 7-10 h per chain; 2 of 30 finished at 11:50 on 11/10 |
| P1 20,000-iteration chains (one per case) | running (resumable checkpoints) but slow: 2,500 of 20,000 iterations after 8-11 h (11:50 on 11/10), slowing as the GP grows; will NOT reach 20,000 by the submission target. Plan: analyse the length reached by about 11/17 from the checkpoints and report the full 20,000 as a limitation |
| P2 oracle posterior (8 data sets) | done; `4D/analysis/p2_oracle_compare.py`; in manuscript (Section on independent reference) |
| P3 feature checks, self-consistency SBC | done: self-consistency SBC is still overconfident for log10 p2 and log10 delta; in manuscript |
| P3 mixing test (5x longer chains), Student-t / eps / recalibration repairs | running |
| P4 NPE with zero indicator, prior change | done; in manuscript |
| P5 Study I with 100 replicates | running (about 25% done) |
| P6 extended simulator validation | running |
| P6 Ricker benchmark | surrogate trained, SBC running |
| P7 extra-feature NPE | running (seed 2 of 3 for the first feature set) |

## Needs the authors
* Zenodo DOI for the GitHub release, competing interests, funding, generative-AI disclosure, VT open-access/APC waiver check.
* Action 0: the three S&C comparison papers' full text is not in `lit_review/`; none were read, so no benchmark table was made.
