# Revision status: Statistics and Computing plan

Updated 2026-11-10. The manuscript, supplement and this file are kept in step with each change.

## Done (no compute needed)
* Action 0b, citation audit: 50 bib entries, every non-arXiv entry checked against Crossref or the PMLR page. BayesFlow now cites IEEE TNNLS 2022; liao2010 replaced by McKay et al. 1979; the arXiv-only hendrycks2016 dropped. `talts2018` stays arXiv-only (no journal version found) and is flagged in `manuscript.bib`. Added: Luria-Delbruck lineage, ABC regression/NN, surrogate MCMC (incl. Reiser et al. 2025, S&C), synthetic likelihood, neural likelihood and benchmarks, SBC, conformal, hurdle and heteroscedastic regression, recent S&C ABC work. Every key is cited and none is undefined.
* Dinh et al. 2025 has authors Dinh, Liu, Xiang, Liu, Tavare; the plan's author list was incomplete.
* Format: abstract rewritten (212 words); conformal removed from the abstract and keywords as a posterior-calibration claim; "Study I, Table n" caption prefixes removed; supplement renamed Online Resource 1; "Statements and Declarations" heading; limitations paragraph added.

## Blocked: stat86 unreachable
`ssh stat86` times out at TCP (128.173.70.86:22 filtered), so this machine is off the VT network/VPN. Key auth could not be tested. All heavy items need that first.
| Item | Status |
|---|---|
| P1 extra GPS-ABC chains, 20k chains | not started |
| P2 oracle posterior | not started |
| P3 SBC diagnostics and repairs | not started |
| P4 NPE comparison | not started |
| P5 Study I with 100 replicates | not started |
| P6 Ricker benchmark, validation | not started |
| P7 extra-feature NPE | not started (needs torch/sbi; not installed locally) |

## Needs the authors
* Zenodo DOI for the GitHub release, competing interests, funding, generative-AI disclosure, VT open-access/APC waiver check.
* Action 0: the three S&C comparison papers' full text is not in `lit_review/`; none were read, so no benchmark table was made.
