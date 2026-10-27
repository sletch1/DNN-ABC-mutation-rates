# Architecture search, 4-parameter model (Study II)

Twelve candidate networks for the zero-aware surrogate, five seeds each (60 fits),
trained on the 50,000-row training set (10,000 design points x 5 replicates), split
by design point 80/10/10. Run with `../model/arch_search.py`, reported by
`../model/select_and_report.py`; the paper's Table 7 and Supplement S2 come from here.

| File | Contents |
|---|---|
| `arch_search_4D_summary.csv` | one row per architecture: parameters, validation loss (mean, sd over seeds), test error as a multiple of the noise floor, conformal 95% coverage, AUC of the zero head, single-query latency |
| `arch_search_4D_raw.csv` | every fit (architecture x seed), all validation and test metrics |
| `arch_search_4D_meta.json` | noise floors, split sizes, share of zero rows (0.30), wall time |
| `arch_search_report_4d.pdf` | the generated report |
| `gp_baseline_4D.json` | Gaussian-process baseline (1,500 and 300 design points) |
| `tab_arch_4d.tex`, `tab_arch_supp_4d.tex` | tables as included in the manuscript |

## Selection rule (fixed before looking at test metrics)

Take the lowest mean validation loss; among models within one standard error of it,
choose the cheapest by single-query latency. Test metrics are reported, never used to choose.

## Result

The chosen model is `res_128x3_gelu`, a residual network of three 128-unit blocks with
GELU: validation loss -3.342 (sd 0.004), test error 0.93 / 0.94 times the noise floor
for log10 d-bar / log10 s, conformal 95% coverage 0.948, AUC of the zero head 0.998.

Why it is the final model:
- It has the best mean validation loss, and the runner-up (4x64 GELU MLP, -3.337) is
  0.0045 worse, more than one standard error (0.002), so the cheaper-model clause of the rule does not apply.
- Its error is at the noise floor, so there is little left for a larger network to gain:
  the 529k-parameter residual network is no better (0.95), and its validation loss is worse.
- The smaller and tanh/ReLU/SiLU models of about 9k parameters sit 2-10% above the
  floor, and the 437- and 773-parameter models 13-15% above, which is a real accuracy loss.
- The cost of the choice is latency (about 2 ms per query against about 1 ms for plain
  MLPs), which is irrelevant here because the sampler spends about 40 s per data set in total.
- Coverage, AUC and the zero head are the same for all architectures, so the decision
  rests on the accuracy of the two Gaussian heads.
