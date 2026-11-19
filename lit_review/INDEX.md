# Literature review: recent, high-prestige, directly relevant papers

Rule for this folder: published 2024 or later, in Statistics and Computing (S&C) or a top statistics or machine-learning venue, and directly relevant to the manuscript. The PDFs are not committed (publisher copyright; `lit_review/*.pdf` is gitignored). This index is.

Last updated 2026-11-19. Every entry's bibliographic data was checked against Crossref. All 13 papers are cited in `manuscript.tex`.

| # | Paper | Venue | PDF in folder | Why it is relevant, and where the manuscript cites it |
|---|---|---|---|---|
| 1 | Reiser, Aguilar, Guthke, Bürkner (2025), Uncertainty quantification and propagation in surrogate-based Bayesian inference. doi:10.1007/s11222-025-10597-8 | S&C 35:66 | `reiser2025_surrogate_UQ_StatComput.pdf` | Closest paper. Classifies surrogate errors (aleatoric simulator noise, approximation error, epistemic parameter uncertainty) and propagates them; SBC extended to surrogate pipelines. Related work, SBC paragraph, diagnosis, limitations, and the error-budget subsection. |
| 2 | Soldát, Šmídl (2026), Bayesian inference of expensive simulators: survey and comparison of design choices. doi:10.1007/s11222-026-11015-3 | S&C 36:255 | `soldat2026_expensive_simulators_preprint.pdf` (Research Square preprint) | Finds the choice of surrogate response variable governs efficiency and that a scalar log-likelihood is a poor choice. Related work. |
| 3 | Säilynoja, Schmitt, Bürkner, Vehtari (2026), Posterior SBC: simulation-based calibration checking conditional on data. doi:10.1007/s11222-026-10825-9 | S&C 36:78 | `sailynoja2026_posterior_SBC_StatComput.pdf` | Calibration conditional on one observed data set, against our prior-draw SBC. SBC paragraph. |
| 4 | Fan, White (2026), Neural posterior estimation on exponential random graph models: evaluating bias and implementation challenges. doi:10.1007/s11222-026-10896-8 | S&C 36:141 | `fanwhite2026_NPE_ERGM_StatComput.pdf` | NPE that is biased yet reproduces observed summaries. Related work (calibration). |
| 5 | Hoffmann, Onnela (2026), Unifying summary statistic selection for approximate Bayesian computation. doi:10.1007/s11222-025-10808-2 | S&C 36(2) | `hoffmann2026_summary_statistic_selection_StatComput.pdf` | Summary-statistic selection for ABC. Related work. |
| 6 | Dinh, Liu, Xiang, Liu, Tavaré (2025), ABC sequential Monte Carlo via random forests. doi:10.1007/s11222-025-10748-x | S&C 35(6) | `dinh2025_ABC_SMC_random_forests_StatComput.pdf` | Recent S&C ABC work on the statistic and proposal. Related work. |
| 7 | Wang, Evans, Nott (2026), Robust Bayesian methods using amortized simulation-based inference. doi:10.1007/s11222-026-10998-3 | S&C 36:233 | `wang2026_robust_amortized_SBI_StatComput.pdf` (arXiv version) | Amortized SBI over a class of priors. Prior-change experiment. |
| 8 | Perrin, Marco, Soize, Funfschilling (2026), Estimating intractable posterior distributions through Gaussian process regression and Metropolis-adjusted Langevin procedure. doi:10.1007/s11222-026-10949-y | S&C 36:198 | `perrin2026_GP_regression_MH_StatComput.pdf` (arXiv version) | GP surrogate of the log-likelihood with a gradient-based sampler. Related work; motivates a gradient-based sampler for our surrogate. |
| 9 | Zammit-Mangion, Sainsbury-Dale, Huser (2025), Neural methods for amortized inference. doi:10.1146/annurev-statistics-112723-034123 | Annu Rev Stat Appl 12:311 | **missing** | Review of amortized neural inference. Related work (NPE). |
| 10 | Frazier, Drovandi, Kock, Nott (2025), Pooling information in likelihood-free inference. doi:10.1214/25-ba1572 | Bayesian Analysis (advance publication) | **missing** (hCaptcha) | Pools posteriors across summaries instead of choosing one. Related work. |
| 11 | Nguyen, Tran, Drovandi, Nott (2025), Wasserstein Gaussianization and efficient variational Bayes for robust Bayesian synthetic likelihood. doi:10.1080/10618600.2025.2520581 | JCGS 35:212 | **missing** (no open-access PDF found) | Synthetic likelihood's Gaussian assumption can fail; our exact baseline and surrogates share it. Synthetic-likelihood paragraph. |
| 12 | Peng, Genest, Stephens (2026), Sequential neural posterior estimation in simulation-based Bayesian inference. doi:10.1214/26-ss158 | Statistics Surveys 20:242 | **missing** (hCaptcha) | Survey of sequential NPE, which we excluded from the NPE comparison. NPE-candidates paragraph. |
| 13 | Modrák, Moon, Kim, et al. (2025), Simulation-based calibration checking for Bayesian computation: the choice of test quantities shapes sensitivity. doi:10.1214/23-ba1404 | Bayesian Analysis 20:461 | **missing** (hCaptcha) | The SBC variant we use. SBC paragraph. |

## To add by hand (five PDFs)

Project Euclid shows a CAPTCHA to scripted browsers and was not bypassed, and two papers have no open-access PDF. Download them in a normal browser (VT library login for #9 and #11) and save them in this folder with these names:

* `frazier2025_pooling_LFI_BayesAnal.pdf` (#10)
* `peng2026_sequential_NPE_StatSurveys.pdf` (#12)
* `modrak2025_SBC_BayesAnal.pdf` (#13)
* `zammit2025_neural_amortized_inference_AnnuRev.pdf` (#9)
* `nguyen2025_robust_BSL_JCGS.pdf` (#11)

Items 2, 7 and 8 are preprint or arXiv versions of the published S&C papers. Replace them with the journal PDFs when available.

## Considered and dropped (not directly relevant enough)

Li, Smith, Zhou (2026, S&C), generalized Markov chain importance sampling (multiple-try Metropolis, not our importance-resampling check); Stockman et al. (2024, S&C), SB-ETAS (an earthquake application); Rimella, Jewell, Fearnhead (2025, S&C), simulation-based composite likelihood (hidden Markov models); Kucharský, Bürkner (2026, S&C), amortized mixture models.

## Older papers

The earlier collection of mostly pre-2024 papers (Lu et al. 2023, Meeds and Welling 2014, Papamakarios et al., and others) is in `../../DNN_ABC_local_extras/lit_review/`. It is outside this folder because it does not meet the 2024-or-later rule.

## Find these later: the five missing papers (full citations and links)

Save each PDF in this folder under the filename shown. All five are already cited in `manuscript.tex`; the key is the BibTeX key in `manuscript.bib`.

1. **Frazier, D. T., Drovandi, C., Kock, L., Nott, D. J. (2025).** Pooling information in likelihood-free inference. *Bayesian Analysis* (advance publication). https://doi.org/10.1214/25-ba1572
   Key `frazier2025`. File: `frazier2025_pooling_LFI_BayesAnal.pdf`. Open access on Project Euclid (needs a normal browser; it shows a CAPTCHA to scripts).
2. **Peng, R., Genest, C., Stephens, D. A. (2026).** Sequential neural posterior estimation in simulation-based Bayesian inference. *Statistics Surveys* 20, 242-283. https://doi.org/10.1214/26-ss158
   Key `peng2026`. File: `peng2026_sequential_NPE_StatSurveys.pdf`. Open access on Project Euclid (same CAPTCHA).
3. **Modrak, M., Moon, A. H., Kim, S., et al. (2025).** Simulation-based calibration checking for Bayesian computation: the choice of test quantities shapes sensitivity. *Bayesian Analysis* 20, 461-488. https://doi.org/10.1214/23-ba1404
   Key `modrak2025`. File: `modrak2025_SBC_BayesAnal.pdf`. Open access on Project Euclid (same CAPTCHA).
4. **Zammit-Mangion, A., Sainsbury-Dale, M., Huser, R. (2025).** Neural methods for amortized inference. *Annual Review of Statistics and Its Application* 12, 311-335. https://doi.org/10.1146/annurev-statistics-112723-034123
   Key `zammit2025`. File: `zammit2025_neural_amortized_inference_AnnuRev.pdf`. No open-access PDF found; use the VT library login.
5. **Nguyen, N.-M., Tran, M.-N., Drovandi, C., Nott, D. (2025).** Wasserstein Gaussianization and efficient variational Bayes for robust Bayesian synthetic likelihood. *Journal of Computational and Graphical Statistics* 35(1), 212-228. https://doi.org/10.1080/10618600.2025.2520581
   Key `nguyen2025`. File: `nguyen2025_robust_BSL_JCGS.pdf`. No open-access PDF found; use the VT library login.

What the manuscript says about each (so you can check it against the paper when you have it):
* `frazier2025`: pools posteriors from several summary vectors instead of choosing one (related work, on summary statistics).
* `peng2026`: surveys sequential NPE (cited where sequential methods are excluded from the NPE comparison).
* `modrak2025`: the SBC variant with data-dependent test quantities (cited with the SBC references).
* `zammit2025`: reviews amortized neural inference (related work on NPE).
* `nguyen2025`: synthetic likelihood's Gaussian assumption can fail, motivating Gaussianising transformations (cited in the synthetic-likelihood paragraph).
These claims come from the abstracts only; none of the five has been read in full.
