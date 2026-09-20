# Cranmer, Brehmer & Louppe (2020) — The Frontier of Simulation-Based Inference

**Citation:** Cranmer, K., Brehmer, J. & Louppe, G. (2020). The frontier of
simulation-based inference. *Proceedings of the National Academy of Sciences*
117(48): 30055–30062. DOI: [10.1073/pnas.1912789117](https://doi.org/10.1073/pnas.1912789117)

## What it does

A field-level review of simulation-based inference (SBI): the general problem
of Bayesian/frequentist inference for models where the likelihood is
intractable but forward simulation is possible — the same problem class ABC,
GPS-ABC, and our DNN-ABC all sit inside. Surveys the shift from classical ABC
(reject/accept on a summary-statistic distance) toward neural approaches that
learn the likelihood, likelihood ratio, or posterior directly from simulated
`(parameter, data)` pairs, discussing where each approach's approximation
error comes from and how it scales with dimensionality.

## Relevance to our project

Used as the broad framing citation for what field this whole project sits in
— GPS-ABC (Lu, Zhu & Wu 2023) and our neural-surrogate substitution are both
concrete instances of the general SBI program Cranmer, Brehmer & Louppe survey.
Not used for a specific architectural claim (that's `radev_et_al_bayesflow_2020.md`);
this one is cited in the introduction/discussion of the manuscript to situate
the paper within the broader SBI literature rather than presenting the
GPS-ABC-vs-DNN-ABC comparison as if it were happening in isolation.
