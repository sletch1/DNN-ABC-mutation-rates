# Literature review: architecture choice for NN surrogates in ABC / simulation-based inference

Pulled while deciding whether CNN/RNN/LSTM surrogates are worth building alongside
the existing FFN for the 1-D and 3-D studies (see `../manuscript.tex`,
`../DNN_Models/*/network/model.py`). Six papers, one file each below. All
bibliographic details (authors, venue, volume/issue/pages, DOI) were pulled
directly from the publisher/arXiv page, not from memory.

## The one finding that settles the architecture question

`bayesflow.md` (Radev et al. 2020, §2.4) states it plainly: **the summary-network
architecture should match the probabilistic symmetry of the raw data** — LSTM or
1D-CNN for time series (sequential structure to exploit), a permutation-invariant
pooling network for i.i.d./exchangeable samples, and implicitly a plain
feedforward network otherwise. Our surrogate's input, `(log10 p1, log10 p2, tau,
[log10 p_eff])`, is none of these: it's a fixed-size tuple of physically distinct,
non-exchangeable, non-sequential scalars with no natural adjacency between
components. This is the direct justification for why CNN/RNN/LSTM are expected to
be structurally mismatched here rather than just "worth a try" — and why we build
them anyway for the 3-D study (long enough to be non-degenerate) but not the 1-D
study (a length-1 input makes Conv1d/RNN/LSTM degenerate to a plain `Linear`
layer — see `../DNN_Models/3D/results/arch_families/` once populated).

## Papers

| File | Authors | Venue | What it's cited for |
|---|---|---|---|
| [`sheehan_song_2016.md`](sheehan_song_2016.md) | Sheehan & Song | PLOS Comp Biol 2016 | FFN-on-summary-statistics is the established baseline approach in this exact domain (population genetics); DL vs. ABC speed/accuracy table |
| [`flagel_brandvain_schrider_2019.md`](flagel_brandvain_schrider_2019.md) | Flagel, Brandvain, Schrider | Mol. Biol. Evol. 2019 | CNN *does* help when the input is genuinely image-like (genotype alignment matrices) — the structural contrast case to our tabular input |
| [`jiang_et_al_2017.md`](jiang_et_al_2017.md) | Jiang, Wu, Zheng, Wong | Statistica Sinica 2017 | Earliest FFN-for-ABC-summary-statistics paper; DNN output approximates posterior means directly |
| [`akesson_et_al_2021.md`](akesson_et_al_2021.md) | Åkesson, Singh, Wrede, Hellander | IEEE/ACM TCBB 2021 | CNN as a *learned* summary-statistic extractor for ABC in systems biology; explicit CNN-vs-FFN-vs-partially-exchangeable-net comparison |
| [`radev_et_al_bayesflow_2020.md`](radev_et_al_bayesflow_2020.md) | Radev, Mertens, Voss, Ardizzone, Köthe | arXiv 2020 | **The architecture-selection principle this whole comparison is built on** (§2.4, quoted in full) |
| [`cranmer_brehmer_louppe_2020.md`](cranmer_brehmer_louppe_2020.md) | Cranmer, Brehmer, Louppe | PNAS 2020 | Broad framing citation for simulation-based inference as a field |

## How these get used

- `manuscript.tex`: cited in a new discussion subsection comparing FFN/CNN/RNN/LSTM
  for the 3-D surrogate (added once `DNN_Models/3D/results/arch_families/` results
  land from the background build).
- Not cited for the 1-D study, since CNN/RNN/LSTM were not built there (see
  reasoning above and in `manuscript.tex`'s discussion section once added).
