# Architecture and capacity benchmarks, 1-D model (Study I)

Logs from the 1-D surrogate studies (copied from `../results/logs/`; scripts in
`../network/architecture_search/`). The 1-D network is unchanged from the original work.

| File | Contents |
|---|---|
| `benchmark_arch.md` | candidate architectures against the GP on the mean-curve fit (curve MSE against the denoised response curve) |
| `benchmark_capacity.md` | how small the network can be: hidden layers from 128-64 down to a single 16-unit layer, curve MSE and downstream ABC accuracy and interval length |
| `benchmark_capacity_confirm.md` | downstream ABC accuracy and interval length at the paper's scale (40 replicates, full grid) for each capacity, with how many of the 9 comparisons to the deployed network are resolved |
| `benchmark_round2.md` | second round of architecture candidates |

The deployed network is the 32-16 GELU network (626 parameters), chosen on parsimony: it
has the best point estimate on curve fit and downstream, though not distinguishably better
than 128-64 or 64-32 (`benchmark_capacity_confirm.md`, 0 of 9 comparisons resolved), and
every candidate from 8 to 128-64 hidden units beats the GP baseline. The decisive result of
the search is that BatchNorm makes an otherwise reasonable 128-64 network fit about eleven
times worse (`benchmark_arch.md`). The manuscript's Study I section (Architecture) gives the argument.
