# Architecture and capacity benchmarks, 1-D model (Study I)

Logs from the 1-D surrogate studies (copied from `../results/logs/`; scripts in
`../network/architecture_search/`). The 1-D network is unchanged from the original work.

| File | Contents |
|---|---|
| `benchmark_arch.md` | candidate architectures against the GP on the mean-curve fit (curve MSE against the denoised response curve) |
| `benchmark_capacity.md` | how small the network can be: hidden layers from 128-64 down to a single 16-unit layer, curve MSE and downstream ABC accuracy and interval length |
| `benchmark_capacity_confirm.md` | downstream ABC accuracy and interval length at the paper's scale (40 replicates, full grid) for each capacity, with how many of the 9 comparisons to the deployed network are resolved |
| `benchmark_round2.md` | second round of architecture candidates |

The deployed network is the two-layer 128-64 GELU network (8,642 parameters). Its
mean-curve MSE is 7.6% above the GP's in `benchmark_capacity.md`, and its downstream ABC
intervals are shorter than the GP's. The confirm run finds smaller networks (down to 66
parameters) not resolvably different from it downstream, so the deployed size is not
shown to be necessary; it was kept unchanged as the original work's model.
