"""Three additional surrogate architecture families for the 3-D two-stage model,
benchmarked against the deployed FFN (`HeteroscedasticMLP` in `model.py`).

MAPS  (log10 p1, log10 p2, tau, [log10 p_eff])  ->  ( mean of log10(d_bar), log
predictive variance ), same contract as `model.py`. Every class here reuses
`gaussian_nll`, `Standardizer` and `add_derived` from `model.py` rather than
duplicating them, and copies `HeteroscedasticMLP.forward`'s exact soft-clamped
logvar pattern (softplus toward each bound, not a hard clamp) into its own
forward method.

THE HONEST FRAMING THIS FILE IS BUILT AROUND
----------------------------------------------
The deployed FFN treats its four inputs as an unordered feature vector, which is
what they actually are: `log10 p1`, `log10 p2`, `tau` and the derived
`log10 p_eff` are four independent physical scalars with no spatial or temporal
relationship to one another. Swapping the order of the first two columns does
not change the physics; a plain MLP's dot products don't care what order the
inputs arrive in either, so that's a fine match.

The three architectures below do not have that property.

  - A 1-D convolution learns kernels that look at LOCAL, ORDERED neighbourhoods
    of positions (position 1 is adjacent to position 2, not to position 4). That
    is exactly right for a time series or an image row, where nearby positions
    really are physically related. Here it is not: there is no sense in which
    `log10 p2` is "next to" `tau`, or in which sliding the same 3-tap filter
    across (p1, p2, tau, p_eff) is discovering anything the physics put there.
    We chose a fixed input order (the tuple order used throughout this repo);
    the convolution's inductive bias is a bias toward a structure that is an
    artifact of that arbitrary ordering, not a property of the data.

  - An RNN/LSTM reads its input as a SEQUENCE and carries a hidden state forward
    step by step, which is the right inductive bias when step t genuinely
    depends on step t-1 (a time series, a sentence). Treating four unordered
    physical constants as "four timesteps" imposes a fake temporal dependency:
    the recurrence will happily learn SOME function of the 4-tuple (RNNs are
    universal function approximators given enough width/depth, same as an MLP),
    but the sequential/recurrent MACHINERY it uses to get there is solving a
    problem that isn't in the data. `tau` does not causally follow `log10 p2`
    the way word 2 follows word 1.

So a priori, none of the three should be expected to beat the plain MLP on this
surface, and might do worse if the imposed ordering happens to separate inputs
that need to interact directly (e.g. splitting p1 and p2 with tau in between).
The benchmark script in `architecture_search/benchmark_families.py` tests this
prediction rather than assuming it -- see that file and
`results/arch_families/ARCHITECTURE_FAMILIES.md` for the measured answer, which
this docstring does not presuppose.

PARAMETER BUDGET. All three are sized to land in the same rough ballpark as the
deployed FFN (64-32 GELU, 2,466 parameters; see
`results/model/surrogate_metrics.json`) -- roughly 2,000-10,000 parameters --
because `results/logs/benchmark_round2.md` already showed capacity is not the
binding constraint on this surface above ~700 parameters. The point of this
comparison is structural fit (does the architecture's inductive bias help or
hurt), not a capacity contest.

NO BATCHNORM. Per the repo-wide, hard-won lesson (model.py, README): BatchNorm
measured 11x worse than no-BatchNorm on this kind of smooth regression in the
1-D study, because per-mini-batch statistics inject batch-dependent noise and
bias predictions at domain edges. None of the three classes below use it. Where
a 1-D signal wants any normalisation at all, LayerNorm is used instead
(consistent with the rest of the repo), and it is optional.
"""

import torch
import torch.nn as nn

from model import gaussian_nll, Standardizer, add_derived, FEATURES_ALL  # noqa: F401 (re-exported for callers)

_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}


def _clamp_logvar(logvar, min_logvar, max_logvar):
    """The exact soft-clamp pattern from `HeteroscedasticMLP.forward` (model.py).

    Two back-to-back softplus folds keep logvar inside [min_logvar, max_logvar]
    without a hard clamp, so gradients never vanish at the bounds. Copied
    verbatim (not imported) because it is three lines of tensor ops embedded in
    each class's forward, exactly as the original does it -- there is nothing to
    factor out that model.py itself doesn't already inline.
    """
    logvar = max_logvar - nn.functional.softplus(max_logvar - logvar)
    logvar = min_logvar + nn.functional.softplus(logvar - min_logvar)
    return logvar


class HeteroscedasticCNN1D(nn.Module):
    """Treats the 4-feature input as a single-channel length-4 "signal".

    Shape convention: (batch, in_dim) -> (batch, 1, in_dim) -> Conv1d stack
    (kernel_size=3, padding=1, so length is preserved at every layer despite
    there being only 4 positions) -> flatten -> a linear projection -> the same
    two heteroscedastic heads as the MLP.

    WHY THIS IS THE HONEST TEST, NOT A NATURAL FIT. See the module docstring:
    there is no real spatial structure across (log10 p1, log10 p2, tau,
    log10 p_eff) for a convolution to exploit -- a 3-tap kernel centered on
    position 2 mixes `log10 p2` with its two arbitrary neighbours `log10 p1` and
    `tau` only because of the order we happened to store the columns in. Padding
    keeps the length at 4 throughout so two conv layers can still be stacked on
    an input this short; at length 4 that is already most of the "receptive
    field" the architecture has to offer.

    Kept deliberately small (kernel_size=3, two conv layers, narrow channel
    counts) to land in the ~2,000-10,000 parameter range used across this
    comparison -- see the module docstring on why capacity is not the point.
    """

    def __init__(self, in_dim=4, channels=(16, 16), kernel_size=3, hidden=32,
                 activation="gelu", min_logvar=-12.0, max_logvar=4.0):
        super().__init__()
        act = _ACT[activation]
        pad = kernel_size // 2
        conv_layers = []
        prev_c = 1
        for c in channels:
            conv_layers += [nn.Conv1d(prev_c, c, kernel_size, padding=pad), act()]
            prev_c = c
        self.conv = nn.Sequential(*conv_layers)
        self.proj = nn.Sequential(nn.Linear(prev_c * in_dim, hidden), act())
        self.mean_head = nn.Linear(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)
        self.min_logvar, self.max_logvar = min_logvar, max_logvar

    def forward(self, x):
        h = x.unsqueeze(1)                 # (batch, 1, in_dim) -- one input channel
        h = self.conv(h)                   # (batch, channels[-1], in_dim)
        h = h.flatten(1)                   # (batch, channels[-1] * in_dim)
        h = self.proj(h)
        logvar = _clamp_logvar(self.logvar_head(h), self.min_logvar, self.max_logvar)
        return self.mean_head(h), logvar


class HeteroscedasticRNN(nn.Module):
    """Treats the 4 features as 4 timesteps of a length-1-per-step sequence.

    Shape convention: (batch, in_dim) -> (batch, in_dim, 1) -> a small
    recurrent layer, unrolled across the 4 "timesteps" -> the final hidden
    state -> the same two heteroscedastic heads.

    cell="gru" (default) rather than a vanilla `nn.RNN`: a GRU's gating gives it
    better-conditioned gradients across a recurrence (the vanishing/exploding
    gradient sensitivity of a plain tanh RNN, even over just 4 steps, is a well
    documented failure mode -- Cho et al. 2014 introduced the gate for exactly
    this reason), so it is a strictly safer default at essentially the same
    parameter cost. Pass cell="rnn" to use `nn.RNN` instead if a plain
    Elman-RNN comparison point is wanted.

    WHY THIS IS THE HONEST TEST, NOT A NATURAL FIT. See the module docstring:
    treating (log10 p1, log10 p2, tau, log10 p_eff) as "timestep 1, 2, 3, 4"
    invents a causal/sequential dependency that these four physical scalars do
    not have -- there is no sense in which the value of `tau` is influenced by
    "what came before" in the way a time series or a sentence is. The GRU can
    still, in principle, learn any function of the 4-tuple (a big enough
    recurrent net is also a universal approximator), but it has to do so by
    routing information through a hidden state one arbitrary "timestep" at a
    time, rather than seeing all four inputs at once the way the MLP and the
    CNN's flatten step both do.
    """

    def __init__(self, in_dim=4, hidden_size=32, cell="gru", num_layers=1,
                 min_logvar=-12.0, max_logvar=4.0):
        super().__init__()
        rnn_cls = {"gru": nn.GRU, "rnn": nn.RNN, "lstm": nn.LSTM}[cell]
        self.cell = cell
        self.rnn = rnn_cls(input_size=1, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.mean_head = nn.Linear(hidden_size, 1)
        self.logvar_head = nn.Linear(hidden_size, 1)
        self.min_logvar, self.max_logvar = min_logvar, max_logvar

    def forward(self, x):
        seq = x.unsqueeze(-1)              # (batch, in_dim, 1) -- 4 timesteps of width 1
        out, state = self.rnn(seq)
        h_last = state[0] if isinstance(state, tuple) else state   # LSTM returns (h, c)
        h = h_last[-1]                     # final layer's final hidden state: (batch, hidden)
        logvar = _clamp_logvar(self.logvar_head(h), self.min_logvar, self.max_logvar)
        return self.mean_head(h), logvar


class HeteroscedasticLSTM(HeteroscedasticRNN):
    """Same shape convention as `HeteroscedasticRNN`, using `nn.LSTM`.

    Subclassing rather than duplicating: an LSTM is `HeteroscedasticRNN` with
    `cell="lstm"`, differing only in the extra cell-state `c` that
    `nn.LSTM.forward` returns alongside the hidden state `h` -- handled once in
    the shared `forward` via `isinstance(state, tuple)`. Kept as its own class
    (rather than just documenting `cell="lstm"`) because the task asks for three
    distinct, separately benchmarkable architecture families and this keeps
    `build_family()` and the benchmark script's model list simple and explicit.

    Same honest caveat as `HeteroscedasticRNN` applies, doubly so: an LSTM adds
    gated long-range memory (the cell state `c`) specifically to help a
    recurrence retain information across MANY steps -- its whole reason to
    exist is long sequences. Four steps is not a regime where that machinery
    has anything to do; it is included because the task calls for it as a
    distinct, commonly-used recurrent family, not because there is an a priori
    reason to expect it to help over the GRU (or the MLP) here.
    """

    def __init__(self, in_dim=4, hidden_size=32, num_layers=1,
                 min_logvar=-12.0, max_logvar=4.0):
        super().__init__(in_dim=in_dim, hidden_size=hidden_size, cell="lstm",
                         num_layers=num_layers, min_logvar=min_logvar, max_logvar=max_logvar)


_FAMILIES = {
    "cnn1d": HeteroscedasticCNN1D,
    "rnn": HeteroscedasticRNN,
    "lstm": HeteroscedasticLSTM,
}


def build_family(kind, **kw):
    """Factory for the three new architecture families, parallel to `model.build()`.

    Kept separate from `model.py`'s `build()` (rather than merging into it) so
    that `model.py` -- which backs the already-deployed, already-reported FFN
    surrogate -- stays byte-for-byte untouched. `train.py`'s `build(kind="mlp",
    ...)` calls are unaffected; new code should call `build_family("cnn1d", ...)`
    / `build_family("rnn", ...)` / `build_family("lstm", ...)` instead.
    """
    return _FAMILIES[kind](**kw)
