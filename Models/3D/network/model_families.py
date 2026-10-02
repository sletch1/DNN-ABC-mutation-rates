"""Three additional surrogate architecture families (CNN1D, RNN, LSTM) for
the 3-D two-stage model, benchmarked against the deployed FFN (model.py).
Same contract as model.py: (log10 p1, log10 p2, tau) -> (mean, log
predictive variance); reuses gaussian_nll/Standardizer, copies
HeteroscedasticMLP's soft-clamped logvar pattern.

The honest framing: the deployed FFN treats its 3 inputs as an unordered
feature vector, which is what they physically are (independent scalars, no
spatial/temporal relationship). A 1-D conv's local-neighborhood inductive
bias and an RNN/LSTM's sequential-dependency bias both impose structure
that's an artifact of column ordering, not a property of the data -- so a
priori none of the three should beat the plain MLP here, and might do
worse. architecture_search/benchmark_families.py tests this rather than
assuming it; see results/arch_families/ARCHITECTURE_FAMILIES.md for the
measured answer.

All three sized to the deployed FFN's ballpark (2,466 params) since
benchmark_round2.md already showed capacity isn't the binding constraint
above ~700 params -- this comparison is about structural fit, not capacity.
No BatchNorm (11x worse on this kind of smooth regression per the 1-D
study); LayerNorm optional where used.
"""

import torch
import torch.nn as nn

from model import gaussian_nll, Standardizer, FEATURES_RAW  # noqa: F401 (re-exported for callers)

_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}


def _clamp_logvar(logvar, min_logvar, max_logvar):
    """Soft-clamp pattern from HeteroscedasticMLP.forward (model.py): two
    back-to-back softplus folds keep logvar in [min_logvar, max_logvar]
    without a hard clamp, so gradients never vanish at the bounds.
    """
    logvar = max_logvar - nn.functional.softplus(max_logvar - logvar)
    logvar = min_logvar + nn.functional.softplus(logvar - min_logvar)
    return logvar


class HeteroscedasticCNN1D(nn.Module):
    """Treats the 3-feature input as a single-channel length-3 "signal":
    (batch, in_dim) -> (batch, 1, in_dim) -> Conv1d stack (kernel_size=3,
    padding=1, preserving length) -> flatten -> linear projection -> the
    same two heteroscedastic heads as the MLP. At length 3 a single 3-tap
    kernel already spans every position, so there's essentially no locality
    left to exploit -- the honest test, not a natural fit (see module docstring).
    """

    def __init__(self, in_dim=3, channels=(16, 16), kernel_size=3, hidden=32,
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
        h = x.unsqueeze(1)   # (batch, 1, in_dim)
        h = self.conv(h)     # (batch, channels[-1], in_dim)
        h = h.flatten(1)
        h = self.proj(h)
        logvar = _clamp_logvar(self.logvar_head(h), self.min_logvar, self.max_logvar)
        return self.mean_head(h), logvar


class HeteroscedasticRNN(nn.Module):
    """Treats the 3 features as 3 timesteps of a length-1-per-step sequence:
    (batch, in_dim) -> (batch, in_dim, 1) -> recurrent layer, unrolled over
    3 "timesteps" -> final hidden state -> the same two heteroscedastic
    heads. cell="gru" default (better-conditioned gradients than plain
    nn.RNN even over 3 steps; pass cell="rnn" for a vanilla Elman-RNN).
    Invents a causal/sequential dependency these physical scalars don't
    have -- the honest test, not a natural fit (see module docstring).
    """

    def __init__(self, in_dim=3, hidden_size=32, cell="gru", num_layers=1,
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
        seq = x.unsqueeze(-1)   # (batch, in_dim, 1): one timestep per input
        out, state = self.rnn(seq)
        h_last = state[0] if isinstance(state, tuple) else state   # LSTM returns (h, c)
        h = h_last[-1]   # final layer's final hidden state: (batch, hidden)
        logvar = _clamp_logvar(self.logvar_head(h), self.min_logvar, self.max_logvar)
        return self.mean_head(h), logvar


class HeteroscedasticLSTM(HeteroscedasticRNN):
    """`HeteroscedasticRNN` with cell="lstm" -- subclassed rather than just
    documenting the flag, so it's a distinct, separately benchmarkable
    family for build_family() and the benchmark script's model list. Same
    caveat as HeteroscedasticRNN applies doubly: an LSTM's gated long-range
    memory exists for MANY-step sequences, and 3 steps isn't that regime --
    included because the task calls for a distinct recurrent family, not
    because there's an a priori reason to expect it to help here.
    """

    def __init__(self, in_dim=3, hidden_size=32, num_layers=1,
                 min_logvar=-12.0, max_logvar=4.0):
        super().__init__(in_dim=in_dim, hidden_size=hidden_size, cell="lstm",
                         num_layers=num_layers, min_logvar=min_logvar, max_logvar=max_logvar)


_FAMILIES = {
    "cnn1d": HeteroscedasticCNN1D,
    "rnn": HeteroscedasticRNN,
    "lstm": HeteroscedasticLSTM,
}


def build_family(kind, **kw):
    """Factory for the three new families, parallel to model.build(). Kept
    separate so model.py (the deployed, already-reported FFN) stays
    byte-for-byte untouched; new code calls build_family("cnn1d"/"rnn"/"lstm", ...).
    """
    return _FAMILIES[kind](**kw)
