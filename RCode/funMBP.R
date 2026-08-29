# funMBP.R
# ---------------------------------------------------------------------------
# THE SIMULATOR LIBRARY. Every piece of ground-truth data in this project comes
# out of this file; nothing else here simulates cells. Sourced by all the
# genSlowData_*.R / genTrainingData.R / trainNN.R scripts.
#
# The biology being modelled: a single cell is dropped into a culture and grows
# into a colony. Cells divide at random times, and at each division a daughter
# may MUTATE. Mutation is irreversible, so a mutant's whole sub-lineage is
# mutant. At a "plating time" tp we count the total population Z and the mutant
# population X. Repeating that over J parallel cultures is a "fluctuation
# experiment", and the mutation rate p has to be inferred from the resulting
# spread of (Z, X) -- that inference is what the rest of the repo does.
#
# Formally this is a two-type Markov branching process (bMBP), which is why the
# function names carry that suffix.
#
# WHAT IS IN HERE, and why there is more than one simulator
# ---------------------------------------------------------------------------
#   mut_bMBP_slow       Constant mutation rate, EXACT. Simulates every cell
#                       division literally. This is ground truth, and it is
#                       expensive: cost grows like exp(a*tp).
#   mut_bMBP_fast       Constant mutation rate, APPROXIMATE. Replaces the
#                       generation-by-generation loop with closed-form
#                       (geometric / Yule) draws. Orders of magnitude faster,
#                       and the only way to reach the tiny mutation rates
#                       (p ~ 1e-9) where the exact version is hopeless.
#   fluc_exp1_rev       Runs either of the above over J parallel cultures.
#
#   mut2stage_bMBP_slow TWO-STAGE mutation rate, EXACT -- the 3-D model.
#   fluc_exp_2stage     J parallel cultures under the two-stage model.
#
#   conformalCI         Split-conformal prediction interval; used to put
#                       calibrated error bars on the neural-net surrogate.
#
# The 1-D vs 3-D split: the 1-D study infers a single constant p. The 3-D study
# uses the two-stage model, where the mutation rate STEPS from p1 to p2 at an
# unknown time tau, and infers all three of (p1, p2, tau). See the block comment
# above mut2stage_bMBP_slow for why that model is the one the paper actually
# uses, and ../updates.md for why the earlier (p, a, delta) 3-D attempt was
# abandoned.
#
# Shared argument conventions across every function below:
#   Z0    initial number of non-mutant cells (1 throughout this project)
#   a     rate of the exponential cell lifetime; larger a = faster division
#   delta mutant growth rate RELATIVE to non-mutants (1 = no growth difference)
#   p     per-division mutation probability
#   tp    plating / checking time at which (Z, X) are recorded
#   J     number of parallel cultures in one fluctuation experiment
# ---------------------------------------------------------------------------

mut_bMBP_slow <- function(Z0, a, delta = 1, p, tp) {
  Z <- 0
  X <- 0
  dtvec <- rexp(Z0, a)
  mvec <- rep(0, Z0)
  f_continue <- (dtvec < tp)
  n_continue <- sum(f_continue);
  Z <- Z + sum(!f_continue)
  X <- X + sum((!f_continue) & (mvec == 1));
  while (n_continue > 0) {
    dtvec_last <- dtvec[f_continue]
    mvec_last <- mvec[f_continue]
    mvec <- rbinom(2 * length(mvec_last), 1, (1 - p) * rep(mvec_last, each = 2) + p)
    rate_vec <- ifelse(mvec == 1, a * delta, a)
    dtvec <- rep(dtvec_last, each = 2) + rexp(2 * n_continue, rate_vec)
    f_continue <- (dtvec < tp)
    n_continue <- sum(f_continue)
    Z <- Z + sum(!f_continue)
    X <- X + sum((!f_continue) & (mvec == 1));
  }
  return(c(Z, X))
}

mut_bMBP_fast <- function(Z0, a, delta, p, tp) {
  Z <- sum(rgeom(Z0, exp(-a * tp)))
  M <- round(Z * p)
  if (M > 0) {
    arrtime_vec <- log(matrix(runif(M), nrow = 1) * (exp(a * tp) - 1) + 1) / a
    X <- sum(rgeom(M, exp(-(a * delta) * (tp - arrtime_vec)))) + M
  } else {
    X <- 0
  }
  return(c(Z, X))
}

fluc_exp1_rev <- function(Z0, a, delta, p, tp, J, use_slow = FALSE) {
  Z_vec <- rep(0, J)
  X_vec <- rep(0, J)
  for (i in 1 : J) {
    if (use_slow) {
      data <- mut_bMBP_slow(Z0, a, delta, p, tp)
    } else {
      data <- mut_bMBP_fast(Z0, a, delta, p, tp)
    }
    Z_vec[i] <- data[1]
    X_vec[i] <- data[2]
  }
  return(list(Z_vec, X_vec))
}

conformalCI <- function(data.val, data.new, alpha) {
  x <- data.val$x
  y <- data.val$y
  yhat <- data.val$yhat
  residuals <- abs(y - yhat)
  q <- quantile(residuals, probs = 1 - alpha)
  x.new <- data.new$x
  y.new <- data.new$y
  lower <- y.new - q
  upper <- y.new + q
  
  return(cbind(lower, upper))
}
# ---------------------------------------------------------------------------
# Two-stage (piecewise-constant) mutation rate -- the 3-D model (p1, p2, tau)
# ---------------------------------------------------------------------------
# Direct R port of the professor's MATLAB reference,
# DNN_Models/3D/matlab/mut2stage_bMBP.m. The mutation probability is a step
# function of time,
#
#     p(t) = p1  for 0 < t <= tau
#            p2  for tau < t <= tp
#
# so the three free parameters are (p1, p2, tau). The division rate `a` is fixed
# (never a parameter in the JTB paper) and there is no differential mutant growth
# rate: the MATLAB signature mut2stage_bMBP(a, mu1, mu2, jmpt, chkt) has no
# delta, and every lifetime is exp(a) regardless of mutation status.
#
# mut_time -- which time indexes p(t); see the note in the .m file.
#   "offspring": p evaluated at the daughter's OWN division time. This is the
#                LIVE code (lines 25-26) in mut2stage_bMBP.m.
#   "parent":    p evaluated at the parent's division time, i.e. the birth time
#                of the daughter -- the division event that actually creates it.
#                This is the COMMENTED-OUT version (lines 23-24), and it is the
#                paper's model.
# The two differ by 3-10% in d_bar, so both are generated until the professor
# confirms which is intended.
mut2stage_bMBP_slow <- function(Z0, a, p1, p2, tau, tp, mut_time = "offspring") {
  Z <- 0
  X <- 0
  dtvec <- rexp(Z0, a)
  mvec <- rep(0, Z0)
  f_continue <- (dtvec < tp)
  n_continue <- sum(f_continue)
  Z <- Z + sum(!f_continue)
  X <- X + sum((!f_continue) & (mvec == 1))
  while (n_continue > 0) {
    dtvec_last <- dtvec[f_continue]
    mvec_last <- mvec[f_continue]
    if (mut_time == "parent") {
      # mutation status drawn from the BIRTH time, before the lifetime is drawn
      birth <- rep(dtvec_last, each = 2)
      mu <- ifelse(birth <= tau, p1, p2)
      mvec <- rbinom(2 * n_continue, 1, (1 - mu) * rep(mvec_last, each = 2) + mu)
      dtvec <- birth + rexp(2 * n_continue, a)
    } else {
      # lifetime drawn first, then mutation status from the offspring's own
      # division time -- bit-for-bit the live lines of mut2stage_bMBP.m
      dtvec <- rep(dtvec_last, each = 2) + rexp(2 * n_continue, a)
      mu <- ifelse(dtvec <= tau, p1, p2)
      mvec <- rbinom(2 * n_continue, 1, (1 - mu) * rep(mvec_last, each = 2) + mu)
    }
    f_continue <- (dtvec < tp)
    n_continue <- sum(f_continue)
    Z <- Z + sum(!f_continue)
    X <- X + sum((!f_continue) & (mvec == 1))
  }
  return(c(Z, X))
}

# J parallel cultures under the two-stage model -> list(Z_vec, X_vec)
fluc_exp_2stage <- function(Z0, a, p1, p2, tau, tp, J, mut_time = "offspring") {
  Z_vec <- rep(0, J)
  X_vec <- rep(0, J)
  for (i in 1 : J) {
    data <- mut2stage_bMBP_slow(Z0, a, p1, p2, tau, tp, mut_time)
    Z_vec[i] <- data[1]
    X_vec[i] <- data[2]
  }
  return(list(Z_vec, X_vec))
}
