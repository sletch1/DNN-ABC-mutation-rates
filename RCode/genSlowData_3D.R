# genSlowData_3D.R
# Generates REAL ("slow"/exact) training data for the NN surrogate, 3-D case:
# the TWO-STAGE mutation model with parameters (p1, p2, tau).
#
#     p(t) = p1  for 0 < t <= tau
#            p2  for tau < t <= tp
#
# Exact cell-by-cell simulation, a direct R port of the professor's MATLAB
# reference DNN_Models/3D/matlab/mut2stage_bMBP.m (see mut2stage_bMBP_slow in
# funMBP.R). This REPLACES the previous 3-D design, which varied (p, a, delta)
# under a constant mutation rate -- the wrong model, and one whose `a` axis was
# analytically non-identifiable. See ../updates.md.
#
# Fixed: Z0 = 1, a = 1 (never a parameter in the JTB paper), J = 100, tp = 10.
#   tp = 10 is the professor's own value in runsimu.m and keeps the EXACT
#   simulator affordable (~2.2e4 cells/culture). The paper's Study 2 value of
#   tp = 20 gives ~5e8 cells/culture and is not reachable exactly -- that regime
#   needs the fast (Algorithm 4) two-stage simulator, a separate job.
#
# Design: Latin hypercube over (log10 p1, log10 p2, tau), which is what the
# paper uses (and what a 3-D design needs to avoid the curse of dimensionality
# a full factorial grid runs into). nrep replicates per design point, so the
# downstream split-by-replicate scheme leaves no leakage.
#
# Ranges: log10(p1), log10(p2) in [-5, -1.3]; tau in [0.1, 9.9].
#   With tp fixed at 10 the expected mutant count per culture is ~ Z*p ~ 2.2e4*p,
#   so p below ~1e-5 makes almost every culture mutant-free and d_bar collapses
#   to 0. This range keeps every design point informative.
#
# Output: DNN_Models/3D/data/slow_data_3D.csv
# Row format: Z0,a,delta,p1,p2,tau,tp,J,design,rep,d_bar,d_1,...,d_J
#   matching DNN_Models/1D/data/slow_data_1D.csv, with the single `p` column
#   replaced by (p1, p2, tau) and a `design` index identifying the LHS point.
#   `delta` is carried as a constant 1 purely for format parity with the 1-D
#   file -- the two-stage model has no differential mutant growth rate.
#   d_i = sqrt(X_i / Z_i) for culture i (extinct cultures, Z_i = 0, get d_i = 0)
#
# Usage:
#   Rscript genSlowData_3D.R                       # full run, defaults below
#   Rscript genSlowData_3D.R --ndesign 20 --nrep 2 # smoke test
#   Rscript genSlowData_3D.R --mut-time parent --out convention_check_3D.csv

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0) script_dir <- "."
source(file.path(script_dir, "funMBP.R"))
library(parallel)

# -------------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[i + 1]
}

ndesign  <- as.integer(getarg("--ndesign", 2000))   # LHS design points
nrep     <- as.integer(getarg("--nrep", 10))        # replicates per design point
# --limit runs only the FIRST n design points of the (unchanged) --ndesign LHS.
# This is what makes a paired comparison possible: the design is a function of
# (ndesign, seed), so shrinking --ndesign would silently produce a different set
# of (p1, p2, tau) values. Keep --ndesign fixed and use --limit to run a subset
# whose parameters and per-point seeds match the full run exactly.
limit    <- as.integer(getarg("--limit", ndesign))
mut_time <- getarg("--mut-time", "offspring")       # see funMBP.R / the .m file
outname  <- getarg("--out", "slow_data_3D.csv")
seed0    <- as.integer(getarg("--seed", 1))
# Core count: default to every detected core bar one, so the box is saturated
# without starving the parent process. Honour the scheduler/cgroup allocation
# when there is one (detectCores() reports physical hardware and will happily
# over-subscribe a shared or containerised node), and never exceed the number
# of design points.
default_cores <- function() {
  n <- detectCores(logical = TRUE)
  aff <- suppressWarnings(try(length(parallel::mcaffinity()), silent = TRUE))
  if (!inherits(aff, "try-error") && is.numeric(aff) && aff >= 1) n <- min(n, aff)
  for (v in c("SLURM_CPUS_PER_TASK", "NSLOTS", "OMP_NUM_THREADS")) {
    e <- suppressWarnings(as.integer(Sys.getenv(v, NA)))
    if (!is.na(e) && e >= 1) n <- min(n, e)
  }
  max(1, n - 1)
}
n_cores  <- as.integer(getarg("--cores", default_cores()))

# -------------------------------------------------------------------------
# Fixed parameters
# -------------------------------------------------------------------------
Z0    <- 1
a     <- 1
delta <- 1     # format parity only; the two-stage model has no delta
J     <- 100
tp    <- 10

logp_lo <- -5.0; logp_hi <- -1.3
tau_lo  <-  0.1; tau_hi  <-  9.9

# -------------------------------------------------------------------------
# Latin hypercube design over (log10 p1, log10 p2, tau)
# -------------------------------------------------------------------------
# Base-R LHS: split each axis into `ndesign` equal strata, take one uniform
# draw inside each stratum, then independently permute the strata per axis.
# This guarantees one-dimensional uniformity on every axis (which a random
# sample does not) without needing the `lhs` package on the server.
lhs_design <- function(n, mins, maxs, seed) {
  set.seed(seed)
  d <- length(mins)
  out <- matrix(0, nrow = n, ncol = d)
  for (k in 1 : d) {
    strata <- (sample(n) - 1 + runif(n)) / n   # permuted strata + jitter
    out[, k] <- mins[k] + strata * (maxs[k] - mins[k])
  }
  out
}

D <- lhs_design(ndesign, c(logp_lo, logp_lo, tau_lo), c(logp_hi, logp_hi, tau_hi), seed0)
p1_vec  <- 10 ^ D[, 1]
p2_vec  <- 10 ^ D[, 2]
tau_vec <- D[, 3]

cat(sprintf("Two-stage 3-D ground truth: (p1, p2, tau), mut_time = '%s'\n", mut_time))
cat(sprintf("Fixed: Z0=%d a=%g delta=%g J=%d tp=%g\n", Z0, a, delta, J, tp))
cat(sprintf("LHS: %d design points (running %d) x %d reps = %d rows; log10(p) in [%g, %g], tau in [%g, %g]\n",
            ndesign, limit, nrep, limit * nrep, logp_lo, logp_hi, tau_lo, tau_hi))
cat(sprintf("Using %d cores (of %d detected)\n", n_cores, detectCores()))

outdir <- file.path(script_dir, "..", "DNN_Models", "3D", "data")
if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
outfile <- file.path(outdir, outname)

# -------------------------------------------------------------------------
# Per-design-point worker: nrep replicates of a J-culture fluctuation experiment
# -------------------------------------------------------------------------
run_one_design <- function(idx) {
  p1  <- p1_vec[idx]
  p2  <- p2_vec[idx]
  tau <- tau_vec[idx]

  rows <- vector("list", nrep)
  for (r in 1 : nrep) {
    set.seed(1000 * idx + r)
    data <- fluc_exp_2stage(Z0, a, p1, p2, tau, tp, J, mut_time)
    Z_vec <- data[[1]]
    X_vec <- data[[2]]
    d_vec <- sqrt(X_vec / Z_vec)
    d_vec[is.nan(d_vec)] <- 0
    d_bar <- mean(d_vec)
    rows[[r]] <- c(Z0, a, delta, p1, p2, tau, tp, J, idx, r, d_bar, d_vec)
  }
  if (idx %% 50 == 0 || idx == 1) {
    cat(sprintf("[design %d/%d] p1=%.3e p2=%.3e tau=%.3f done\n", idx, limit, p1, p2, tau))
  }
  do.call(rbind, rows)
}

# -------------------------------------------------------------------------
# Parallel execution across the design
# -------------------------------------------------------------------------
# mc.preschedule = FALSE dispatches design points one at a time as workers free
# up, instead of pre-splitting the design into n_cores fixed chunks up front.
# Prescheduling leaves cores idle at the tail whenever chunks finish unevenly --
# which happens here as soon as anything else lands on the node, or when the
# per-point cost drifts. Each design point is ~10 core-seconds of work, so the
# extra fork per point is negligible against a much better packed schedule.
#
# Results are unaffected: run_one_design seeds itself from (idx, rep), so the
# output is identical no matter what order or on how many cores it runs.
runt <- system.time({
  results_list <- mclapply(1 : limit, run_one_design,
                           mc.cores = n_cores, mc.preschedule = FALSE)
})
cat("Total elapsed time:\n")
print(runt)

# mclapply returns a try-error object for any worker that died (e.g. OOM);
# surface that loudly rather than silently writing a short file.
bad <- which(!vapply(results_list, is.matrix, logical(1)))
if (length(bad) > 0) {
  cat(sprintf("WARNING: %d design points failed (indices: %s)\n",
              length(bad), paste(head(bad, 20), collapse = ", ")))
  results_list <- results_list[-bad]
}

result_mat <- do.call(rbind, results_list)
colnames(result_mat) <- c("Z0", "a", "delta", "p1", "p2", "tau", "tp", "J",
                          "design", "rep", "d_bar", paste0("d_", 1 : J))

write.csv(result_mat, outfile, row.names = FALSE, quote = FALSE)
cat(sprintf("Done. %d rows x %d cols written to %s\n",
            nrow(result_mat), ncol(result_mat), outfile))
cat(sprintf("d_bar: min=%.5f median=%.5f max=%.5f ; zero-d_bar rows = %d\n",
            min(result_mat[, "d_bar"]), median(result_mat[, "d_bar"]),
            max(result_mat[, "d_bar"]), sum(result_mat[, "d_bar"] == 0)))
