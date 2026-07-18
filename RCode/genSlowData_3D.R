# genSlowData_3D.R
# Generates REAL ("slow"/exact, Algorithm 2 in Lu, Zhu & Wu 2023) training data
# for the NN surrogate, 3-D case: p, a (division rate), and delta (relative
# mutant growth rate) all vary. Extends the paper's constant-mutation-rate
# model (which fixes a = 1, delta = 1) to a 3-parameter surrogate.
#
# Grid: log10(p) in [-8, -2] (10 pts, log-spaced, matching the paper's p range),
# a in [0.5, 2] (10 pts, linear), delta in [0.5, 2] (10 pts, linear; this range
# matches the paper's own domain for delta in Table 4). 10 x 10 x 10 = 1000
# combinations, nrep = 10 replicates each. J = 100 fixed (matching the paper).
# Parallelized across the 1000 grid combinations using parallel::mclapply.
#
# Output: Results/slow_data_3D.csv
# Row format: Z0,a,delta,p,tp,J,rep,d_bar,d_1,...,d_J
#   where d_i = sqrt(X_i / Z_i) for culture i (extinct cultures, Z_i = 0, get d_i = 0)

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0) script_dir <- "."
source(file.path(script_dir, "funMBP.R"))
library(parallel)

# -------------------------------------------------------------------------
# Fixed parameters
# -------------------------------------------------------------------------
Z0   <- 1
c    <- 20
J    <- 100
nrep <- 10

logp_vec  <- seq(-8, -2, length.out = 10)
p_vec     <- 10 ^ logp_vec
a_vec     <- seq(0.5, 2, length.out = 10)
delta_vec <- seq(0.5, 2, length.out = 10)

grid <- expand.grid(p = p_vec, a = a_vec, delta = delta_vec)
ncomb <- nrow(grid)

n_cores <- max(1, detectCores() - 2)
cat(sprintf("Using %d cores (of %d detected)\n", n_cores, detectCores()))
cat(sprintf("Grid: %d p x %d a x %d delta = %d combinations x %d reps = %d total simulated cultures-experiments\n",
            length(p_vec), length(a_vec), length(delta_vec), ncomb, nrep, ncomb * nrep))

outdir <- file.path(script_dir, "..", "Results")
if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
outfile <- file.path(outdir, "slow_data_3D.csv")

# -------------------------------------------------------------------------
# Per-combination worker: solve tp (depends on p, a only), run nrep replicates
# -------------------------------------------------------------------------
run_one_combo <- function(idx) {
  p <- grid$p[idx]
  a <- grid$a[idx]
  delta <- grid$delta[idx]

  myfun <- function(t) Z0 * (exp(a * t) - exp(a * t * (1 - 2 * p))) - c
  tp <- uniroot(myfun, c(1, 30), extendInt = "yes")$root

  rows <- vector("list", nrep)
  for (r in 1 : nrep) {
    set.seed(1000 * idx + r)
    data <- fluc_exp1_rev(Z0, a, delta, p, tp, J, use_slow = TRUE)
    Z_vec <- data[[1]]
    X_vec <- data[[2]]
    d_vec <- sqrt(X_vec / Z_vec)
    d_vec[is.nan(d_vec)] <- 0
    d_bar <- mean(d_vec)
    rows[[r]] <- c(Z0, a, delta, p, tp, J, r, d_bar, d_vec)
  }
  cat(sprintf("[combo %d/%d] p = %.3e, a = %.3f, delta = %.3f, tp = %.4f done\n", idx, ncomb, p, a, delta, tp))
  do.call(rbind, rows)
}

# -------------------------------------------------------------------------
# Parallel execution across the 3-D grid
# -------------------------------------------------------------------------
runt <- system.time({
  results_list <- mclapply(1 : ncomb, run_one_combo, mc.cores = n_cores)
})
cat("Total elapsed time:\n")
print(runt)

result_mat <- do.call(rbind, results_list)
colnames(result_mat) <- c("Z0", "a", "delta", "p", "tp", "J", "rep", "d_bar", paste0("d_", 1 : J))

write.csv(result_mat, outfile, row.names = FALSE)
cat(sprintf("Done. %d rows written to %s\n", nrow(result_mat), outfile))
