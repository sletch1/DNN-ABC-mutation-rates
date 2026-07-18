# genSlowData_1D.R
# Generates REAL ("slow"/exact, Algorithm 2 in Lu, Zhu & Wu 2023) training data
# for the NN surrogate, constant-mutation-rate case (1-D: only p varies).
# Matches the paper's own Fig. 2 design: a = 1, delta = 1 (no differential growth),
# J = 100, log10(p) swept over 101 equally spaced points in [-8, -2], 10 replicates
# per p. Parallelized across the p-grid using parallel::mclapply (fork-based,
# Linux only) to make use of multiple cores on the server.
#
# Output: Results/slow_data_1D.csv
# Row format: Z0,a,delta,p,tp,J,rep,d_bar,d_1,...,d_J
#   where d_i = sqrt(X_i / Z_i) for culture i (extinct cultures, Z_i = 0, get d_i = 0)

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0) script_dir <- "."
source(file.path(script_dir, "funMBP.R"))
library(parallel)

# -------------------------------------------------------------------------
# Fixed parameters (matching the paper's constant-mutation-rate design)
# -------------------------------------------------------------------------
Z0    <- 1
a     <- 1
delta <- 1
c     <- 20
J     <- 100
nrep  <- 10

logp_vec <- seq(-8, -2, length.out = 101)
p_vec <- 10 ^ logp_vec
np <- length(p_vec)

n_cores <- max(1, detectCores() - 2)
cat(sprintf("Using %d cores (of %d detected)\n", n_cores, detectCores()))

outdir <- file.path(script_dir, "..", "Results")
if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
outfile <- file.path(outdir, "slow_data_1D.csv")

# -------------------------------------------------------------------------
# Per-p worker: solve tp, run nrep replicates of the slow simulator, return rows
# -------------------------------------------------------------------------
run_one_p <- function(i) {
  p <- p_vec[i]
  myfun <- function(t) Z0 * (exp(a * t) - exp(a * t * (1 - 2 * p))) - c
  tp <- uniroot(myfun, c(1, 30), extendInt = "yes")$root

  rows <- vector("list", nrep)
  for (r in 1 : nrep) {
    set.seed(1000 * i + r)
    data <- fluc_exp1_rev(Z0, a, delta, p, tp, J, use_slow = TRUE)
    Z_vec <- data[[1]]
    X_vec <- data[[2]]
    d_vec <- sqrt(X_vec / Z_vec)
    d_vec[is.nan(d_vec)] <- 0
    d_bar <- mean(d_vec)
    rows[[r]] <- c(Z0, a, delta, p, tp, J, r, d_bar, d_vec)
  }
  cat(sprintf("[p-index %d/%d] p = %.3e, tp = %.4f done\n", i, np, p, tp))
  do.call(rbind, rows)
}

# -------------------------------------------------------------------------
# Parallel execution across the p-grid
# -------------------------------------------------------------------------
runt <- system.time({
  results_list <- mclapply(1 : np, run_one_p, mc.cores = n_cores)
})
cat("Total elapsed time:\n")
print(runt)

result_mat <- do.call(rbind, results_list)
colnames(result_mat) <- c("Z0", "a", "delta", "p", "tp", "J", "rep", "d_bar", paste0("d_", 1 : J))

write.csv(result_mat, outfile, row.names = FALSE)
cat(sprintf("Done. %d rows written to %s\n", nrow(result_mat), outfile))
