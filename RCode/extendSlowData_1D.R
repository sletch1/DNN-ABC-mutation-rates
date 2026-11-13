# extendSlowData_1D.R
# Extends slow_data_1D.csv upward in p, to fix a prior-truncation artifact:
# the ABC-MCMC prior's upper bound (log10 p = -2.0) coincided EXACTLY with
# the p = 1e-2 grid cell's true value, so the sampler could never propose
# theta above the truth there and every method's 95% credible interval
# measured ~0% posterior coverage at that cell -- not a bug in any one
# method, but a structural artifact of the experimental grid (see
# results/logs/coverage-boundary-note if present, or the relevant manuscript
# discussion).
#
# Fix: extend the ground-truth grid 9 points further up in log10(p), using
# the SAME step size (0.06) as the original 101-point grid
# (genSlowData_1D.R: seq(-8, -2, length.out = 101)), so the combined grid is
# evenly spaced from -8 to -1.46 with no gap or density change at the
# join. This lets the surrogates be retrained with real data above p = 1e-2,
# and the ABC-MCMC prior's upper bound to move to -1.5 -- comfortably
# interior for every existing grid truth, including the p = 1e-2 cell (0.5
# log10-units of headroom, versus 0 before).
#
# Seeds continue the original indexing (i = 102..110) so they cannot collide
# with the existing 101 points' seeds (1000*i + r for i = 1..101).
#
# Output: appends to the same file genSlowData_1D.R writes,
# Results/slow_data_1D.csv (relative to this script's directory).

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0) script_dir <- "."
source(file.path(script_dir, "funMBP.R"))
library(parallel)

Z0 <- 1; a <- 1; delta <- 1; c <- 20; J <- 100; nrep <- 10

# Continues the original seq(-8, -2, length.out = 101) (step = 0.06) upward.
new_logp <- seq(-1.94, -1.46, by = 0.06)
new_p <- 10 ^ new_logp
n_new <- length(new_p)
start_index <- 102  # continues the original i = 1..101 indexing

cat(sprintf("Extending with %d new p values: %s\n", n_new,
            paste(sprintf("%.4g", new_p), collapse = ", ")))

n_cores <- max(1, detectCores() - 2)

outfile <- file.path(script_dir, "..", "Models", "1D", "data", "slow_data_1D.csv")
stopifnot(file.exists(outfile))  # this script only appends; it never creates

run_one_p <- function(k) {
  i <- start_index + k - 1
  p <- new_p[k]
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
  cat(sprintf("[new p %d/%d] p = %.4e, tp = %.4f done\n", k, n_new, p, tp))
  do.call(rbind, rows)
}

runt <- system.time({
  results_list <- mclapply(1 : n_new, run_one_p, mc.cores = n_cores)
})
cat("Extension elapsed time:\n"); print(runt)

result_mat <- do.call(rbind, results_list)
colnames(result_mat) <- c("Z0", "a", "delta", "p", "tp", "J", "rep", "d_bar", paste0("d_", 1 : J))

write.table(result_mat, outfile, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE)
cat(sprintf("Done. %d rows appended to %s\n", nrow(result_mat), outfile))
