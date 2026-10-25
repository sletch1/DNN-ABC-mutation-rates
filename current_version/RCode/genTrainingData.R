# genTrainingData.R
# ---------------------------------------------------------------------------
# THE QUICK LOOK. A deliberately small, fast version of the training-data sweep
# whose only job is to plot the simulator's input/output relationship so you can
# eyeball it before committing to a real run.
#
# Use this to sanity-check that the simulator behaves before spending hours in
# genSlowData_1D.R / genSlowData_3D.R, which do the same sweep at full
# resolution, in parallel, and write a CSV.
#
# It sweeps the mutation rate p over log10(p) in [-8, -2] (11 points, 5
# replicates each -- versus 101 x 10 for the real run), and at each p:
#   - solves for the plating time tp giving a fixed expected mutant count (20),
#   - runs a J = 100 culture fluctuation experiment via funMBP.R,
#   - reduces it to the summary statistic d_bar = mean_i sqrt(X_i / Z_i).
#
# Output is a single scatter of log10(p) against d_bar. It should be smooth and
# monotone increasing; if it is not, something is wrong upstream in funMBP.R.
# Nothing is written to disk except that plot -- no CSV, no model.
# ---------------------------------------------------------------------------

workpath <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(workpath) == 0) workpath <- "."
source(file.path(workpath, "funMBP.R"))

a <- 1
c <- 20
J <- 100
Z0 <- 1
delta <- 1
use_slow <- FALSE # set TRUE to run the literal cell-splitting simulation (mut_bMBP_slow)
logp_vec <- seq(-8, -2, length.out = 11)#101)
p_vec <- 10 ^ (logp_vec)
np <- length(p_vec)
nrep <- 5#10
input <- rep(logp_vec, each = nrep)
output <- rep(NA, np * nrep)

set.seed(0)
# fast: 11/5/0.08; slow: 11/5/11500
runt <- system.time({
  for (i in 1 : np) {
    p <- p_vec[i]
    myfun <- function(t, Z0, a, p, c) {Z0 * (exp(a * t) - exp(a * t * (1 - 2 * p))) - c}
    tp <- uniroot(myfun, c(1, 30), Z0 = Z0, a = a, p = p, c = c)$root
    for (j in 1 : nrep) {
      data <- fluc_exp1_rev(Z0, a, delta, p, tp, J, use_slow = use_slow)
      Z_vec <- data[[1]]
      X_vec <- data[[2]]
      output[(i - 1) * nrep + j] <- mean(sqrt(X_vec / Z_vec))
    }
  }
})
print(runt)

if (interactive()) {
  x11()
  plot(input, output, type = "p")
} else {
  png(file.path(workpath, "genTrainingData_plot.png"))
  plot(input, output, type = "p")
  dev.off()
}
