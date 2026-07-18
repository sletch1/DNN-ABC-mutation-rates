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
