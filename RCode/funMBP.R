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