c# 1. Set environmental variable BEFORE calling any library
Sys.setenv(PYTHONHASHSEED = "0")

library(keras3)
library(tensorflow)
library(abind)

# 2. Set seeds for all backend engines
set.seed(1)
tensorflow::set_random_seed(0)
reticulate::py_set_seed(0)

# 3. Force Determinism (Can significantly slow down training)
Sys.setenv(TF_DETERMINISTIC_OPS = "1")
Sys.setenv(TF_CUDNN_DETERMINISTIC = "1")

workpath <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(workpath) == 0) workpath <- "."
source(file.path(workpath, "funMBP.R"))

a <- 1
c <- 20
J <- 100
Z0 <- 1
delta <- 1
use_slow <- FALSE # set TRUE to run the literal cell-splitting simulation (mut_bMBP_slow)
logp_vec <- seq(-8, -2, length.out = 101)
p_vec <- 10 ^ (logp_vec)
np <- length(p_vec)
nrep <- 10
input <- rep(logp_vec, each = nrep)
output <- rep(NA, np * nrep)

# set.seed(0)
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

build_mlp <- function(input_shape) {#32/16/8; 64/16/4
  model <- keras_model_sequential() %>%
    layer_dense(units = 32, activation = "relu", input_shape = input_shape) %>%
    layer_batch_normalization() %>%
    layer_dropout(0.3) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(0.1) %>%
    
    # Output
    layer_dense(units = 1, activation = "linear")
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "mse",
    # loss = variance_preserving_loss,
    metrics = c("mae")
  )
  
  return(model)
}

input.shape <- 1 # 2
mlp_model <- build_mlp(input.shape)

feature <- input[!is.nan(output)]
response <- log10(output[!is.nan(output)]) # use log(Dbar) as response
# feature <- cbind(c(p_mat), c(tp_mat))
# response <- cbind(c(Z_mat), c(X_mat))
nsample <- length(feature)
# train.idx <- sample(1 : nsample, floor(0.8 * nsample))
# test.idx <- setdiff(1 : nsample, train.idx)
train.idx <- 1 : nsample
val.idx <- 1 : nsample
test.idx <- 1 : nsample
x.train <- feature[train.idx]
x.val <- feature[val.idx]
x.test <- feature[test.idx]
y.train <- response[train.idx]
y.val <- response[val.idx]
y.test <- response[test.idx]

cat("Training MLP...\n")
runt <- system.time({
  history <- mlp_model %>% fit(
    x.train, y.train,
    epochs = 200,
    batch_size = 16,  # Smaller batch size for complex data
    validation_split = 0.2,
    verbose = 1,
    callbacks = list(
      callback_early_stopping(patience = 20, restore_best_weights = TRUE),
      callback_reduce_lr_on_plateau(factor = 0.5, patience = 8),
      callback_model_checkpoint("best_model.h5", save_best_only = TRUE)
    )
  )
})

# Predictions
# y.pred <- mlp_model %>% predict(x.test)
y.pred <- mlp_model %>% predict(x.test)
# Conformal confidence interval
data.val <- data.frame(x = x.val, y = y.val, yhat = y.pred)
x.new <- seq(-8, -2, length.out = 1001)
y.new <- mlp_model %>% predict(x.new)
data.new <- data.frame(x = x.new, y = y.new)
alpha <- 0.05
CI <- conformalCI(data.val, data.new, alpha)

if (!interactive()) png(file.path(workpath, "trainNN_fit_diagnostic.png")) else x11()
plot(y.test, y.pred, type = "p", xlab = "log(D.bar)", ylab = "log(Dhat.bar)", asp = 1)
abline(0, 1, col = "red")
if (!interactive()) dev.off()

if (!interactive()) png(file.path(workpath, "trainNN_fit_with_CI.png")) else x11()
plot(x.test, 10 ^ y.test, type = "p", xlab = "log(p)", ylab = "log(D.bar)")
lines(x.new, 10 ^ y.new, col = "red")
for (i in 1 : length(x.new))  {
  lines(c(x.new[i], x.new[i]), c(10 ^ CI[i, 1], 10 ^ CI[i, 2]), col = rgb(0, 1, 0, 0.15))
}
# polygon
legend("topleft", legend = c("training data", "fitted function", "conformal CI"), col = c("black", "red", "green"), pch = c(1, NA, NA), lty = c(0, 1, 1))
if (!interactive()) dev.off()
