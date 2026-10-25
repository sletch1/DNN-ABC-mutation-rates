## This file plots out the mean response surface and standard deviation surface
## of the selected statistic (4th root of mutation freq.) vs. each one/two 
## parameters and finally selects both mean and standard deviation as summary
## stats used in the ABC estimator for more than 2-d parameter space

library(R.matlab)
library(hetGP)
#### -------------- Part 1: constant p and differential growth ------------- ####
# This is a training set generated with the exact simulator countsizeBDtree
# The parameter space is 2-d
init_data <- readMat('init1.mat')

# plot the mean response curve vs. each of the parameters
par(mfrow = c(1,2))
plot(init_data$theta.list[,1], apply(init_data$X,c(1,2),mean), pch = 20,
     xlab = "Delta", ylab = "Mean")
abline(v = log10(0.8),col = "red")
plot(init_data$theta.list[,2], apply(init_data$X,c(1,2),mean), pch = 20,
     xlab = "P", ylab = "Mean")
abline(v = -6, col = "red")

# plot the sd response curve vs. each of the parameters
par(mfrow = c(1,2))
plot(init_data$theta.list[,1], apply(init_data$X,c(1,2),sd), pch = 20,
     xlab = "Delta", ylab = "Standard deviation")
abline(v = log10(0.8),col = "red")
plot(init_data$theta.list[,2], apply(init_data$X,c(1,2),sd), pch = 20,
     xlab = "P", ylab = "Standard deviation")
abline(v = -6, col = "red")

## Plot the 2-d response surface of 
## ----------------- 1. Mean 
y = apply(init_data$X,c(1,2),mean)
# The parameters for fitting GP model is rescaled to a unit interval
X = init_data$theta.list.s

gpfit <- mleHomGP(X,y,lower =c(0.1,0.1), upper = c(4,4), init = list(theta_init = c(1,1)))
par(mfrow = c(1,2))
X.fit <- as.matrix(expand.grid(seq(0,1,len=10),seq(0,1,len=10)), ncol =2)
y.fit <- predict(gpfit,X.fit)
image(seq(0,1,len=10),seq(0,1,len=10), matrix(y.fit$mean,ncol=10), xlab = "Delta",
      ylab = "P", main = "Mean")
points(1+log10(0.8), 0.5, col="green", pch = 4)
## ----------------- 2. sd 
y = apply(init_data$X,c(1,2),sd)
X = init_data$theta.list.s

gpfit <- mleHomGP(X,y,lower =c(0.1,0.1), upper = c(4,4), init = list(theta_init = c(1,1)))
X.fit <- as.matrix(expand.grid(seq(0,1,len=10),seq(0,1,len=10)), ncol =2)
y.fit <- predict(gpfit,X.fit)
image(seq(0,1,len=10),seq(0,1,len=10), matrix(y.fit$mean,ncol=10), xlab = "Delta",
      ylab = "P", main = "Standard deviation")
points(1+log10(0.8), 0.5, col="green", pch = 4)

#### -------- Part 2: Stage-wise mut. prob. and differential growth ------- ####
# This is a training set generated from the exact simulator countsizeBDtree2
# The paramter space is 4-d, in the order of delta, p1, p2 and tau
init_data <- readMat('init2.mat')

# plot the mean response curve vs. each of the parameters
par(mfrow=c(2,2))
xNames = c("delta","p1","p2","tau")
for(i in 1:4){
  plot(init_data$theta.list.s[,i], apply(init_data$X,c(1,2),mean), pch = 20,
       xlab = xNames[i],ylab = "Mean")
}
# plot the sd response curve vs. each of the parameters
for(i in 1:4){
  plot(init_data$theta.list.s[,i], apply(init_data$X,c(1,2),sd), pch = 20,
       xlab = xNames[i],ylab = "Sd")
}

plot(init_data$theta.list[,1], apply(init_data$X,c(1,2),mean), pch = 20)

## Plot the 2-d response surface of 
## ----------------- Mean 
y = apply(init_data$X,c(1,2),mean)
## ----------------- Sd
y = apply(init_data$X,c(1,2),sd)
# The parameters for fitting GP model is rescaled to a unit interval
X = init_data$theta.list.s

gpfit <- mleHomGP(X,y,lower =c(0.1,0.1,0.1,0.1), upper = c(4,4,4,4), init = list(theta_init = c(1,1,1,1)))
X.fit <- as.matrix(cbind(log10(.8) + 1,expand.grid(seq(0,1,len=10),seq(0,1,len=10)),(4-0.1)/13.2), ncol =4)
y.fit <- predict(gpfit,X.fit)
image(seq(0,1,len=10),seq(0,1,len=10), matrix(y.fit$mean,ncol=10), xlab = "p1",
      ylab = "p2")
points((2+log10(5))/6,0.5,col = "green", pch = 4)

X.fit <- as.matrix(expand.grid(seq(0,1,len=10),seq(0,1,len=10)), ncol =2)
X.fit <- cbind(X.fit[,1], (2+log10(5))/6, X.fit[,2],0.9/13.2)
y.fit <- predict(gpfit,X.fit)
image(seq(0,1,len=10),seq(0,1,len=10), matrix(y.fit$mean,ncol=10), xlab = "delta",
      ylab = "p2")
points(log10(.8) + 1, 0.5,col = "green", pch = 4)

X.fit <- as.matrix(expand.grid(seq(0,1,len=10),seq(0,1,len=10)), ncol =2)
X.fit <- cbind(log10(.8) + 1, (2+log10(5))/6, X.fit[,2], X.fit[,1])
y.fit <- predict(gpfit,X.fit)
image(seq(0,1,len=10),seq(0,1,len=10), matrix(y.fit$mean,ncol=10), xlab = "tau",
      ylab = "p2")
points(3.9/13.2, 0.5,col = "green", pch = 4)

X.fit <- as.matrix(expand.grid(seq(0,1,len=10),seq(0,1,len=10)), ncol =2)
X.fit <- cbind(X.fit[,1], (2+log10(5))/6, 0.5, X.fit[,2])
y.fit <- predict(gpfit,X.fit)
image(seq(0,1,len=10),seq(0,1,len=10), matrix(y.fit$mean,ncol=10), xlab = "delta",
      ylab = "tau")
points(log10(.8) + 1,3.9/13.2,col = "green", pch = 4)
