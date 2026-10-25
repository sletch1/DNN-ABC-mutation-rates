library(R.matlab)

library(ggplot2)
library(ggpubr)
library(LaplacesDemon)
burnin = 5000
ps = 4:8
Jcase = 1:3
N = 100

MSE <- RMSE <- matrix(NA, length(ps) * length(Jcase),3)
runningTime <- rep(NA, length(ps) * length(Jcase))
CILength <- matrix(NA, 5, 3)
for(p in ps){
  for(j in Jcase){
    SE <- matrix(NA, 100, 3)
    time <- l <- u <- rep(NA,100)
    
    for(i in 1:100){
      sample <- try(readMat(paste0('../Result/simulation/simu1/MCMC/p',p,'/sample', j,"_",i,".mat")))
      if(inherits(sample, "try-error"))
      {
        #error handling code, maybe just skip this iteration using
        next
      }
      res <- sample$sample[-(1:burnin),]
      time[i] <- sample$runningTime
      SE[i,1] <- (10^ sample$theta.mu - 10^(-p))^2
      SE[i,2] <- (10^ mean(res)- 10^(-p))^2
      SE[i,3] <- (10^ median(res)- 10^(-p))^2
      l[i] <- 10^quantile(res,0.025)
      u[i] <- 10^quantile(res,0.975)
      
    }
    CILength[p-3,j] <- mean(u-l, na.rm = T)
    runningTime[(p-4)*3 + j] <- mean(time, na.rm = T)
    MSE[((p-4)*3 + j),] = colMeans(SE,na.rm = T)
    RMSE[((p-4)*3 + j),] = sqrt(MSE[((p-4)*3 + j),]) / (10^(-p))
  }
}
RMSE
signif(CILength,3)
write.csv(CILength, file = '../Result/simulation/simu1/CILength_mcmc.csv', row.names = F)
write.csv(RMSE, file = '../Result/simulation/simu1/RMSE.csv', row.names = F)
write.csv(MSE, file = '../Result/simulation/simu1/MSE.csv', row.names = F)
write.csv(runningTime, file = '../Result/simulation/simu1/runningTime.csv', row.names = F)
j = 0
for(i in 1:100){
  sample <- try(readMat(paste0("../Result/psample1_",i,".mat")))
  if(inherits(sample, "try-error"))
  {
    #error handling code, maybe just skip this iteration using
    next
  }
  #rest of iteration for case of no error
  res <- sample$sample[-(1:burnin),]
  j = j + 1
  M[j,] <- c(10^colMeans(res[,1:3]),mean(res[,4]))
  Std[j,] <- c(apply(10^res[,1:3], 2, sd), sd(res[,4]))
  med[j,] <- c(apply(10^res[,1:3], 2, quantile,0.5), quantile(res[,4],0.5))
  Q1[j,] <- c(apply(10^res[,1:3], 2, quantile,0.025), quantile(res[,4],0.025))
  Q3[j,] <- c(apply(10^res[,1:3], 2, quantile,0.975), quantile(res[,4],0.975))
  runningTime[j] <- sample$runningTime
  
  modes <- apply(10^res[,1:3],2,Modes)
  mode_delta[[j]] <- modes[[1]]$modes
  mode_p1[[j]] <- modes[[2]]$modes
  mode_p2[[j]] <- modes[[3]]$modes
  
  mode_tau[[j]] <- Modes(res[,4])
}

colMeans(M)
apply(M,2,sd)
rowMeans((t(M) - matrix(rep(c(0.8,1e-9,1e-8,14),N),nrow=4))^2)
sqrt(rowMeans((t(M) - matrix(rep(c(0.8,1e-9,1e-8,14),N),nrow=4))^2))/colMeans(M)
colMeans(med)
Modes(unlist(mode_p1))
Modes(unlist(mode_p2))
Modes(unlist(mode_tau))
Modes(unlist(mode_delta))
sum(0.8 >= Q1[,1] & 0.8 <= Q3[,1]) / N
sum(1e-9 >= Q1[,2] & 1e-9 <= Q3[,2]) / N
sum(5e-8 >= Q1[,3] & 5e-8 <= Q3[,3]) / N
sum(14 >= Q1[,4] & 14 <= Q3[,4]) / N
colMeans(Q3- Q1)
sd(runningTime)/3600

#### ---------------- For real data --------------------------
setwd("../Result/werngren_v1")
bacName = c("H37Rv", "E865", "E729", "E740", "E1221", "E1449", "Harlingen",
            "E26_95", "E80", "E55", "E26_94", "E3942", "E47")
burnin = 20000

for(i in bacName[12]){
  sample <- readMat(paste0("sample_a10_",i,"_1.mat"))
  sample1 <- readMat(paste0("sample_a10_",i,"_1_m1.mat"))
  sample1a <- readMat(paste0("sample_a10_",i,"_1_m1a.mat"))
  
  data <- readMat(paste0(i,".mat"))
  assign(paste0("MOM_",i),data$MOM)
  assign(paste0("p_init_",i),data$p.init)
  assign(paste0("MLE_",i),data$MLE)
  
  if(i == "H37Rv"){
    sample$sample = sample$Sample
  }
  assign(paste0("acc_",i),sum(diff(sample$sample[,1])!=0)/nrow(sample$sample))
  assign(paste0("acc_",i,"_m1"),sum(diff(sample1$sample)!=0)/length(sample1$sample))
  assign(paste0("acc_",i,"_m1a"),sum(diff(sample1a$sample[,1])!=0)/nrow(sample1a$sample))
  
  res <- sample$sample[-(1:burnin),]
  res[,1:3] <- 10^res[,1:3]
  res1 <- sample1$sample[-(1:burnin),]
  res1 <- 10^res1
  res1a <- sample1a$sample[-(1:burnin),]
  res1a <- 10^res1a
  
  assign(paste0("med_",i), apply(res,2, quantile, 0.5))
  assign(paste0("med_",i,"_m1"), quantile(res1,0.5))
  assign(paste0("med_",i,"_m1a"), apply(res1a,2, quantile, 0.5))
  
  assign(paste0("Q1_",i), apply(res,2, quantile, 0.025))
  assign(paste0("Q1_",i,"_m1"), quantile(res1,0.025))
  assign(paste0("Q1_",i,"_m1a"), apply(res1a,2, quantile, 0.025))
  
  assign(paste0("Q3_",i), apply(res,2, quantile, 0.975))
  assign(paste0("Q3_",i,"_m1"), quantile(res1,0.975))
  assign(paste0("Q3_",i,"_m1a"), apply(res1a,2, quantile, 0.975))
  
  assign(paste0("M_",i), apply(res,2, mean))
  assign(paste0("M_",i,"_m1"), mean(res1))
  assign(paste0("M_",i,"_m1a"), apply(res1a,2, mean))
  
  assign(paste0("Std_",i), apply(res,2, sd))
  assign(paste0("Std_",i,"_m1"), sd(res1))
  assign(paste0("Std_",i,"_m1a"), apply(res1a,2, sd))
  
  assign(paste0("mode_",i), apply(res,2, Modes))
  assign(paste0("mode_",i,"_m1"), Modes(res1))
  assign(paste0("mode_",i,"_m1a"), apply(res1a,2, Modes))
  
  assign(paste0("res_",i), res)
  assign(paste0("res_",i,"_m1"), res1)
  assign(paste0("res_",i,"_m1a"), res1a)
}

for(i in bacName[12]){
  cat(i,":\n")
  cat("Mutation rate:", get(paste0("p_init_",i)),"\n")
  cat("MOM:", signif(get(paste0("MOM_",i)),3),"\n")
  cat("Model 1: \n")
  cat("Mean: ",signif(get(paste0("M_",i,"_m1")),3), "\n")
  cat("Median: ",signif(get(paste0("med_",i,"_m1")),3), "\n")
  cat("Modes: ",signif(get(paste0("mode_",i,"_m1"))$modes[1],3), "\n")
  cat("Standard deviation: ", signif(get(paste0("Std_",i,"_m1")),3), "\n")
  cat("Confidence interval: (", signif(get(paste0("Q1_",i,"_m1")),3),",",
      signif(get(paste0("Q3_",i,"_m1")),3), ")", "\n")
}

for(i in bacName[12]){
  res <- get(paste0("res_",i,"_m1a"))
  res <- log10(res)
  kern.fit <- kde(res,xmin = c(log10(0.5),-11), 
                  xmax = c(log10(2),-7), 
                  gridsize = c(50,50))
  top = kern.fit$estimate[order(kern.fit$estimate,decreasing = T)[1]]
  idx = which(kern.fit$estimate == top,arr.ind = T)
  joint.mode_m1a <- c(10^kern.fit$eval.points[[1]][idx[1]],10^kern.fit$eval.points[[2]][idx[2]])
  
  assign(paste0("joint.mode_",i,"_m1a"), joint.mode_m1a)
}

for(i in bacName[12]){
  cat(i,":\n")
  
  cat("Model 1a: \n")
  cat("Mean: ",signif(get(paste0("M_",i,"_m1a")),3), "\n")
  cat("Median: ",signif(get(paste0("med_",i,"_m1a")),3), "\n")
  cat("Modes of delta: ",signif(get(paste0("mode_",i,"_m1a"))[[1]]$modes,3), "\n")
  cat("Modes of p: ",signif(get(paste0("mode_",i,"_m1a"))[[2]]$modes,3), "\n")
  cat("Joint mode:", signif(get(paste0("joint.mode_",i,"_m1a")),3), "\n")
  cat("Standard deviation: ", signif(get(paste0("Std_",i,"_m1a")),3), "\n")
  cat("Confidence interval of delta: (", signif(get(paste0("Q1_",i,"_m1a"))[1],3),",",
      signif(get(paste0("Q3_",i,"_m1a"))[1],3), ")", "\n")
  cat("Confidence interval of p: (", signif(get(paste0("Q1_",i,"_m1a"))[2],3),",",
      signif(get(paste0("Q3_",i,"_m1a"))[2],3), ")", "\n")
}
for(i in bacName[12]){
  res <- get(paste0("res_",i))
  res[,1:3] <- log10(res[,1:3])
  kern.fit <- kde(res,xmin = c(log10(0.5),-11,-11, 4), 
                  xmax = c(log10(2),-7,-7, 18), 
                  gridsize = c(4,15,15,7))
  top = kern.fit$estimate[order(kern.fit$estimate,decreasing = T)[1]]
  idx = which(kern.fit$estimate == top,arr.ind = T)
  joint.mode <- c(10^kern.fit$eval.points[[1]][idx[1]],10^kern.fit$eval.points[[2]][idx[2]],
                  10^kern.fit$eval.points[[3]][idx[3]],kern.fit$eval.points[[4]][idx[4]])
  
  assign(paste0("joint.mode_",i), joint.mode)
}

for(i in bacName[12]){
  cat(i,":\n")
  
  cat("Two-stage model: \n")
  cat("Mean: ",signif(get(paste0("M_",i)),3), "\n")
  cat("Median: ",signif(get(paste0("med_",i)),3), "\n")
  cat("Modes of delta: ",signif(get(paste0("mode_",i))[[1]]$modes,3), "\n")
  cat("Modes of p1: ",signif(get(paste0("mode_",i))[[2]]$modes,3), "\n")
  cat("Modes of p2: ",signif(get(paste0("mode_",i))[[3]]$modes,3), "\n")
  cat("Modes of tau: ",signif(get(paste0("mode_",i))[[4]]$modes,3), "\n")
  #cat("Joint mode:", signif(get(paste0("joint.mode_",i)),3), "\n")
  cat("Standard deviation: ", signif(get(paste0("Std_",i)),3), "\n")
  cat("Confidence interval of delta: (", signif(get(paste0("Q1_",i))[1],3),",",
      signif(get(paste0("Q3_",i))[1],3), ")", "\n")
  cat("Confidence interval of p1: (", signif(get(paste0("Q1_",i))[2],3),",",
      signif(get(paste0("Q3_",i))[2],3),")", "\n")
  cat("Confidence interval of p2: (", signif(get(paste0("Q1_",i))[3],3),",",
      signif(get(paste0("Q3_",i))[3],3),")", "\n")
  cat("Confidence interval of tau: (", signif(get(paste0("Q1_",i))[4],3),",",
      signif(get(paste0("Q3_",i))[4],3),")", "\n")
}


sample <- readMat("sample_post_H37Rv.mat")
varNames <- c("Delta","P1","P2","Tau")
par(mfrow = c(2,2))
for(i in 1:4){
  plot(sample$Sample[,i], pch = 20, type = "l", xlab = "Steps", ylab = varNames[i])
}

burnin = 2000
sum(diff(sample$Sample[(burnin + 1):10000,1]) !=0)/(10000 - burnin)
res <- sample$Sample[(burnin + 1):10000,]
res <- data.frame(res)
res[,1] <- 10^res[,1]
post.mean = colMeans(res)
post.mean
apply(res,2,sd)
apply(res,2,quantile, c(0.025, 0.5, 0.975))
names(res) <- c("delta","p1","p2","tau")
head(res)
apply(res,2,quantile, c(0.025,0.975))
apply(res,2,Modes)

p_2d_delta_p1 = ggplot(res, aes(x=delta, y=p1) ) +
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_distiller(palette=4, direction=-1) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  geom_point(aes(x= delta, y = p1), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
  geom_point(x = 0.8, y = -9, col = "red", shape = "cross")+
  theme(
    legend.position='none'
  )+ 
  xlab(expression(delta))+
  ylab(expression(log[10](P[1])))

p_2d_p1_p2 = ggplot(res, aes(x=p1, y=p2) ) +
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_distiller(palette=4, direction=-1) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  geom_point(aes(x= p1, y = p2), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
  geom_point(y = -8, x = -9, col = "red", shape = "cross")+
  theme(
    legend.position='none'
  )+ 
  xlab(expression(log[10](P[1])))+
  ylab(expression(log[10](P[2])))

p_2d_p2_tau = ggplot(res, aes(x=p2, y=tau) ) +
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_distiller(palette=4, direction=-1) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  geom_point(aes(x= p2, y = tau), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
  geom_point(x = -8, y = 14, col = "red", shape = "cross")+
  theme(
    legend.position='none'
  )+ 
  xlab(expression(log[10](P[2])))+
  ylab(expression(tau))

p_2d_delta_tau = ggplot(res, aes(x=delta, y=tau) ) +
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_distiller(palette=4, direction=-1) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  geom_point(aes(x= delta, y = tau), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
  geom_point(x = 0.8, y = 14, col = "red", shape = "cross")+
  theme(
    legend.position='none'
  )+ 
  xlab(expression(delta))+
  ylab(expression(tau))

p_2d_p1_tau = ggplot(res, aes(x=p1, y=tau) ) +
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_distiller(palette=4, direction=-1) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  geom_point(aes(x= p1, y = tau), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
  geom_point(x = -9, y = 14, col = "red", shape = "cross")+
  theme(
    legend.position='none'
  )+ 
  xlab(expression(log[10](P[1])))+
  ylab(expression(tau))

p_2d_delta_p2 = ggplot(res, aes(x=delta, y=p2) ) +
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_distiller(palette=4, direction=-1) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  geom_point(aes(x= delta, y = p2), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
  geom_point(x = 0.8, y = -8, col = "red", shape = "cross")+
  theme(
    legend.position='none'
  )+ 
  xlab(expression(delta))+
  ylab(expression(log[10](P[2])))

p_delta <- ggplot(res, aes(x = delta)) +
  geom_histogram(aes(y = stat(density)), bins = 15)+
  geom_vline(xintercept = 0.8, col = "red") +
  theme_classic()+xlab(NULL)
p_delta_v <- ggplot(res, aes(y = delta)) +
  geom_histogram(aes(x = stat(density)), bins = 15)+
  geom_hline(yintercept = 0.8, col = "red") +
  theme_classic()+xlab(NULL)
p_p1 <- ggplot(res, aes(x = p1)) +
  geom_histogram(aes(y = stat(density)), bins = 15)+
  geom_vline(xintercept = -9, col = "red") +
  theme_classic()+ylab(NULL)
p_p1_v <- ggplot(res, aes(y = p1)) +
  geom_histogram(aes(x = stat(density)), bins = 15)+
  geom_hline(yintercept = -9, col = "red") +
  theme_classic()+ylab(NULL)
p_p2 <- ggplot(res, aes(x = p2)) +
  geom_histogram(aes(y = stat(density)), bins = 15)+
  geom_vline(xintercept = -8, col = "red") +
  theme_classic()+ylab(NULL)
p_p2_v <- ggplot(res, aes(y = p2)) +
  geom_histogram(aes(x = stat(density)), bins = 15)+
  geom_hline(yintercept = -8, col = "red") +
  theme_classic()+ylab(NULL)
p_tau <- ggplot(res, aes(x = tau)) +
  geom_histogram(aes(y = stat(density)), bins = 15)+
  geom_vline(xintercept = 14, col = "red") +
  theme_classic()+ylab(NULL)
p_tau_v <- ggplot(res, aes(y = tau)) +
  geom_histogram(aes(x = stat(density)), bins = 15)+
  geom_hline(yintercept = 14, col = "red") +
  theme_classic()+ylab(NULL)


ggarrange(p_delta,NULL,p_2d_delta_p2, p_p2_v, ncol=2,nrow = 2,widths = c(3,1),heights= c(1,3),
          align = "hv")

ggarrange(p_p1,NULL,p_2d_p1_p2, p_p2_v, ncol=2,nrow = 2,widths = c(3,1),heights= c(1,3),
          align = "hv")
ggarrange(p_p2,NULL,p_2d_p2_tau, p_tau_v, ncol=2,nrow = 2,widths = c(3,1),heights= c(1,3),
          align = "hv")
ggarrange(p_delta,NULL,p_2d_delta_p1, p_p1_v, ncol=2,nrow = 2,widths = c(3,1),heights= c(1,3),
          align = "hv")
ggarrange(p_p1,NULL,p_2d_p1_tau, p_tau_v, ncol=2,nrow = 2,widths = c(3,1),heights= c(1,3),
          align = "hv")
ggarrange(p_delta,NULL,p_2d_delta_tau, p_tau_v, ncol=2,nrow = 2,widths = c(3,1),heights= c(1,3),
          align = "hv")
## -------------- Find the MAPs for the joint dist.--------------------
library(MASS)
library(ks)
kern.fit <- kde(res,
                xmin = c(log10(0.5),-11,-11, 4), xmax = c(log10(2),-7,-7, 18), gridsize = c(50,50,50,50))
#plot(kern.fit, drawpoints = T)
top = kern.fit$estimate[order(kern.fit$estimate,decreasing = T)[1]]
top
idx = which(kern.fit$estimate == top,arr.ind = T)
kern.fit$eval.points[[1]][idx[1]]
kern.fit$eval.points[[2]][idx[2]]
kern.fit$eval.points[[3]][idx[3]]
top5 = kern.fit$estimate[order(kern.fit$estimate,decreasing = T)[1:100]]
idx = matrix(NA,100,3)
for(i in 1:100){
  idx[i,] <- which(kern.fit$estimate==top5[i], arr.ind = T)
}

kern.pred <- predict(kern.fit, x = c(p1_high[1],p2_high[1],tau_high[1]))

#kern.fit$estimate[idx[1],idx[2],idx[3]]
p1_high <- kern.fit$eval.points[[1]][idx[,1]]
p2_high <- kern.fit$eval.points[[2]][idx[,2]]
tau_high <- kern.fit$eval.points[[3]][idx[,3]]
rank = 100:1

##### ------------------ Real Data -----------------------------
p_2d_delta_p1 = ggplot(res, aes(x=delta, y=p1) ) +
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_distiller(palette=4, direction=-1) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  geom_point(aes(x= delta, y = p1), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
  geom_point(x = post.mean[1], y = post.mean[2], col = "red", shape = "cross")+
  theme(
    legend.position='none'
  )+ 
  xlab(expression(delta))+
  ylab(expression(log[10](P[1])))

p_2d_p1_p2 = ggplot(res, aes(x=p1, y=p2) ) +
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_distiller(palette=4, direction=-1) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  geom_point(aes(x= p1, y = p2), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
  geom_point(x = post.mean[2], y = post.mean[3], col = "red", shape = "cross")+
  theme(
    legend.position='none'
  )+ 
  xlab(expression(log[10](P[1])))+
  ylab(expression(log[10](P[2])))

p_2d_p2_tau = ggplot(res, aes(x=p2, y=tau) ) +
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_distiller(palette=4, direction=-1) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  geom_point(aes(x= p2, y = tau), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
  geom_point(x = post.mean[3], y = post.mean[4], col = "red", shape = "cross")+
  theme(
    legend.position='none'
  )+ 
  xlab(expression(log[10](P[2])))+
  ylab(expression(tau))

p_2d_delta_tau = ggplot(res, aes(x=delta, y=tau) ) +
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_distiller(palette=4, direction=-1) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  geom_point(aes(x= delta, y = tau), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
  geom_point(x = post.mean[1], y = post.mean[4], col = "red", shape = "cross")+
  theme(
    legend.position='none'
  )+ 
  xlab(expression(delta))+
  ylab(expression(tau))

p_2d_p1_tau = ggplot(res, aes(x=p1, y=tau) ) +
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_distiller(palette=4, direction=-1) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  geom_point(aes(x= p1, y = tau), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
  geom_point(x = post.mean[2], y = post.mean[4], col = "red", shape = "cross")+
  theme(
    legend.position='none'
  )+ 
  xlab(expression(log[10](P[1])))+
  ylab(expression(tau))

p_2d_delta_p2 = ggplot(res, aes(x=delta, y=p2) ) +
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_distiller(palette=4, direction=-1) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  geom_point(aes(x= delta, y = p2), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
  geom_point(x = post.mean[1], y = post.mean[3], col = "red", shape = "cross")+
  theme(
    legend.position='none'
  )+ 
  xlab(expression(delta))+
  ylab(expression(log[10](P[2])))

p_delta <- ggplot(res, aes(x = delta)) +
  geom_histogram(aes(y = stat(density)), bins = 15)+
  geom_vline(xintercept = post.mean[1], col = "red") +
  theme_classic()+xlab(NULL)
p_delta_v <- ggplot(res, aes(y = delta)) +
  geom_histogram(aes(x = stat(density)), bins = 15)+
  geom_hline(yintercept = post.mean[1], col = "red") +
  theme_classic()+xlab(NULL)
p_p1 <- ggplot(res, aes(x = p1)) +
  geom_histogram(aes(y = stat(density)), bins = 15)+
  geom_vline(xintercept = post.mean[2], col = "red") +
  theme_classic()+ylab(NULL)
p_p1_v <- ggplot(res, aes(y = p1)) +
  geom_histogram(aes(x = stat(density)), bins = 15)+
  geom_hline(yintercept = post.mean[2], col = "red") +
  theme_classic()+ylab(NULL)
p_p2 <- ggplot(res, aes(x = p2)) +
  geom_histogram(aes(y = stat(density)), bins = 15)+
  geom_vline(xintercept = post.mean[3], col = "red") +
  theme_classic()+ylab(NULL)
p_p2_v <- ggplot(res, aes(y = p2)) +
  geom_histogram(aes(x = stat(density)), bins = 15)+
  geom_hline(yintercept = post.mean[3], col = "red") +
  theme_classic()+ylab(NULL)
p_tau <- ggplot(res, aes(x = tau)) +
  geom_histogram(aes(y = stat(density)), bins = 15)+
  geom_vline(xintercept = post.mean[4], col = "red") +
  theme_classic()+ylab(NULL)
p_tau_v <- ggplot(res, aes(y = tau)) +
  geom_histogram(aes(x = stat(density)), bins = 15)+
  geom_hline(yintercept = post.mean[4], col = "red") +
  theme_classic()+ylab(NULL)


ggarrange(p_delta,NULL,p_2d_delta_p2, p_p2_v, ncol=2,nrow = 2,widths = c(3,1),heights= c(1,3),
          align = "hv")

ggarrange(p_p1,NULL,p_2d_p1_p2, p_p2_v, ncol=2,nrow = 2,widths = c(3,1),heights= c(1,3),
          align = "hv")
ggarrange(p_p2,NULL,p_2d_p2_tau, p_tau_v, ncol=2,nrow = 2,widths = c(3,1),heights= c(1,3),
          align = "hv")
ggarrange(p_delta,NULL,p_2d_delta_p1, p_p1_v, ncol=2,nrow = 2,widths = c(3,1),heights= c(1,3),
          align = "hv")
ggarrange(p_p1,NULL,p_2d_p1_tau, p_tau_v, ncol=2,nrow = 2,widths = c(3,1),heights= c(1,3),
          align = "hv")
ggarrange(p_delta,NULL,p_2d_delta_tau, p_tau_v, ncol=2,nrow = 2,widths = c(3,1),heights= c(1,3),
          align = "hv")