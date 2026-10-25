library(ggplot2)
library(ggpubr)
library(LaplacesDemon)
library(R.matlab)
burnin = 5000
N = 100
comp_idx = c()

for(i in 1:100){
  sample <- try(readMat(paste0("../Result/simulation/simu2/sample1_",i,".mat")))
  if(inherits(sample, "try-error"))
  {
    #error handling code, maybe just skip this iteration using
    next
  }
  comp_idx = c(comp_idx,i)
}
N = length(comp_idx)
M <- Std <- med <- Q1 <- Q3 <- matrix(NA,N,4)
runningTime <- rep(NA,N)
mode_delta <- mode_p1 <- mode_p2 <- mode_tau <- list()
#rest of iteration for case of no error
j = 1
for(i in comp_idx){
  sample <- readMat(paste0("../Result/simulation/simu2/sample2_",i,".mat"))
  res <- sample$sample[-(1:burnin),]
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
  j = j+1
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

colMeans(Q1)
colMeans(Q3)
# sum(0.8 >= Q1[,1] & 0.8 <= Q3[,1]) / N
# sum(1e-9 >= Q1[,2] & 1e-9 <= Q3[,2]) / N
# sum(5e-8 >= Q1[,3] & 5e-8 <= Q3[,3]) / N
# sum(14 >= Q1[,4] & 14 <= Q3[,4]) / N
colMeans(Q3- Q1)
sd(runningTime)/3600

##### --------------------- Plots ------------------------####
library(R.matlab)
library(ggpubr)
library(ggplot2)
library(cowplot)
burnin = 5000
case = 2
if(case == 1){
  truth = c(0.8,-9,-8,14)
}else{
  truth = c(0.8,-9,log10(5e-8),14)
}

seed = 99
for(seed in 4:4){
  sample <- try(readMat(paste0("../Result/simulation/simu2/sample",case,"_",seed,".mat")))
  # if(inherits(sample, "try-error"))
  # {
  #   #error handling code, maybe just skip this iteration using
  #   next
  # }
  res <- sample$sample[-(1:burnin),]
  res <- data.frame(res)
  res[,1] <- 10^res[,1]
  names(res) <- c("delta","p1","p2","tau")
  jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
  p_2d_p2_p1 = ggplot(res, aes(x=p1, y=p2) ) +
    stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
    scale_fill_gradientn(colours = jet.colors(7)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    #geom_point(aes(x= p1, y = p2), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
    geom_point(y = truth[3], x = truth[2], col = "red", shape = "cross")+
    theme(
      legend.position='right',
      panel.border = element_blank()
    )+ 
    ggtitle(expression(paste("(A) Joint posterior of ", log[10](p[1]), " and ", log[10](p[2]))))+
    xlab(expression(log[10](P[1])))+
    ylab(expression(log[10](P[2])))
  
  p_2d_tau_p1 = ggplot(res, aes(x=p1, y=tau) ) +
    stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
    scale_fill_gradientn(colours = jet.colors(7)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    #geom_point(aes(x= p1, y = p2), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
    geom_point(y = truth[4], x = truth[2], col = "red", shape = "cross")+
    theme(
      legend.position='right'
    )+ 
    ggtitle(expression(paste("(B) Joint posterior of ", log[10](p[1]), " and ", tau)))+
    xlab(expression(log[10](P[1])))+
    ylab(expression(tau))
  
  p_2d_p2_tau = ggplot(res, aes(x=tau, y=p2) ) +
    stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
    scale_fill_gradientn(colours = jet.colors(7)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    #geom_point(aes(x= p1, y = p2), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
    geom_point(y = truth[3], x = truth[4], col = "red", shape = "cross")+
    theme(
      legend.position='right'
    )+ 
    ggtitle(expression(paste("(C) Joint posterior of ", tau, " and ", log[10](p[2]))))+
    xlab(expression(tau))+
    ylab(expression(log[10](P[2])))
  
  # Marginal histograms
  p_p1 <- ggplot(res, aes(x = p1)) +
    geom_histogram(aes(y = stat(density)), bins = 15,
                   color = "black", fill = "lightblue")+
    geom_density(lwd = 1.2, colour = 2)+
    geom_vline(xintercept = truth[2], col = "red", size = 1) +
    scale_y_reverse(expand = c(0,0))+
    scale_x_continuous(position = "top")+
    theme(panel.grid = element_blank(),
          axis.line.x = element_line(size = 1),
          panel.border = element_blank(),
          panel.background = element_blank(),
          axis.ticks.y=element_blank(),
          axis.text.y = element_blank())+
    xlab(NULL)+
    ylab(NULL)
  p_p2_v <- ggplot(res, aes(y = p2)) +
    geom_histogram(aes(x = stat(density)), bins = 15,
                   color = "black", fill = "lightblue")+
    geom_density(lwd = 1.2, colour = 2)+
    geom_hline(yintercept = truth[3], col = "red", size = 1) +
    scale_x_reverse(expand = c(0,0))+
    scale_y_continuous(position = "right")+
    theme(panel.grid = element_blank(),
          axis.line.y = element_line(size = 1),
          panel.border = element_blank(),
          panel.background = element_blank(),
          axis.ticks.x=element_blank(),
          axis.text.x = element_blank())+
    xlab(NULL) + 
    ylab(NULL)
  p_tau <- ggplot(res, aes(x = tau)) +
    geom_histogram(aes(y = stat(density)), bins = 15,
                   color = "black", fill = "lightblue")+
    geom_density(lwd = 1.2, colour = 2)+
    geom_vline(xintercept = truth[4], col = "red", size = 1) +
    scale_y_reverse(expand = c(0,0))+
    scale_x_continuous(position = "top")+
    theme(panel.grid = element_blank(),
          axis.line.x = element_line(size = 1),
          panel.border = element_blank(),
          panel.background = element_blank(),
          axis.ticks.y=element_blank(),
          axis.text.y = element_blank())+
    xlab(NULL)+
    ylab(NULL)
  p_tau_v <- ggplot(res, aes(y = tau)) +
    geom_histogram(aes(x = stat(density)), bins = 15,
                   color = "black", fill = "lightblue")+
    geom_density(lwd = 1.2, colour = 2)+
    geom_hline(yintercept = truth[4], col = "red", size = 1) +
    scale_x_reverse(expand = c(0,0))+
    scale_y_continuous(position = "right")+
    theme(panel.grid = element_blank(),
          axis.line.y = element_line(size = 1),
          panel.border = element_blank(),
          panel.background = element_blank(),
          axis.ticks.x=element_blank(),
          axis.text.x = element_blank())+
    xlab(NULL) + 
    ylab(NULL)
  png(paste0("../Result/simulation/simu2/sample",case,"_",seed,"_p2_p1.png"))
  ggarrange(p_p2_v,p_2d_p2_p1,NULL,p_p1,nrow = 2,ncol = 2,heights = c(3,1), widths = c(1,3),
            align = "hv")
  dev.off()
  png(paste0("../Result/simulation/simu2/sample",case,"_",seed,"_tau_p1.png"))
  ggarrange(p_tau_v,p_2d_tau_p1,NULL,p_p1,nrow = 2,ncol = 2,heights = c(3,1), widths = c(1,3),
            align = "hv")
  dev.off()
  png(paste0("../Result/simulation/simu2/sample",case,"_",seed,"_p2_tau.png"))
  ggarrange(p_p2_v,p_2d_p2_tau,NULL,p_tau,nrow = 2,ncol = 2,heights = c(3,1), widths = c(1,3),
            align = "hv")
  dev.off()
  
  
  
  ## The set of plots including delta
  p_2d_delta_p1 = ggplot(res, aes(x=p1, y=delta) ) +
    stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
    scale_fill_gradientn(colours = jet.colors(7)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    #geom_point(aes(x= p1, y = p2), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
    geom_point(y = truth[1], x = truth[2], col = "red", shape = "cross")+
    theme(
      legend.position='right',
      panel.border = element_blank()
    )+ 
    ggtitle(expression(paste("(A) Joint posterior of ", log[10](p[1]), " and ", delta)))+
    xlab(expression(log[10](P[1])))+
    ylab(expression(delta))
  
  p_2d_delta_tau = ggplot(res, aes(x=tau, y=delta) ) +
    stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
    scale_fill_gradientn(colours = jet.colors(7)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    #geom_point(aes(x= p1, y = p2), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
    geom_point(y = truth[1], x = truth[4], col = "red", shape = "cross")+
    theme(
      legend.position='right'
    )+ 
    ggtitle(expression(paste("(B) Joint posterior of ", tau, " and ", delta)))+
    xlab(expression(tau))+
    ylab(expression(delta))
  
  p_2d_delta_p2 = ggplot(res, aes(y=delta, x=p2) ) +
    stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
    scale_fill_gradientn(colours = jet.colors(7)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    #geom_point(aes(x= p1, y = p2), col = "green", size = 0.8, shape = 1, alpha = 0.5)+
    geom_point(y = truth[1], x = truth[3], col = "red", shape = "cross")+
    theme(
      legend.position='right'
    )+ 
    ggtitle(expression(paste("(C) Joint posterior of ", log[10](p[2]), " and ", delta)))+
    ylab(expression(delta))+
    xlab(expression(log[10](P[2])))
  
  # Marginal histograms
  p_p2 <- ggplot(res, aes(x = p2)) +
    geom_histogram(aes(y = stat(density)), bins = 15,
                   color = "black", fill = "lightblue")+
    geom_density(lwd = 1.2, colour = 2)+
    geom_vline(xintercept = truth[3], col = "red", size = 1) +
    scale_y_reverse(expand = c(0,0))+
    scale_x_continuous(position = "top")+
    theme(panel.grid = element_blank(),
          axis.line.x = element_line(size = 1),
          panel.border = element_blank(),
          panel.background = element_blank(),
          axis.ticks.y=element_blank(),
          axis.text.y = element_blank())+
    xlab(NULL) + 
    ylab(NULL)
  p_delta_v <- ggplot(res, aes(y = delta)) +
    geom_histogram(aes(x = stat(density)), bins = 15,
                   color = "black", fill = "lightblue")+
    geom_density(lwd = 1.2, colour = 2)+
    geom_hline(yintercept = truth[1], col = "red", size = 1) +
    scale_x_reverse(expand = c(0,0))+
    scale_y_continuous(position = "right")+
    theme(panel.grid = element_blank(),
          axis.line.y = element_line(size = 1),
          panel.border = element_blank(),
          panel.background = element_blank(),
          axis.ticks.x=element_blank(),
          axis.text.x = element_blank())+
    xlab(NULL) + 
    ylab(NULL)
  png(paste0("../Result/simulation/simu2/sample",case,"_",seed,"_delta_p1.png"))
  ggarrange(p_delta_v,p_2d_delta_p1,NULL,p_p1,nrow = 2,ncol = 2,heights = c(3,1), widths = c(1,3),
            align = "hv")
  dev.off()
  png(paste0("../Result/simulation/simu2/sample",case,"_",seed,"_delta_p2.png"))
  ggarrange(p_delta_v,p_2d_delta_p2,NULL,p_p2,nrow = 2,ncol = 2,heights = c(3,1), widths = c(1,3),
            align = "hv")
  dev.off()
  png(paste0("../Result/simulation/simu2/sample",case,"_",seed,"_delta_tau.png"))
  ggarrange(p_delta_v,p_2d_delta_tau,NULL,p_tau,nrow = 2,ncol = 2,heights = c(3,1), widths = c(1,3),
            align = "hv")
  dev.off()
}



