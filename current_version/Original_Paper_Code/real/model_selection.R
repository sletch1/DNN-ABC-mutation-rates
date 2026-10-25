bacName = c("H37Rv", "E865", "E729", "E740", "E1221", "E1449", "Harlingen",
            "E26_95", "E80", "E55", "E26_94", "E3942", "E47")
library(R.matlab)
library(flexclust)
i = 7 
nmc = 25
res1 <- matrix(NA,2,13)
res1a <- res <- matrix(NA,2,13)
p_res <- matrix(NA,6,13)
for(i in 1:13){
  for(metric in c("mean","median")){
    data <- readMat(paste0('../Result/werngren_v1/',bacName[i],'.mat'))
    simu = readMat(paste0('../Result/model selection/simu_nmc',nmc,'_',metric,'_',
                          bacName[i],'.mat'))
    simu1 = readMat(paste0('../Result/model selection/simu_nmc',nmc,'_',metric,'_',
                           bacName[i],'_m1.mat'))
    simu1a <- readMat(paste0('../Result/model selection/simu_nmc',nmc,'_',metric,'_',
                             bacName[i],'_m1a.mat'))
    
    #
    X_t = data$X.t; N_t = rep(data$N.t,length(X_t))
    df = rbind(cbind(X_t,N_t), cbind(simu$Xt, simu$Nt))
    #df.scaled = scale(df)
    #dist_mat = dist2(df.scaled[1:length(X_t),], df.scaled[-(1:length(X_t)),])
    #max(dist_mat)
    #min(dist_mat)
    #mean(dist_mat)
    #colMeans(df[1:length(X_t),])
    #colMeans(df[-(1:length(X_t)),])
    #sum((colMeans(df.scaled[1:length(X_t),]) - colMeans(df.scaled[-(1:length(X_t)),]))^2)
    #df = rbind(cbind(X_t,N_t), cbind(simu1$Xt, simu1$Nt))
    #df.scaled = scale(df)
    #dist_mat1 = dist2(df.scaled[1:length(X_t),], df.scaled[-(1:length(X_t)),])
    #max(dist_mat1)
    #min(dist_mat1)
    #mean(dist_mat1)
    
    #df = rbind(cbind(X_t,N_t), cbind(simu1a$Xt, simu1a$Nt))
    #df.scaled = scale(df)
    #dist_mat1a = dist2(df.scaled[1:length(X_t),], df.scaled[-(1:length(X_t)),])
    #max(dist_mat1a)
    #min(dist_mat1a)
    #mean(dist_mat1a)
    
    
    #ratio_scaled = scale(c(X_t/N_t, simu$Xt/simu$Nt))
    #dist_ratio_scaled <- dist2(ratio_scaled[1:length(X_t)],ratio_scaled[-(1:length(X_t))])
    #ratio_scaled = scale(c(X_t/N_t, simu1$Xt/simu1$Nt))
    #dist_ratio_scaled_1 <- dist2(ratio_scaled[1:length(X_t)],ratio_scaled[-(1:length(X_t))])
    #ratio_scaled = scale(c(X_t/N_t, simu1a$Xt/simu1a$Nt))
    #dist_ratio_scaled_1a <- dist2(ratio_scaled[1:length(X_t)],ratio_scaled[-(1:length(X_t))])
    #min(dist_ratio_scaled)
    #max(dist_ratio_scaled)
    #mean(dist_ratio_scaled)
    #(mean(ratio_scaled[1:length(X_t)]) - mean(ratio_scaled[-(1:length(X_t))]))^2
    
    ratio_data <- X_t/N_t
    ratio_simu <- simu$Xt/simu$Nt
    ratio_simu1 <- simu1$Xt/simu1$Nt
    ratio_simu1a <- simu1a$Xt/simu1a$Nt
    
    dist_ratio <- dist2(ratio_data, ratio_simu)
    dist_ratio1 <- dist2(ratio_data, ratio_simu1)
    dist_ratio1a <- dist2(ratio_data, ratio_simu1a)
    #t.test(dist_ratio, dist_ratio1a)
    #t.test(dist_ratio1a, dist_ratio1)
    
    
    #assign(paste0("mat_",metric,"_1"),mean(dist_mat1))
    #assign(paste0("mat_",metric,"_1a"),mean(dist_mat1a))
    #assign(paste0("mat_",metric),mean(dist_mat))
    
    #assign(paste0("p_mat_",metric,"_1a"),wilcox.test(dist_mat1a,dist_mat1)$p.value)
    #assign(paste0("p_mat_",metric),wilcox.test(dist_mat,dist_mat1a)$p.value)
    
    assign(paste0("ratio_",metric,"_1"),median(dist_ratio1))
    assign(paste0("ratio_",metric,"_1a"),median(dist_ratio1a))
    assign(paste0("ratio_",metric),median(dist_ratio))
    
    assign(paste0("p_ratio_",metric,"_1a_1"),wilcox.test(dist_ratio1a,dist_ratio1,alternative = "less")$p.value)
    assign(paste0("p_ratio_",metric,"_2_1"),wilcox.test(dist_ratio,dist_ratio1, alternative = "less")$p.value)
    assign(paste0("p_ratio_",metric,"_2_1a"),wilcox.test(dist_ratio,dist_ratio1a, alternative = "less")$p.value)
    
    #assign(paste0("ratios_",metric,"_1"),mean(dist_ratio_scaled_1))
    #assign(paste0("ratios_",metric,"_1a"),mean(dist_ratio_scaled_1a))
    #assign(paste0("ratios_",metric),mean(dist_ratio_scaled))
    #assign(paste0("p_ratios_",metric,"_1a"),wilcox.test(dist_ratio_scaled_1a,dist_ratio_scaled_1)$p.value)
    #assign(paste0("p_ratios_",metric),wilcox.test(dist_ratio_scaled,dist_ratio_scaled_1a)$p.value)
  }
  # res1[,i] <- signif(c(mat_mean_1,mat_median_1,ratio_mean_1,
  #                        ratio_median_1,ratios_mean_1,ratios_median_1),3)
  # res1a[,i] <- signif(c(mat_mean_1a,p_mat_mean_1a,mat_median_1a,p_mat_median_1a,
  #                       ratio_mean_1a, p_ratio_mean_1a, ratio_median_1a, p_ratio_median_1a,
  #                       ratios_mean_1a, p_ratios_mean_1a, ratios_median_1a, p_ratios_median_1a),3)
  # res[,i] <- signif(c(mat_mean,p_mat_mean,mat_median,p_mat_median,
  #                     ratio_mean, p_ratio_mean, ratio_median, p_ratio_median,
  #                     ratios_mean, p_ratios_mean, ratios_median, p_ratios_median),3)
  res1[,i] <- signif(c(ratio_mean_1,ratio_median_1),3)
  res1a[,i] <- signif(c(ratio_mean_1a, ratio_median_1a),3)
  res[,i] <- signif(c(ratio_mean, ratio_median),3)
  p_res[,i] <- signif(c(p_ratio_mean_1a_1, p_ratio_mean_2_1a,p_ratio_mean_2_1,
                        p_ratio_median_1a_1, p_ratio_median_2_1a, p_ratio_median_2_1),3)
}

write.csv(rbind(res1[1,],res1a[1,],res[1,],
                  p_res[1:3,]), 
          file = "../Result/model selection/model_selection_meanRatio_nmc25_medianDist.csv",
          row.names =F)
write.csv(rbind(res1[2,],res1a[2,],res[2,],
                  p_res[4:6,]), 
          file = "../Result/model selection/model_selection_medianRatio_nmc25_medianDist.csv",
          row.names =F)

#write.csv(res1a, file = "../Result/model selection/model_selection_M1a_nmc25.csv",row.names = F)
#write.csv(res, file = "../Result/model selection/modle_selection_nmc25.csv",row.names = F)



