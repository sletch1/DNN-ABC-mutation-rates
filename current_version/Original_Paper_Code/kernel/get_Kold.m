function [Kern1] = get_Kold(kern_param, theta_list, init_feat_s,sig2_n)

% This function calculates the inverse of K_old_old, which is the kernel matrix of
% theta_list
% Input:
% kern_param: a 1 by 2 vector. Hyperparameters for the kernel function. See
%            se_kern_fast for definition. 
% theta_list: the initial grid of theta on which to sample initial echoes.
% ini_feature: the intial feature extracted. 
% feat_idx: if an integer, only consider prediction for that index of the
%           feature vector. if empty [], we will do all features one by
%           one (assuming features are independent of each other).
% Output:
% Kern: the structure of storing the kernel values: K_old_old and the
% inversed term used in GP perdiction
% Reference:
% Chapter 2: Rasmussen and Williams Gaussian Processes for Machine Learning


J = size(init_feat_s,2);     

% step 1: get y.(refer equation 2.21, Rasmussen and Williams 2006) 
% step 2: calculate kernel, K(TH, TH), K(TH, th), K(th, th)
% here I assume th is a scalar, on which I want to predict f*. 
% OK, I think I will treat th as a vector, that will save me a lot of
% computation time. 
% for caes 1: 
sig2_n= min(1e-4, var(init_feat_s)*0.005); 

N = size(theta_list,1);

K_old_old = se_kern_fast(kern_param, theta_list);

% Mu_cond = NaN(1, length(feat_idx));
% Var_cond = NaN(1, length(feat_idx));

K_iv = NaN(N,N,J);

for i = 1:J
    K_Uoo = jitterChol(K_old_old + sig2_n(i)*eye(N));
    K_iv(:,:,i) = solve_triu(K_Uoo, eye(N));
end

Kern1.K_iv = K_iv;
Kern1.old_old = K_old_old;
 