function[tau,theta,sig_2n] = get_kern_param(init_param, theta_list, init_feat_s)

% This function calculates the optimal hyperparameters:tau,theta and sigma
% square
% Input:
% 
% theta_list: the initial grid of theta on which to sample initial echoes.
% ini_feature_s: the intial feature extracted. 
%
% Output:
% tau: the parameter in front of the kernel function
% theta: scale parameter
% sig_2n: the nugget effect
% Reference:
% Chapter 2: Rasmussen and Williams Gaussian Processes for Machine Learning

% step 1: get y.(refer equation 2.21, Rasmussen and Williams 2006) 
% step 2: calculate kernel, K(TH, TH), K(TH, th), K(th, th)
% here I assume th is a scalar, on which I want to predict f*. 
% OK, I think I will treat th as a vector, that will save me a lot of
% computation time. 
% for caes 1: sig2_n= min(1e-4, var(init_feat_s)*0.005); 
f = @(x)loglikelihood(x, theta_list,init_feat_s);
[x,~] = fminunc(f,init_param);
theta= x(1);
sig_2n = x(2);

 N = size(theta_list,1);
 K_old_old = se_kern_fast([1 theta], theta_list);
 K = K_old_old + sig_2n*eye(N);
 K_Uoo = jitterChol(K);
 K_iv = solve_triu(K_Uoo, eye(N));
 tau = sqrt((init_feat_s'*K_iv)*(K_iv'*init_feat_s)/N);

