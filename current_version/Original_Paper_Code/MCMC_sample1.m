%% This function is for generating samples for ABC-MCMC algorithm
% Input: theta_mu: prior for theta, log scale
%        theta_sigma: prior variance for theta, 0 if parameter is not
%                     estimated in this case.
%        param_range; a structure storing range of each parameter
%        obs_X: observed data
%        model_spec: model specifications, including: 
  %                     num_training_theta: vector of 2, the number of training data points 
%                                         for initial training and additional trainings.
%                     eps: accuracy threshold, 
%                     psi: error threshold
%                     N: sample size
%                     chkt: time checking point
%                     a: parameter for birth-death process
%                     num_rep: number of replicates at each training point
%                     time_update: for showing progress
%                     init_param: initial values of GP's hyperparameters.
%                     theta_old: the initial values of parameters to be estimated.
%                     bounds: bounds for the parameters to be estimated
%                     trans_step: stepwidth for proposal distribution
function [Y, m, s] = MCMC_sample1(a, a_dt, theta, p_range,chkt, num_rep, Z0)

    Y = NaN(num_rep,1);
    for i = 1:num_rep
          [Z, X] = mut_bMBP_rev(Z0, a, a_dt, 10^theta, chkt);
          Y(i) = sqrt(sqrt(X/Z));
    end
    m = mean(Y);
    s = std(Y);
end