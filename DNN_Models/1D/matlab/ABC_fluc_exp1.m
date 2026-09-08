function [sample_theta, accp_rate, train_time] = ABC_fluc_exp1(nMCMC, Z_vec, X_vec, theta_ini, s, range, ns, a, t0, J, nsample, sigma0, kparams0, eps, gps)
% ROLE: the ORIGINAL ABC / GPS-ABC sampler from the professor's code drop, and
% the reference this project's DNN_Models/1D/abc/abc_mcmc.py is ported from.
% Random-walk Metropolis-Hastings on theta = log10(p) with a truncated-normal
% proposal; gps=false runs exact ABC-MCMC (re-simulating every iteration),
% gps=true swaps in the GP surrogate from trainGPS.m.
%
% Superseded by ABC_fluc_exp1_rev.m in this folder, which adds Z0/delta and
% replaces this file's UNIFORM prior with a truncated shifted exponential
% (lambda = 2). Both are kept: this one documents the prior and acceptance
% form the Python port matches, the _rev one the prior the paper actually uses.
% Estimate mutation rate for fluctuation experiment with constant mutation rate
% Bayesian setting: prior for theta ~ unif, proposal theta_can|theta ~ TN
% nMCMC: number of MCMC iterations
% Z_vec: vector of total # of viable cells at t0 for J cultures
% X_vec: vector of # of mutants at t0 for J cultures
% theta_ini: initial value of theta
% s: proposal TN sd
% range: proposal TN bound, (lb = -7.5, ub = -1.5) for mutation rate in log10 scale
% ns: number of simulated samples
% a: rate parameter of exponential life time
% t0: time of plating
% J: number of parallel cultures
% nsample: number of training samples per grid point
% sigma0: initial value for the noise sd of the GP model
% kparams0: initial values for the kernel parameters, the length scale and the signal sd
% eps: normal sd for obs|sim
% gps: boolean scalar, false: ABC-MCMC, true: GPS-ABC
% sample_theta: posterior sample of theta
% accp_rate: acceptance rate

train_time = NaN;
if gps == true
    theta_vec = linspace(range(1), range(2), 51)';
    p_vec = 10 .^ theta_vec;
    tic;
    [gprMd, ~, ~] = trainGPS(a, p_vec, t0, J, nsample, sigma0, kparams0);
    train_time = toc;
end

obs = mean(sqrt(X_vec ./ Z_vec));
sample_theta = NaN(1, nMCMC);
% MOM = (1 - log(mean(Z_vec - X_vec)) / log(mean(Z_vec))) / 2;
% sample_theta(1) = log10(MOM); % use MOM as initial value
sample_theta(1) = theta_ini;
naccp = 0;
for i = 2 : nMCMC
    theta = sample_theta(i - 1);
    sample_theta(i) = theta;
    theta_can = tnrnd(theta, s, range, 1);
    delta = (normcdf((range(2) - theta) / s) - normcdf((range(1) - theta) / s)) / (normcdf((range(2) - theta_can) / s) - normcdf((range(1) - theta_can) / s));
    like_single = NaN(1, ns);
    like_can_single = NaN(1, ns);
    for j = 1 : ns
        if gps == false
            [Z_sim, X_sim] = fluc_exp1(a, 10 ^ theta, t0, J);
            sim = mean(sqrt(X_sim ./ Z_sim));
            [Z_can_sim, X_can_sim] = fluc_exp1(a, 10 ^ theta_can, t0, J);
            sim_can = mean(sqrt(X_can_sim ./ Z_can_sim));
        else
            [Spred, Ssd] = predict(gprMd, theta);
            sim = normrnd(Spred, Ssd);
            [Spred_can, Ssd_can] = predict(gprMd, theta_can);
            sim_can = normrnd(Spred_can, Ssd_can);
        end
        like_single(j) = normpdf(obs, sim, eps);
        like_can_single(j) = normpdf(obs, sim_can, eps);
    end
    like = mean(like_single);
    like_can = mean(like_can_single);
    alpha = min(1, like_can / like * delta);
    if (unifrnd(0, 1) < alpha)
        sample_theta(i) = theta_can;
        naccp = naccp + 1;
    end
end
accp_rate = naccp / (nMCMC - 1);

end
