function [sample_theta, accp_rate, train_time] = ABC_fluc_exp1(nMCMC, Z_vec, X_vec, theta_ini, s, range, ns, a, tp, J, nsample, sigma0, kparams0, eps, gps)
% Estimate mutation rate for fluctuation experiment with constant mutation rate
% Bayesian setting: prior for theta ~ unif, proposal theta_can|theta ~ TN
% nMCMC: number of MCMC iterations
% Z_vec: vector of total # of viable cells at tp for J cultures
% X_vec: vector of # of mutants at tp for J cultures
% theta_ini: initial value of theta
% s: proposal TN sd
% range: proposal TN bound, *(lb = -10, ub = -2) for mutation rate in log10 scale
% ns: number of simulated samples
% a: rate parameter of exponential life time
% tp: time of plating
% J: number of parallel cultures
% nsample: number of training samples per grid point
% sigma0: initial value for the noise sd of the GP model
% kparams0: initial values for the kernel parameters, the length scale and the signal sd
% eps: normal sd for obs|sim
% gps: boolean scalar, false: ABC-MCMC, true: GPS-ABC
% sample_theta: posterior sample of theta
% accp_rate: acceptance rate

Z0 = 1;
lambda = 2; % added last for truncated exp prior
delt = 1;
train_time = NaN;
if gps == true
    theta_vec = linspace(range(1), range(2), 51)';
    p_vec = 10 .^ theta_vec;
    tic;
    [gprMd, ~, ~] = trainGPS(Z0, a, p_vec, tp, J, nsample, sigma0, kparams0);
    train_time = toc;
end

obs = mean(sqrt(X_vec ./ Z_vec));
sample_theta = NaN(1, nMCMC);
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
            [Z_sim, X_sim] = fluc_exp1(Z0, a, 10 ^ theta, tp, J);
%             [Z_sim, X_sim] = fluc_exp1_rev(Z0, a, delt, 10 ^ theta, tp, J);
            sim = mean(sqrt(X_sim ./ Z_sim));
            [Z_can_sim, X_can_sim] = fluc_exp1(Z0, a, 10 ^ theta_can, tp, J);
%             [Z_can_sim, X_can_sim] = fluc_exp1_rev(Z0, a, delt, 10 ^ theta_can, tp, J);
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
    prior = exp(lambda * (range(1) - theta)) ./ (1 - exp(lambda * (range(1) - range(2)))); % prior: truncated shifted exp(lambda), assume lambda = 1
    prior_can = exp(lambda * (range(1) - theta_can)) ./ (1 - exp(lambda * (range(1) - range(2))));
%     alpha = min(1, like_can / like * delta);
    alpha = exp(min(0, log(like_can) - log(like) + log(delta) + log(prior_can) - log(prior)));
    if (unifrnd(0, 1) < alpha)
        sample_theta(i) = theta_can;
        naccp = naccp + 1;
    end
end
accp_rate = naccp / (nMCMC - 1);

end