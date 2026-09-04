function [gprMd, theta_rep_vec, S_vec] = trainGPS_rev(Z0, a, p_vec, tp, J, nsample, sigma0, kparams0)
% ROLE: fits the Gaussian-process surrogate that IS GPS-ABC -- trains a GP
% mapping theta = log10(p) to the summary statistic mean(sqrt(X/Z)) on a grid
% of p_vec, so ABC_fluc_exp1_rev.m can call predict(gprMd, theta) instead of
% re-running the simulator inside the MCMC loop. This is the surrogate the
% neural network in DNN_Models/ is benchmarked against.
%
% RECONSTRUCTED, not original: ABC_fluc_exp1_rev.m calls trainGPS_rev(Z0, a,
% p_vec, tp, J, nsample, sigma0, kparams0), but no file by that name shipped
% in the professor's code drops -- only the older trainGPS.m (no Z0 arg,
% built on the pre-delta fluc_exp1). This file closes that gap by applying the
% exact same Z0/delta edit that turns fluc_exp1.m into fluc_exp1_rev.m and
% mut_bMBP.m into mut_bMBP_rev.m, with delta hardcoded to 1 to match every
% other call site in ABC_fluc_exp1_rev.m (delt = 1). Spot-check against MATLAB
% before relying on it -- this was inferred from the call site, not verified
% by running it against the professor's own output.
%
% Z0: # of non-mutants at t = 0
% a: rate parameter of exponential life time
% p_vec: grid of mutation probability, column vector
% tp: time of plating
% J: number of parallel cultures
% nsample: number of training samples per grid point
% sigma0: initial value for the noise sd of the GP model
% kparams0: initial values for the kernel parameters, the length scale and the signal sd
% gprMd: GP regression model
% theta_rep_vec: parameter vector with each element repeated nsample times, theta = log10(p)
% S_vec: summary statistic vector corresponding to theta_rep_vec

delta = 1; % matches the fixed delt = 1 used throughout ABC_fluc_exp1_rev.m

np = length(p_vec);
S_mat = NaN(np, nsample); % sqrt(X/Z)
for i = 1 : np
    p = p_vec(i);
    for j = 1 : nsample
        [Z_vec, X_vec] = fluc_exp1_rev(Z0, a, delta, p, tp, J);
        S_mat(i, j) = mean(sqrt(X_vec ./ Z_vec));
    end
end
S_vec = reshape(S_mat', [np * nsample, 1]);
theta_vec = log10(p_vec);
theta_rep_vec = repelem(theta_vec, nsample);
gprMd = fitrgp(theta_rep_vec, S_vec, 'KernelFunction', 'squaredexponential', 'KernelParameters', kparams0, 'Sigma', sigma0);
end
