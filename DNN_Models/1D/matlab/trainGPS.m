function [gprMd, theta_rep_vec, S_vec] = trainGPS(a, p_vec, t0, J, nsample, sigma0, kparams0)
% ROLE: the ORIGINAL GP trainer from the professor's code drop -- the file that
% trainGPS_rev.m in this folder was reconstructed from. Fits a GP mapping
% theta = log10(p) to the summary statistic mean(sqrt(X/Z)) over a grid of
% p_vec, which is the surrogate GPS-ABC queries in place of the simulator.
%
% Differs from trainGPS_rev.m in taking no Z0 argument and calling the
% pre-delta fluc_exp1.m. Kept unmodified as the provenance for the paper's
% GPS-ABC baseline and for the reconstruction note in trainGPS_rev.m.
% Train GPS model for fluctuation experiment with constant mutation rate
% a: rate parameter of exponential life time
% p_vec: grid of mutation probability, column vector
% t0: time of plating
% J: number of parallel cultures
% nsample: number of training samples
% sigma0: initial value for the noise sd of the GP model
% kparams0: initial values for the kernel parameters, the length scale and the signal sd
% gprMd: GP regression model
% theta_rep_vec: parameter vector with each element repeated nsample times, theta = log10(p)
% S_vec: summary statistic vector corresponding to theta_rep_vec

np = length(p_vec);
S_mat = NaN(np, nsample); % sqrt(X/Z)
for i = 1 : np
    p = p_vec(i);
    for j = 1 : nsample
        [Z_vec, X_vec] = fluc_exp1(a, p, t0, J);
        S_mat(i, j) = mean(sqrt(X_vec ./ Z_vec));
    end
end
S_vec = reshape(S_mat', [np * nsample, 1]);
theta_vec = log10(p_vec);
theta_rep_vec = repelem(theta_vec, nsample);
% theta_rep_vec = reshape(repmat(theta_vec', nsample, 1), [], 1); % for old version Matlab
gprMd = fitrgp(theta_rep_vec, S_vec, 'KernelFunction', 'squaredexponential', 'KernelParameters', kparams0, 'Sigma', sigma0);
end
