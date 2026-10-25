function [gprMd, theta_rep_vec, S_vec] = trainGPS(Z0, a, p_vec, tp_vec, J, nsample, sigma0, kparams0)
% Train GPS model for fluctuation experiment with constant mutation rate
% Z0: # of non-mutants at t = 0
% a: rate parameter of exponential life time
% p_vec: grid of mutation probability, column vector
% tp_vec: time of plating corresponding to p_vec
% J: number of parallel cultures
% nsample: number of training samples
% sigma0: initial value for the noise sd of the GP model
% kparams0: initial values for the kernel parameters, the length scale and the signal sd
% gprMd: GP regression model
% theta_rep_vec: parameter vector with each element repeated nsample times, theta = log10(p)
% S_vec: summary statistic vector corresponding to theta_rep_vec

delta = 1;
np = length(p_vec);
S_mat = NaN(np, nsample); % sqrt(X/Z)
for i = 1 : np
    p = p_vec(i);
    tp = tp_vec(i);
    for j = 1 : nsample
%         [Z_vec, X_vec] = fluc_exp1(Z0, a, p, tp, J);
        [Z_vec, X_vec] = fluc_exp1_rev(Z0, a, delta, p, tp, J);
        S_mat(i, j) = mean(sqrt(X_vec ./ Z_vec));
    end
end
S_vec = reshape(S_mat', [np * nsample, 1]);
theta_vec = log10(p_vec);
theta_rep_vec = repelem(theta_vec, nsample);
% theta_rep_vec = reshape(repmat(theta_vec', nsample, 1), [], 1); % for old version Matlab
gprMd = fitrgp(theta_rep_vec, S_vec, 'KernelFunction', 'squaredexponential', 'KernelParameters', kparams0, 'Sigma', sigma0);
end
