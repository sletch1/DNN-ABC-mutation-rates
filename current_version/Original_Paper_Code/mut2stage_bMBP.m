function [Z, X] = mut2stage_bMBP(Z0, a, p1, p2, tau, tp)
% Generate (z, x) data for bMBP model with 2-stage mutations
% Z0: # of non-mutants at t = 0
% a: rate parameter of exponential life time
% p1: mutation probability in stage 1
% p2: mutation probability in stage 2
% tau: transition time from stage 1 to stage 2
% tp: time of plating
% Z: total # of viable cells at tp
% X: # of mutants at tp

Z = 0;
X = 0;
dtvec = exprnd(1 / a, [1, Z0]); % unit initial size, death time
mvec = repelem(0, Z0); % is mutant?
f_continue = (dtvec < tp); % flag of particles that will continue to divide
n_continue = sum(f_continue);
Z = Z + sum(~f_continue);
X = X + sum((~f_continue) & (mvec == 1));
while n_continue > 0
	dtvec_last = dtvec(f_continue);
	mvec_last = mvec(f_continue);
	dtvec = repelem(dtvec_last, 2) + exprnd(1 / a, [1, 2 * n_continue]); % 2 offsprings
% 	dtvec = reshape([dtvec_last; dtvec_last], 1, []) + exprnd(1 / a, [1, 2 * n_continue]); % for old version Matlab
    p = p1 .* (dtvec <= tau) + p2 .* (dtvec > tau); % mutation probability depending on tau
	mvec = binornd(1, (1 - p) .* repelem(mvec_last, 2) + p); % mutant always produces mutant offsprings
% 	mvec = binornd(1, (1 - repelem(p, 2)) .* repelem(mvec_last, 2) + repelem(p, 2)); % mutant always produces mutant offsprings
	f_continue = (dtvec < tp);
	n_continue = sum(f_continue);
	Z = Z + sum(~f_continue);
	X = X + sum((~f_continue) & (mvec == 1));
end
end