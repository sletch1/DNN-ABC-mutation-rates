function [Zt, Xt] = mut2stage_bMBP(a, mu1, mu2, jmpt, chkt)
% Generate (z, x) data for bMBP model with 2-stage mutations
% a: rate parameter of exponential life time
% mu1: mutation probability in stage 1
% mu2: mutation probability in stage 2
% jmpt: jumping time
% chkt: checking time
% Zt: overall population size at chkt
% Xt: mutant population size at chkt

Zt = 0;
Xt = 0;
dtvec = exprnd(1 / a, [1, 1]); % unit initial size, death time
mvec = 0; % is mutant?
f_continue = (dtvec < chkt); % flag of particles that will continue to divide
n_continue = sum(f_continue);
Zt = Zt + sum(~f_continue);
Xt = Xt + sum((~f_continue) & (mvec == 1));
while n_continue > 0
	dtvec_last = dtvec(f_continue);
	mvec_last = mvec(f_continue);
	dtvec = repelem(dtvec_last, 2) + exprnd(1 / a, [1, 2 * n_continue]); % 2 offsprings
%   mu = mu1 .* (dtvec_last <= jmpt) + mu2 .* (dtvec_last > jmpt); % mutation probability depending jumping time
% 	mvec = binornd(1, (1 - repelem(mu, 2)) .* repelem(mvec_last, 2) + repelem(mu, 2)); % mutant always produces mutant offsprings
    mu = mu1 .* (dtvec <= jmpt) + mu2 .* (dtvec > jmpt); % mutation probability depending jumping time
	mvec = binornd(1, (1 - mu) .* repelem(mvec_last, 2) + mu); % mutant always produces mutant offsprings
	f_continue = (dtvec < chkt);
	n_continue = sum(f_continue);
	Zt = Zt + sum(~f_continue);
	Xt = Xt + sum((~f_continue) & (mvec == 1));
end
