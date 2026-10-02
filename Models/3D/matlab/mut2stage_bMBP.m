function [Z, X] = mut2stage_bMBP(Z0, a, p1, p2, tau, tp)
% ROLE: exact cell-by-cell two-stage engine underneath the 3-D study.
% Verified 2026-10-02 against github.com/lruijin/ABC_mutation-rate (the
% paper's own cited repo, master branch, root-level mut2stage_bMBP.m) --
% byte-identical below this header except for the header itself.
%
% CORRECTS A PRIOR MISMATCH: the file that lived at this path before did not
% match anything in the public repo -- wrong signature (no Z0 argument),
% wrong variable names (mu1/mu2/jmpt/chkt instead of p1/p2/tau/tp), and a
% genuinely different pair of commented-out/live lines that this project's
% "mut_time" convention investigation (abc/simulator.py, tests/
% validate_simulator.py, the manuscript's Study II section) was built
% around. In THIS, the real file, there is no parent/offspring choice at
% all: the live mutation-probability line always indexes p(t) by the
% newly-drawn child's own division time (`dtvec`, not `dtvec_last`) -- i.e.
% unambiguously the "offspring" convention -- and the two commented-out
% lines below are (a) an old-MATLAB-syntax alternative for building `dtvec`
% itself and (b) a redundant `repelem(p, 2)` variant of the mvec line, not a
% parent-vs-offspring switch. See the matching "ROLE" note on
% mut2stage_bMBP_rev.m for the companion fast/revised algorithm found at the
% same commit, which is a different composition (two mut_bMBP_rev calls),
% not a drop-in replacement for this exact simulator.
%
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
