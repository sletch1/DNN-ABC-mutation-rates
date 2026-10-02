function [Z, X] = mut2stage_bMBP_rev(Z0, a, delta, p1, p2, tau, tp)
% ROLE: the "Revision"-folder fast two-stage engine, added here 2026-10-02
% after pulling github.com/lruijin/ABC_mutation-rate (master branch,
% root-level mut2stage_bMBP_rev.m) -- byte-identical below this header.
% Did not exist locally before; this project's Python "mut2stage_fast"
% (abc/simulator.py) is an independent inverse-CDF/Yule-arrival construction
% and is NOT a port of this file -- the two are different algorithms that
% have not been checked against each other for agreement. This file composes
% two calls to mut_bMBP_rev (stage 1 for duration tau, then stage 2 for the
% remaining tp-tau starting from the stage-1 survivors), not a single
% generation-by-generation loop the way mut2stage_bMBP.m (the exact
% simulator) is. Needs reconciling with abc/simulator.py:mut2stage_fast
% before any claim is made that the two agree.
%
% Generate (z, x) data for bMBP model with constant mutation
% Z0: # of non-mutants at t = 0
% a: rate parameter of exponential life time for non-mutants
% delta: growth parameter for mutants relative to non-mutants
% p1: mutation probability in stage 1
% p2: mutation probability in stage 2
% tau: transition time from stage 1 to stage 2
% tp: time of plating
% Z: total # of viable cells at tp
% X: # of mutants at tp

[Z1, X1] = mut_bMBP_rev(Z0, a, delta, p1, tau);
[Z, X2] = mut_bMBP_rev(Z1 - X1, a, delta, p2, tp - tau);
X = sum(geornd(exp(-(a * delta) .* (tp - tau)), 1, X1)) + X1 + X2;
end
