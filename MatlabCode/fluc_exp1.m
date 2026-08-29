function [Z_vec, X_vec] = fluc_exp1(Z0, a, p, tp, J)
% ROLE: one fluctuation experiment = J parallel cultures. Original version, no
% differential mutant growth (see fluc_exp1_rev.m for the delta-aware one).
% Generate fluctuation data for parallel cultures based on constant mutation rate assumption
% Z0: # of non-mutants at t = 0
% a: rate parameter of exponential life time
% p: mutation probability of each single particle
% tp: time of plating
% J: number of parallel cultures
% Z_vec: vector of total # of viable cells at t0 for J cultures
% X_vec: vector of # of mutants at t0 for J cultures

Z_vec = zeros(1, J);
X_vec = zeros(1, J);
for i = 1 : J
    [Zt, Xt] = mut_bMBP(Z0, a, p, tp);
    Z_vec(i) = Zt;
    X_vec(i) = Xt;
end
end