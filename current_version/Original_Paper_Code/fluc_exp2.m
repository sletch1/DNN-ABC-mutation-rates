function [Z_vec, X_vec] = fluc_exp2(Z0, a, p1, p2, tau, tp, J)
% Generate fluctuation data for parallel cultures based on 2-stage mutation rate assumption
% Z0: # of non-mutants at t = 0
% a: rate parameter of exponential life time
% p1: mutation probability in stage 1
% p2: mutation probability in stage 2
% tau: transition time from stage 1 to stage 2
% tp: time of plating
% J: number of parallel cultures
% Z_vec: vector of total # of viable cells at tp for J cultures
% X_vec: vector of # of mutants at tp for J cultures

Z_vec = zeros(1, J);
X_vec = zeros(1, J);
for i = 1 : J
    [Zt, Xt] = mut2stage_bMBP(Z0, a, p1, p2, tau, tp);
    Z_vec(i) = Zt;
    X_vec(i) = Xt;
end
