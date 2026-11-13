function [Z_vec, X_vec] = fluc_exp1_rev(Z0, a, delta, p, tp, J)
% Generate fluctuation data for parallel cultures based on constant mutation rate assumption
% Z0: # of non-mutants at t = 0
% a: rate parameter of exponential life time
% delta: growth parameter for mutants relative to non-mutants
% p: mutation probability of each single particle
% tp: time of plating
% J: number of parallel cultures
% Z_vec: vector of total # of viable cells at tp for J cultures
% X_vec: vector of # of mutants at tp for J cultures

Z_vec = zeros(1, J);
X_vec = zeros(1, J);
for i = 1 : J
    [Zt, Xt] = mut_bMBP_rev(Z0, a, delta, p, tp);
    Z_vec(i) = Zt;
    X_vec(i) = Xt;
end
end