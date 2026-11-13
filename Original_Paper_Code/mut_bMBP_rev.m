function [Z, X] = mut_bMBP_rev(Z0, a, delta, p, tp)
% Generate (z, x) data for bMBP model with constant mutation
% Z0: # of non-mutants at t = 0
% a: rate parameter of exponential life time for non-mutants
% delta: growth parameter for mutants relative to non-mutants
% p: mutation probability of each single particle
% tp: time of plating
% Z: total # of viable cells at tp
% X: # of mutants at tp

Z = sum(geornd(exp(-a * tp) .* ones(1, Z0)));
M = round(Z * p); % using binornd(Z, p) is too time consuming
if M > 0
    arrtime_vec = log(unifrnd(0, 1, 1, M) .* (exp(a * tp) - 1) + 1) ./ a;
%     arrtime_vec = repelem(arrtime_vec, 2); %%% pre-division mutation
    X = sum(geornd(exp(-(a * delta) .* (tp - arrtime_vec)))) + M;
else
    X = 0;
end
end