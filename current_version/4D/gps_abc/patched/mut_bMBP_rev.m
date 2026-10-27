function [Z, X] = mut_bMBP_rev(Z0, a, delta, p, tp)
% PATCHED COPY of Original_Paper_Code/mut_bMBP_rev.m (the original is left untouched).
% The only change is the draw of Z. The original builds geornd(exp(-a*tp) .* ones(1, Z0)) and sums it,
% which allocates a Z0-element array; in the second stage of the two-stage model Z0 is about e^tau, i.e. up to
% ~4e8 at the paper's tau <= 19.9, and the MATLAB workers run out of memory. A sum of Z0 iid geometric (failures)
% variables is exactly NegativeBinomial(Z0, p), so Z has the same distribution and no array is allocated.
%   Z = sum(geornd(exp(-a * tp) .* ones(1, Z0)))   -->   Z = nbinrnd(Z0, exp(-a * tp))
% Everything else is identical to the original.
if Z0 <= 0
    Z = 0;
else
    Z = nbinrnd(Z0, exp(-a * tp));
end
M = round(Z * p);
if M > 0
    arrtime_vec = log(unifrnd(0, 1, 1, M) .* (exp(a * tp) - 1) + 1) ./ a;
    X = sum(geornd(exp(-(a * delta) .* (tp - arrtime_vec)))) + M;
else
    X = 0;
end
end
