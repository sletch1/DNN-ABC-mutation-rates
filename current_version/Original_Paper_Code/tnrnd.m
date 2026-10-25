function sample = tnrnd(m, s, range, n)
% Generate truncated normal random sample
% m: mean of normal distribution
% s: sd of normal distribution
% range: vector, range(1): lower bound; range(2): upper bound
% n: number of random sample
% sample: random sample
    untruncated = makedist('Normal', m, s);
    truncated = truncate(untruncated, range(1), range(2));
    sample = random(truncated, 1, n);
end