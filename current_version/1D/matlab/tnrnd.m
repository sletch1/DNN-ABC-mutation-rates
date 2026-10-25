function sample = tnrnd(m, s, range, n)
% ROLE: truncated-normal RNG for the MCMC proposal step in ABC_fluc_exp1_rev.m --
% draws candidate theta ~ N(m, s) truncated to `range`, used for both the
% gps=false (raw simulator) and gps=true (GP surrogate) branches of the sampler.
% Was missing from this folder even though ABC_fluc_exp1_rev.m calls it; pulled
% in from the professor's original code to make that sampler runnable again.
    untruncated = makedist('Normal', m, s);
    truncated = truncate(untruncated, range(1), range(2));
    sample = random(truncated, 1, n);
end
