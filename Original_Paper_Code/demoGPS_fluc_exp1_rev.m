% For Figure 2, demonstrate GP regression for multiple samples of parallel cultured data with constant mutation
addpath('C:/Users/Xiaowei/Documents/Work/MutationProject/Revision/Code');
a = 1;
Z0 = 1;
theta_vec = (-8 : 0.06 : -2)';
p_vec = 10 .^ theta_vec;
np = length(p_vec);
tp_vec = NaN(1, np);
c = 20;
for i = 1 : np
    p = p_vec(i);
    myfun = @(t, Z0, a, p, c) Z0 * (exp(a * t) - exp(a * t * (1 - 2 * p))) - c;  % parameterized function
    fun = @(t) myfun(t, Z0, a, p, c);    % function of x alone
    tp = fzero(fun, 20);
    tp_vec(i) = tp;
end
J = 100;
nsample = 10;
sigma0 = 0.02;
kparams0 = [1, 1];
rng(0);
tic;
[gprMd, theta_rep_vec, S_vec] = trainGPS(Z0, a, p_vec, tp_vec, J, nsample, sigma0, kparams0);
toc;
[Spred, Ssd, Sint] = predict(gprMd, theta_vec);
figure();
plot(theta_rep_vec, S_vec, 'r.', 'MarkerSize', 12);
hold on
plot(theta_vec, Spred, 'b-', 'LineWidth', 1.5);
h = fill([theta_vec', fliplr(theta_vec')], [Sint(:, 1)', fliplr(Sint(:, 2)')], [0.8, 0.8, 0.8], 'LineStyle', '--');
set(h, 'facealpha', 0.5);
xlabel('Mutation rate in $\log_{10}$ scale', 'interpreter', 'latex');
ylabel('Summary statistic VS. Predictive mean');
legend({'$\overline{\sqrt{x/z}}$, 10 replicates at each grid point', 'Predictive mean in GP regression', '95\% predictive interval in GP regression'}, 'interpreter', 'latex', 'Location', 'Best');
hold off;
set(gcf, 'PaperUnits', 'centimeters', 'PaperPosition', [0 0 16 12]);
saveas(gcf, 'C:/Users/Xiaowei/Documents/Work/MutationProject/Revision/Result/Fig2_rev', 'epsc');
% saveas(gcf, 'C:/Users/Xiaowei/Documents/Work/MutationProject/Revision/Result/Fig2_rev', 'pdf');