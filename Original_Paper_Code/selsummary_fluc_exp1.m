% For Figure 1, choose summary statistic for parallel cultured data with constant mutation
addpath('C:/Users/xwwu/Documents/Work/ABC/Code');
a = 1;
p_vec = linspace(1e-8, 1e-2, 1001);
np = length(p_vec);
% tp = 20;
% p_vec = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4];
% tp_vec = [20, 18, 16, 14, 12];
% exp(a .* tp_vec) - ((1 - p_vec) .* exp(a .* tp_vec .* (1 - 2 * p_vec)) - p_vec) ./ (1 - 2 * p_vec)

c = 20;
J = 100; %30
Z0 = 1;
delta = 1;
tp_vec = NaN(1, np);
S1_vec = NaN(1, np); % [1 - log(bar{Y})/log(bar{Z})] / 2
S2_vec = NaN(1, np); % log(Y/Z)
S3_vec = NaN(1, np); % sqrt(X/Z)
rng(1);
tic;
for i = 1 : np
    p = p_vec(i);
    myfun = @(t, Z0, a, p, c) Z0 * (exp(a * t) - exp(a * t * (1 - 2 * p))) - c;  % parameterized function
    fun = @(t) myfun(t, Z0, a, p, c);    % function of x alone
    tp = fzero(fun, 20);
    tp_vec(i) = tp;
%     myfun = @(t, p, c) exp(t) - exp(t * (1 - 2 * p)) - c;  % parameterized function
%     fun = @(t) myfun(t, p, c);    % function of x alone
    [Z_vec, X_vec] = fluc_exp1(Z0, a, p, tp, J);
%     [Z_vec, X_vec] = fluc_exp1_rev(Z0, a, delta, p, tp, J);
    [~, phat] = MOMMLE_fluc_exp1(Z_vec, X_vec);
    S1_vec(i) = phat;
    S2_vec(i) = mean(log(1 - X_vec ./ Z_vec));
    S3_vec(i) = mean(sqrt(X_vec ./ Z_vec));
end
toc;
figure;
subplot(2, 2, 1);
plot(p_vec, S1_vec, 'k-');
xlabel('$p$', 'interpreter', 'latex', 'Position', [0.0105, 0.0015]);
ylabel('$\hat{p}_{MOM}$', 'interpreter', 'latex');
title('(A) $\hat{p}_{MOM}$ VS. $p$', 'interpreter', 'latex');
hold on;
plot([min(p_vec), max(p_vec)], [min(p_vec), max(p_vec)], 'r--', 'LineWidth', 2);
subplot(2, 2, 2);
plot(p_vec, S2_vec, 'k-');
xlabel('$p$', 'interpreter', 'latex', 'Position', [0.0105, -0.185]);
ylabel('$\overline{\log(Y/Z)}$', 'interpreter', 'latex');
title('(B) $\overline{\log(Y/Z)}$ VS. $p$', 'interpreter', 'latex');
subplot(2, 2, 3);
plot(p_vec, S3_vec, 'k-');
xlabel('$p$', 'interpreter', 'latex', 'Position', [0.0105, 0.02]);
ylabel('$\overline{\sqrt{X/Z}}$', 'interpreter', 'latex');
title('(C) $\overline{\sqrt{X/Z}}$ VS. $p$', 'interpreter', 'latex');
subplot(2, 2, 4);
plot(log10(p_vec), S3_vec, 'k-');
xlabel('$\log_{10}(p)$', 'interpreter', 'latex', 'Position', [-1.1, 0.015]);
ylabel('$\overline{\sqrt{X/Z}}$', 'interpreter', 'latex');
title('(D) $\overline{\sqrt{X/Z}}$ VS. $\log_{10}(p)$', 'interpreter', 'latex');
set(gcf, 'PaperUnits', 'centimeters', 'PaperPosition', [0 0 22 12]);
saveas(gcf, 'C:/Users/xwwu/Documents/Work/ABC/Result/Fig1_rev', 'epsc');
% saveas(gcf, 'C:/Users/xwwu/Documents/Work/ABC/Result/Fig1_rev', 'pdf');
