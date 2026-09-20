% ROLE: THE 3-D / TWO-STAGE REFERENCE. This is the professor's Study 2 setup --
% the GP surrogate for the two-stage model with parameters (mu1, mu2, jmpt),
% and therefore the file Models/3D is benchmarked against.
%
% Read it before changing anything in Models/3D/abc/surrogates.py. Its
% GPS-ABC baseline differs from ours in several load-bearing ways:
%   - the GP is fit on RAW (mu1, mu2, jmpt), not log10 of the rates;
%   - it predicts RAW S = mean(sqrt(X/Z)), not log10(d_bar);
%   - the kernel is an ISOTROPIC squaredexponential (kparams0 = [1, 1],
%     sigma0 = 0.02), not an anisotropic RBF with a learned white-noise term;
%   - J = 30, a full factorial 11 x 11 x 9 = 1089 design, one replicate each.
% Our implementation is a stronger GP than this on every one of those axes,
% which is why the manuscript now reports a faithful-reference GP alongside it.
%
% Note also that fluc_exp2(a, mu1, mu2, jmpt, chkt, J), called on line ~22, did
% not ship in any of the professor's code drops; it is the J-culture wrapper
% around mut2stage_bMBP.m (Models/3D/matlab/).
addpath('C:/Users/Xiaowei/Documents/Work/MutationProject/MyCode');
a = 1;
log10p1_vec = (-7 : 0.5 : -2)';
log10p2_vec = (-7 : 0.5 : -2)';
jmpt_vec = (1 : 9)';
mu1_vec = 10 .^ log10p1_vec;
mu2_vec = 10 .^ log10p2_vec;
nmu1 = length(mu1_vec);
nmu2 = length(mu2_vec);
njmpt = length(jmpt_vec);
[X, Y, Z] = meshgrid(mu1_vec, mu2_vec, jmpt_vec);
Cov_mat = [X(:), Y(:), Z(:)];
chkt = 10;
J = 30;
S_vec = NaN(nmu1 * nmu2 * njmpt, 1); % sqrt(X/Z)
rng(1);
tic;
for i = 1 : (nmu1 * nmu2 * njmpt)
    mu1 = Cov_mat(i, 1);
    mu2 = Cov_mat(i, 2);
    jmpt = Cov_mat(i, 3);
    [Zt_vec, Xt_vec] = fluc_exp2(a, mu1, mu2, jmpt, chkt, J);
    S_vec(i, 1) = mean(sqrt(Xt_vec ./ Zt_vec));
end
toc;
sigma0 = 0.02;
kparams0 = [1, 1];
gprMd = fitrgp(Cov_mat, S_vec, 'KernelFunction', 'squaredexponential', 'KernelParameters', kparams0, 'Sigma', sigma0);
[Spred, Ssd, Sint] = predict(gprMd, Cov_mat);
temp = [S_vec, Spred];
% for i = 1 : nmu1
%     mu1 = mu1_vec(i);
%     for j = 1 : nmu2
%         mu2 = mu2_vec(j);
%         for k = 1 : njmpt
%             jmpt = jmpt_vec(k);
%             [Zt_vec, Xt_vec] = fluc_exp2(a, mu1, mu2, jmpt, chkt, J);
%             S_vec(i, 1) = mean(sqrt(Xt_vec ./ Zt_vec));
%         end
%     end
% end
% , 'Basis', 'linear', 'FitMethod', 'exact', 'PredictMethod', 'exact'
% Spred = resubPredict(gprMd);
figure();
plot(Spred, S_vec - Spred, 'r.', 'MarkerSize', 12);

xlabel('Mutation rate in $\log_{10}$ scale', 'interpreter', 'latex');
% ylabel('$\overline{\sqrt{X_t/Z_t}}$', 'interpreter', 'latex');
ylabel('Summary statistic VS. Predicted mean');
% legend({'data', strcat('kparams0 =', mat2str(kparams0), ', sigma0 =', num2str(sigma0))}, 'Location', 'Best');
legend({'Summary statistic $\overline{\sqrt{X_t/Z_t}}$', 'Predicted mean', '95\% prediction intervals'}, 'interpreter', 'latex', 'Location', 'Best');
hold off;
set(gcf, 'PaperUnits', 'centimeters', 'PaperPosition', [0 0 16 12]);
saveas(gcf, 'C:/Users/Xiaowei/Documents/Work/MutationProject/MyResult/Fig2', 'epsc');

x = 1 : 10; y = 0.1 : 0.1 : 1; z = 1.1 : 0.1 : 2;
[X, Y, Z] = meshgrid(x, y, z);
