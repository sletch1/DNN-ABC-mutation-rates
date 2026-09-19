% Test code for function mut_bMBP and mut2stage_bMBP
addpath('C:/Users/Xiaowei/Documents/Work/MutationProject/MyCode');
% example 1: constant mutation rate
a = 1;
p = 1e-7; % 0.01;
t0 = 20; % 10
nsimu = 1; % 1000;
Z1_vec = zeros(1, nsimu);
X1_vec = zeros(1, nsimu);
rng(1);
tic;
for i = 1 : nsimu
    [Z, X] = mut_bMBP(a, p, t0);
    Z1_vec(i) = Z;
    X1_vec(i) = X;
end
toc;
% sum(Z1_vec == X1_vec)
% figure;
% ecdf(Z1_vec);
% hold on;
% p = exp(-a * t0);
% x = linspace(min(Z1_vec), max(Z1_vec), 1000);
% plot(x, geocdf(x, p), 'r-')
% figure;
% ecdf(Z1_vec - X1_vec);
% hold on;
% ptilde = (p ^ 2) / (p ^ 2 + (1 - p) ^ 2);
% c = a * (1 - 2 * p);
% q = 1 - (1 - ptilde) * (exp(c * t0) - 1) / ((1 - ptilde) * exp(c * t0) - ptilde);
% x = linspace(min(Z1_vec - X1_vec), max(Z1_vec - X1_vec), 1000);
% plot(x, geocdf(x, q), 'r-')

% example 2: 2-stage mutation rate
a = 1;
p1 = 0.001;
p2 = 0.02;
tau = 5;
t0 = 10;
nsimu = 1000;
Z2_vec = zeros(1, nsimu);
X2_vec = zeros(1, nsimu);
rng(1);
tic;
for i = 1 : nsimu
    [Z, X] = mut2stage_bMBP(a, p1, p2, tau, t0);
    Z2_vec(i) = Z;
    X2_vec(i) = X;
end
toc;

figure;
plot(Z2_vec, X2_vec, '.');
figure;
plot(X1_vec, X2_vec, '.');

