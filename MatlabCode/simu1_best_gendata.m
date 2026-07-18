addpath('C:/Users/xwwu/Documents/Work/ABC/Code');
p_vec = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4];
tp_vec = [20, 18, 16, 14, 12];
% E[X]: 193.0660 235.3755 283.3510 335.6821 389.1430
np = length(p_vec);
nrep = 10;
J_vec = [10, 50, 100];
nJ = length(J_vec);
Z0 = 1;
a = 1;
delta = 1;
nsimu = 100;

tic;
for i = 1 : np
    p = p_vec(i);
    tp = tp_vec(i);
    for j = 1 : nJ
        J = J_vec(j);
        Z_cub = NaN(nsimu, J, nrep);
        X_cub = NaN(nsimu, J, nrep);
        phat_MOM_mat = NaN(nsimu, nrep);
        phat_MLE_mat = NaN(nsimu, nrep);
        for k = 1 : nrep
            seed = nrep * (nJ * (i - 1) + j - 1) + k;
            rng(seed);
            for l = 1 : nsimu
                [Z_vec, X_vec] = fluc_exp1_rev(Z0, a, delta, p, tp, J);
                Z_cub(l, :, k) = Z_vec;
                X_cub(l, :, k) = X_vec;
                [phat_MOM, phat_MLE] = MOMMLE_fluc_exp1(Z_vec, X_vec);
                phat_MOM_mat(l, k) = phat_MOM;
                phat_MLE_mat(l, k) = phat_MLE;
            end
        end
        save(strcat('C:/Users/xwwu/Documents/Work/ABC/Result/simu1data_i', int2str(i), '_j', int2str(j), '.mat'));
    end
end
toc;
