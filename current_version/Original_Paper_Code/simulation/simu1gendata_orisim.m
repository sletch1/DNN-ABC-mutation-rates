addpath('C:/Users/Xiaowei/Documents/Work/MutationProject/Revision/Code');
p_vec = [1e-4, 1e-3, 1e-2];
np = length(p_vec);
tp_vec = [10, 8, 6];
J_vec = [10, 50, 100];
nJ = length(J_vec);

Z0 = 1;
a = 1;
delta = 1;
nsimu = 100;

for i = 1 : np
    p = p_vec(i);
    tp = tp_vec(i);
    for j = 1 : nJ
        J = J_vec(j);
		Z_mat = NaN(nsimu, J);
		X_mat = NaN(nsimu, J);
        phat_MOM_vec = NaN(1, nsimu);
        phat_MLE_vec = NaN(1, nsimu);
        seed = i * 10 + j;
        rng(seed);
        for k = 1 : nsimu
            [Z_vec, X_vec] = fluc_exp1(Z0, a, p, tp, J);
%            [Z_vec, X_vec] = fluc_exp1_rev(Z0, a, delta, p, tp, J);
            Z_mat(k, :) = Z_vec;
            X_mat(k, :) = X_vec;

            [phat_MOM, phat_MLE] = MOMMLE_fluc_exp1(Z_vec, X_vec);
            phat_MOM_vec(k) = phat_MOM;
            phat_MLE_vec(k) = phat_MLE;
        end
        save(strcat('C:/Users/Xiaowei/Documents/Work/MutationProject/Revision/Result/simu1data_orisim_i', int2str(i), '_j', int2str(j), '.mat'));
    end
end
