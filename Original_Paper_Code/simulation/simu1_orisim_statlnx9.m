% nohup matlab -nodisplay -nosplash -r "i = 1; j = 1" <./simu1_orisim_statlnx9.m> ./temp01.txt &
% nohup matlab -nodisplay -nosplash -r "i = 1; j = 2" <./simu1_orisim_statlnx9.m> ./temp02.txt &
% nohup matlab -nodisplay -nosplash -r "i = 1; j = 3" <./simu1_orisim_statlnx9.m> ./temp03.txt &
% nohup matlab -nodisplay -nosplash -r "i = 2; j = 1" <./simu1_orisim_statlnx9.m> ./temp04.txt &
% nohup matlab -nodisplay -nosplash -r "i = 2; j = 2" <./simu1_orisim_statlnx9.m> ./temp05.txt &
% nohup matlab -nodisplay -nosplash -r "i = 2; j = 3" <./simu1_orisim_statlnx9.m> ./temp06.txt &
% nohup matlab -nodisplay -nosplash -r "i = 3; j = 1" <./simu1_orisim_statlnx9.m> ./temp07.txt &
% nohup matlab -nodisplay -nosplash -r "i = 3; j = 2" <./simu1_orisim_statlnx9.m> ./temp08.txt &
% nohup matlab -nodisplay -nosplash -r "i = 3; j = 3" <./simu1_orisim_statlnx9.m> ./temp09.txt &
addpath('/home/hongxiao/xiaowei/ABC/Revision/Code');
nMCMC = 100;
burnin = nMCMC * 9 / 10;
s = 0.15;
range = [-5, -1];
ns = 10; % 50, 100 too slow
nsample = 10;
sigma0 = 0.02;
kparams0 = [1, 1];
eps = 0.005;

load(strcat('/home/hongxiao/xiaowei/ABC/Revision/Result/simu1data_orisim_i', int2str(i), '_j', int2str(j), '.mat'));

sample_theta_GPS_mat = NaN(nsimu, nMCMC);
accp_rate_GPS_vec = NaN(1, nsimu);
total_time_GPS_vec = NaN(1, nsimu);
train_time_GPS_vec = NaN(1, nsimu);
phat_GPS_vec = NaN(1, nsimu);

sample_theta_ABC_mat = NaN(nsimu, nMCMC);
accp_rate_ABC_vec = NaN(1, nsimu);
total_time_ABC_vec = NaN(1, nsimu);
phat_ABC_vec = NaN(1, nsimu);
for k = 1 : nsimu
	Z_vec = Z_mat(k, :);
	X_vec = X_mat(k, :);
	theta_ini = log10(phat_MOM_vec(k));
	clock1 = tic;
	[sample_theta_GPS, accp_rate_GPS, train_time_GPS] = ABC_fluc_exp1(nMCMC, Z_vec, X_vec, theta_ini, s, range, ns, a, tp, J, nsample, sigma0, kparams0, eps, true);
	total_time_GPS = toc(clock1);
	phat_GPS = 10 ^ mean(sample_theta_GPS((burnin + 1) : end));
	sample_theta_GPS_mat(k, :) = sample_theta_GPS;
	accp_rate_GPS_vec(k) = accp_rate_GPS;
	train_time_GPS_vec(k) = train_time_GPS;
	total_time_GPS_vec(k) = total_time_GPS;
	phat_GPS_vec(k) = phat_GPS;
	
	clock2 = tic;
	[sample_theta_ABC, accp_rate_ABC, ~] = ABC_fluc_exp1(nMCMC, Z_vec, X_vec, theta_ini, s, range, ns, a, tp, J, nsample, sigma0, kparams0, eps, false);
	total_time_ABC = toc(clock2);
	phat_ABC = 10 ^ mean(sample_theta_ABC((burnin + 1) : end));
	sample_theta_ABC_mat(k, :) = sample_theta_ABC;
	accp_rate_ABC_vec(k) = accp_rate_ABC;
	total_time_ABC_vec(k) = total_time_ABC;
	phat_ABC_vec(k) = phat_ABC;
end
save(strcat('/home/hongxiao/xiaowei/ABC/Revision/Result/simu1result_orisim_i', int2str(i), '_j', int2str(j), '.mat'));
