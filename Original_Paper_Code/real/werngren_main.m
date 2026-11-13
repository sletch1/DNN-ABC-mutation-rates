%%%%%%%%%%%%%%%%%% Part 0: add paths of all functions %%%%%%%%%%%%%%%%%%%%%
cd('~/ABC/code_rev');
addpath('~/ABC/code/kernel')
addpath('~/ABC/code/lightspeed')
addpath('~/ABC/code/optim/Matlab')

%%%%%%%%% Part 1: manually input real data and save into .mat file %%%%%%%%%
%%%%%%%%% H37Rv
X_t = [3 4 4 5 5 5 5 5 6 6 6 7 7 8 8 8 9 9 9 10 11 11 12 13 17]'; 
N_t = 2.3 * 1e8;                                                   
p_init = 8.6e-9;
obs_X = sqrt(sqrt(X_t./N_t));                         
%MOM = 0.5*(1-log(mean(N_t)-mean(X_t))/log(mean(N_t)));
[MOM,MLE] =  MOMMLE_fluc_exp1(N_t, X_t);
save('H37Rv.mat','obs_X','p_init','MOM','MLE','X_t','N_t');

%%%%%%%%% E865/94
X_t = [0 1 1 1 2 2 2 2 2 3 3 4 4 4 4 5 5 6 6 6 8 10 11]';
N_t = 0.5 * 1e8;                                                   
p_init = 2.4e-8;
obs_X = sqrt(sqrt(X_t./N_t));                        
%MOM = 0.5*(1-log(mean(N_t)-mean(X_t))/log(mean(N_t)));
[MOM,MLE] =  MOMMLE_fluc_exp1(N_t, X_t);
save('E865.mat','obs_X','p_init','MOM','MLE','X_t','N_t');

%%%%%%%%% E729/94
X_t = [0 0 1 1 1 2 2 2 2 2 2 3 3 3 4 4 4 5 6 7 7 7 10 12 18]';    
N_t = 1.3 * 1e8;                                    
p_init = 9.6e-9;
obs_X = sqrt(sqrt(X_t./N_t));                        
%MOM = 0.5*(1-log(mean(N_t)-mean(X_t))/log(mean(N_t)));
[MOM,MLE] =  MOMMLE_fluc_exp1(N_t, X_t);
save('E729.mat','obs_X','p_init','MOM','MLE','X_t','N_t');

%%%%%%%%% E740/94
X_t = [0 0 1 2 2 2 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 6 9 9 12]';    
N_t = 1 * 1e8;                                    
p_init = 1.1e-8;
obs_X = sqrt(sqrt(X_t./N_t));                        
%MOM = 0.5*(1-log(mean(N_t)-mean(X_t))/log(mean(N_t)));
[MOM,MLE] =  MOMMLE_fluc_exp1(N_t, X_t);
save('E740.mat','obs_X','p_init','MOM','MLE','X_t','N_t');

%%%%%%%%% E1221/94
X_t = [0 0 0 1 1 1 1 1 1 1 2 2 2 2 2 3 3 3 5 5 5 6 1 12 16]';    
N_t = 1.6 * 1e8;                                    
p_init = 6.5e-9;
obs_X = sqrt(sqrt(X_t./N_t));                        
%MOM = 0.5*(1-log(mean(N_t)-mean(X_t))/log(mean(N_t)));
[MOM,MLE] =  MOMMLE_fluc_exp1(N_t, X_t);
save('E1221.mat','obs_X','p_init','MOM','MLE','X_t','N_t');

%%%%%%%%% E1449/94
X_t = [0 1 2 2 3 3 3 3 4 4 4 4 4 5 5 5 6 6 8 8 9 11 14 15]';    
N_t = 1 * 1e8;                                    
p_init = 1.5e-8;
obs_X = sqrt(sqrt(X_t./N_t));                        
%MOM = 0.5*(1-log(mean(N_t)-mean(X_t))/log(mean(N_t)));
[MOM,MLE] =  MOMMLE_fluc_exp1(N_t, X_t);
save('E1449.mat','obs_X','p_init','MOM','MLE','X_t','N_t');

%%%%%%%%% Harlingen
X_t = [1 1 2 2 3 4 4 4 4 5 5 5 5 5 5 5 5 5 6 6 6 6 6 9 14]';    
N_t = 1 * 1e8;                                    
p_init = 1.4e-8;
obs_X = sqrt(sqrt(X_t./N_t));                        
%MOM = 0.5*(1-log(mean(N_t)-mean(X_t))/log(mean(N_t)));
[MOM,MLE] =  MOMMLE_fluc_exp1(N_t, X_t);
save('Harlingen.mat','obs_X','p_init','MOM','MLE','X_t','N_t');

%%%%%%%%%%%%%%%%%%%%%%%%%%% Beijing genotype %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% E26/95
X_t = [0 0 0 0 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 4 6 6 9 10 21]';    
N_t = 0.8 * 1e8;                                    
p_init = 1.3e-8;
obs_X = sqrt(sqrt(X_t./N_t));                        
%MOM = 0.5*(1-log(mean(N_t)-mean(X_t))/log(mean(N_t)));
[MOM,MLE] =  MOMMLE_fluc_exp1(N_t, X_t);
save('E26_95.mat','obs_X','p_init','MOM','MLE','X_t','N_t');

%%%%%%%%% E80/95
X_t = [0 0 1 1 1 1 1 1 2 3 3 3 3 3 3 3 4 4 5 5 5 6 6 6 7]';    
N_t = 1.2 * 1e8;                                    
p_init = 7.9e-9;
obs_X = sqrt(sqrt(X_t./N_t));                        
%MOM = 0.5*(1-log(mean(N_t)-mean(X_t))/log(mean(N_t)));
[MOM,MLE] =  MOMMLE_fluc_exp1(N_t, X_t);
save('E80.mat','obs_X','p_init','MOM','MLE','X_t','N_t');

%%%%%%%%% E55/94
X_t = [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 3 3 3 3 5 6 6]';    
N_t = 1.2 * 1e8;                                    
p_init = 0.8e-8;
obs_X = sqrt(sqrt(X_t./N_t));                        
%MOM = 0.5*(1-log(mean(N_t)-mean(X_t))/log(mean(N_t)));
[MOM,MLE] =  MOMMLE_fluc_exp1(N_t, X_t);
save('E55.mat','obs_X','p_init','MOM','MLE','X_t','N_t');

%%%%%%%%% E26/94
X_t = [0 0 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 4 4 4 4 5 5]';    
N_t = 0.8 * 1e8;                                    
p_init = 9.4e-9;
obs_X = sqrt(sqrt(X_t./N_t));                        
%MOM = 0.5*(1-log(mean(N_t)-mean(X_t))/log(mean(N_t)));
[MOM,MLE] =  MOMMLE_fluc_exp1(N_t, X_t);
save('E26_94.mat','obs_X','p_init','MOM','MLE','X_t','N_t');

%%%%%%%%% E3942/94
X_t = [0 0 1 1 2 2 2 2 2 3 3 3 3 4 5 5 5 6 7 9 9 10 10 10 16]';    
N_t = 0.9 * 1e8;                                    
p_init = 1.5e-8;
obs_X = sqrt(sqrt(X_t./N_t));                        
%MOM = 0.5*(1-log(mean(N_t)-mean(X_t))/log(mean(N_t)));
[MOM,MLE] =  MOMMLE_fluc_exp1(N_t, X_t);
save('E3942.mat','obs_X','p_init','MOM','MLE','X_t','N_t');

%%%%%%%%% E47/94
X_t = [0 0 0 0 0 0 1 1 1 1 1 1 1 2 4 4 4 6 6 6 7 10 11 13 15]';    
N_t = 0.9 * 1e8;                                    
p_init = 1.2e-8;
obs_X = sqrt(sqrt(X_t./N_t));                        
%MOM = 0.5*(1-log(mean(N_t)-mean(X_t))/log(mean(N_t)));
[MOM,MLE] =  MOMMLE_fluc_exp1(N_t, X_t);
save('E47.mat','obs_X','p_init','MOM','MLE','X_t','N_t');

%%%%%%%%%%%%%%%%%% Part 2: two-stage mutation rate estimation %%%%%%%%%%%%%%%
% model specifications:
  clear;
  cd('~/ABC/code_rev');
  addpath('~/ABC/code/kernel')
  addpath('~/ABC/code/lightspeed')
  addpath('~/ABC/code/optim/Matlab')
  
  load('E47.mat');
  theta_sigma = [20 20 20 20];        % sd of prior distribution
  p_range = [-11 -7; -11 -7; 4 18];            % range for the three parameters 
  dt_range = [log10(0.5) log10(2)];

  % model specifications
    model_spec.num_training_theta = [1500; 10];    % Number of settings to train and refine GP model 
    model_spec.ksi = 0.35;                         % The threshold of incorrect decision to refine model
    model_spec.eps = 1e-8;                         % Discrepancy allowed between the true data and simulated data
    model_spec.M = 100;                            % Number of samples drawn from the surrogate model
    model_spec.N = 1000;                          % Total number of samples to be drawn from the posterior
    model_spec.a = 1;                              % Hyperparameter of the geo dist, known, same as the one to generate data
    model_spec.chkt = 20;                          % Plating time, known, same as the one to generate data
    model_spec.Z0 = 1;                              % Number of cultures in one sample, 1 culture per sample in simulation.
    model_spec.constraint = 0;                     % If theta1 smaller than theta2, 1 if yes.
    model_spec.saveSample = 0;
    model_spec.num_rep = 5;                       % The number of replications at each setting 
    model_spec.time_update = 500;                 % The number to update process onto the screen

    %kern_param = [0.01,0.4];
    model_spec.init_param = [1 1 1 1 1]';            % Hyperparameter for GP model, in the order of lengthscale parameters and the nugget parameter 
    model_spec.bounds.upper = [4 4 4 4];             % Upper bounds for the three lengthscale parameters
    model_spec.bounds.lower = [0.1 0.1 0.1 0.1];       % Lower bounds for the three lengthscale parameters

    model_spec.trans_step = [0.4 1 1 1];          % The transition steps for H37Rv

    model_spec.theta_old = [log10(0.8) log10(p_init) log10(p_init) 14];    % t_d = 4  % The initial value of the three parameters

    model_spec.init = 'post/init_2a.mat';

% obtain posterior samples
%POOL = parpool('local',20);
theta_mu = [log10(0.8) log10(p_init) log10(p_init) 14];    % t_d = 4  % The initial value of the three parameters
[sample, runningTime, initTime] = ABC_mu2a(theta_mu, theta_sigma, dt_range, p_range, obs_X, model_spec);
%delete(POOL)

% calculate the acceptance rate.
acc_rate = sum(diff(sample)~=0) / model_spec.N;

save('sample_post_H37Rv.mat','model_spec','theta_mu','theta_sigma','Sample','runningTime','dt_range','p_range');
%%%%%%%%%%%%%%%%%% Part 3: constant mutation rates estimation %%%%%%%%%%%%%%%
% reset the working space and load in data and functions again
clear;
cd('~/ABC/code_rev');
load('E47.mat');
addpath('~/ABC/code/kernel')
addpath('~/ABC/code/lightspeed')
addpath('~/ABC/code/optim/Matlab')

% model specifications
  theta_mu = log10(p_init) % mean of the prior distribution
  theta_sigma = 20;        % sd of prior distribution
  a_dt = 1                 % relative efficiency is 1, no difference between mutant and normal
  p_range = [-11 -7];            % range for the three parameters 

  % model specifications
    model_spec.num_training_theta = [1500; 10];    % Number of settings to train and refine GP model 
    model_spec.ksi = 0.3;                         % The threshold of incorrect decision to refine model
    model_spec.eps = 1e-8;                         % Discrepancy allowed between the true data and simulated data
    model_spec.M = 100;                            % Number of samples drawn from the surrogate model
    model_spec.N = 1000;                          % Total number of samples to be drawn from the posterior
    model_spec.a = 1;                              % Hyperparameter of the geo dist, known, same as the one to generate data
    model_spec.chkt = 20;                          % Plating time, known, same as the one to generate data
    model_spec.Z0 = 1;                              % Number of cultures in one sample, 1 culture per sample in simulation.
    model_spec.constraint = 0;                     % If theta1 smaller than theta2, 1 if yes.
    model_spec.saveSample = 0;
    model_spec.num_rep = 5;                       % The number of replications at each setting 
    model_spec.time_update = 500;                 % The number to update process onto the screen

    %kern_param = [0.01,0.4];
    model_spec.init_param = [1 1]';            % Hyperparameter for GP model, in the order of lengthscale parameters and the nugget parameter 
    model_spec.bounds.upper = 4;             % Upper bounds for the three lengthscale parameters
    model_spec.bounds.lower = 0.1;       % Lower bounds for the three lengthscale parameters

    model_spec.trans_step = 0.4;         % The transition steps of the MCMC algorithm 

    model_spec.theta_old = log10(p_init);    % t_d = 4  % The initial value of the three parameters
    model_spec.saveSample = 0; 
    model_spec.init = 'post/init1.mat';
    
    [sample, runningTime, initTime] = ABC_mu1(theta_mu, theta_sigma, a_dt, p_range, obs_X, model_spec);
    acc_rate = sum(diff(sample)~=0) / model_spec.N;
    save('sample_post_H37Rv_m1.mat','model_spec','theta_mu','theta_sigma','Sample','runningTime','dt_range','p_range');
    %%%%%%%%%%%%%%%%%% Part 4: constant mutation rates with different growth rates estimation %%%%%%%%%%%%%%%
% reset the working space and load in data and functions again
clear;
cd('~/ABC/code_rev');
load('E47.mat');
addpath('~/ABC/code/kernel')
addpath('~/ABC/code/lightspeed')
addpath('~/ABC/code/optim/Matlab')

% model specifications
  theta_mu = [0 log10(p_init)]; % mean of the prior distribution
  theta_sigma = [20 20];        % sd of prior distribution
  dt_range = [log10(0.5) log10(2)];
  p_range = [-11 -7];            % range for the three parameters 

  % model specifications
    model_spec.num_training_theta = [1500; 10];    % Number of settings to train and refine GP model 
    model_spec.ksi = 0.3;                         % The threshold of incorrect decision to refine model
    model_spec.eps = 1e-8;                         % Discrepancy allowed between the true data and simulated data
    model_spec.M = 100;                            % Number of samples drawn from the surrogate model
    model_spec.N = 1000;                          % Total number of samples to be drawn from the posterior
    model_spec.a = 1;                              % Hyperparameter of the geo dist, known, same as the one to generate data
    model_spec.chkt = 20;                          % Plating time, known, same as the one to generate data
    model_spec.Z0 = 1;                              % Number of cultures in one sample, 1 culture per sample in simulation.
    model_spec.constraint = 0;                     % If theta1 smaller than theta2, 1 if yes.
    model_spec.saveSample = 1;
    model_spec.num_rep = 5;                       % The number of replications at each setting 
    model_spec.time_update = 500;                 % The number to update process onto the screen

    %kern_param = [0.01,0.4];
    model_spec.init_param = [1 1 1]';            % Hyperparameter for GP model, in the order of lengthscale parameters and the nugget parameter 
    model_spec.bounds.upper = [4 4];             % Upper bounds for the three lengthscale parameters
    model_spec.bounds.lower = [0.1 0.1];       % Lower bounds for the three lengthscale parameters

    model_spec.trans_step = [0.4 0.8];         % The transition steps of the MCMC algorithm 

    model_spec.theta_old = [0 log10(p_init)];   
    model_spec.saveSample = 1; 
    model_spec.init = 'post/init1a.mat';
    
    [sample, runningTime, initTime] = ABC_mu1a(theta_mu, theta_sigma, dt_range, p_range, obs_X, model_spec);
    acc_rate = sum(diff(sample)~=0) / model_spec.N;
    
    save('sample_post_H37Rv_m1a.mat','model_spec','theta_mu','theta_sigma','sample','runningTime','dt_range','p_range');%%%%%%%%% Part 1: manually input real data and save into .mat file %%%%%%%%%
%%%%%%%%% Part 5: summarize the estimation result and regenerate data %%%%%%%%%
bacName = ["H37Rv" "E865" "E729" "E740" "E1221" "E1449" "Harlingen" "E26_95" "E80" "E55" "E26_94" "E3942" "E47"];
nmc = 25;
chkt = 20;
POOL = parpool('local',10);
for i = 1:length(bacName)
  fprintf([bacName(i)],'\n');
  fprintf('Two-stage model \n');
  burnin = 5000;
  % load in the sample from the two-stage model
  load(strcat("sample_post_",bacName(i),".mat"));
  % obtain sample after burnin
  if i == 1
    sample = Sample;
  end
  N = size(sample,1);
  sample = sample((burnin+1):N,:);
  % calculate posterior mean and simulate data from it
  M = mean(sample);
  a_dt = 10^M(1);
  p1 = 10^ M(2);
  p2 = 10^M(3);
  tj = M(4);
  Nt = NaN(nmc,1);
  Xt = NaN(nmc,1);
  parfor j = 1:nmc
    [Nt_tmp, Xt_tmp] = mut2stage_bMBP_rev(1, 1, a_dt, p1, p2, tj, chkt);
    while Nt_tmp == 0
       [Nt_tmp, Xt_tmp] = mut2stage_bMBP_rev(1, 1, a_dt, p1, p2, tj, chkt);
    end
    Nt(j) = Nt_tmp;
    Xt(j) = Xt_tmp;
  end
  
  save(strcat('model_selection/simu_nmc',num2str(nmc),'_mean_',bacName(i),'.mat'),"M","Nt","Xt");
  % calculate posterior median and simulate data from it
  M = median(sample);
  a_dt = 10^M(1);
  p1 = 10^ M(2);
  p2 = 10^M(3);
  tj = M(4);
  Nt = NaN(nmc,1);
  Xt = NaN(nmc,1);
  parfor j = 1:nmc
    [Nt_tmp, Xt_tmp] = mut2stage_bMBP_rev(1, 1, a_dt, p1, p2, tj, chkt);
    while Nt_tmp == 0
       [Nt_tmp, Xt_tmp] = mut2stage_bMBP_rev(1, 1, a_dt, p1, p2, tj, chkt);
    end
    Nt(j) = Nt_tmp;
    Xt(j) = Xt_tmp;
  end
  save(strcat("model_selection/simu_nmc",num2str(nmc),"_median_",bacName(i),".mat"),"M","Nt","Xt");
  
  fprintf('M1a \n');

  % load in the sample from the constant mutation rate model with different growth rates
  load(strcat("sample_post_",bacName(i),"_m1a.mat"));
  % obtain sample after burnin
  N = size(sample,1);
  sample = sample((burnin+1):N,:);
  % calculate posterior mean and simulate data from it
  M = mean(sample);
  a_dt = 10^M(1);
  p = 10^ M(2);
  Nt = NaN(nmc,1);
  Xt = NaN(nmc,1);
  parfor j = 1:nmc
    [Nt_tmp, Xt_tmp] = mut_bMBP_rev(1, 1, a_dt, p, chkt);
    Nt(j) = Nt_tmp;
    Xt(j) = Xt_tmp;
  end
  
  save(strcat("model_selection/simu_nmc",num2str(nmc),"_mean_",bacName(i),"_m1a.mat"),"M","Nt","Xt");
  % calculate posterior median and simulate data from it
  M = median(sample);
  a_dt = 10^M(1);
  p = 10^ M(2);
  Nt = NaN(nmc,1);
  Xt = NaN(nmc,1);
  parfor j = 1:nmc
    [Nt_tmp, Xt_tmp] = mut_bMBP_rev(1, 1, a_dt, p, chkt);
    Nt(j) = Nt_tmp;
    Xt(j) = Xt_tmp;
  end
  save(strcat("model_selection/simu_nmc",num2str(nmc),"_median_",bacName(i),"_m1a.mat"),"M","Nt","Xt");
  fprintf(['M1 \n']);
  % load in the sample from the constant mutation rate model
  load(strcat("sample_post_",bacName(i),"_m1.mat"));
  % obtain sample after burnin
  N = size(sample,1);
  sample = sample((burnin+1):N,:);
  % calculate posterior mean and simulate data from it
  M = mean(sample);
  p = 10^ M(1);
  Nt = NaN(nmc,1);
  Xt = NaN(nmc,1);
  parfor j = 1:nmc
    [Nt_tmp, Xt_tmp] = mut_bMBP_rev(1, 1, 1, p, chkt);
    Nt(j) = Nt_tmp;
    Xt(j) = Xt_tmp;
  end
  save(strcat("model_selection/simu_nmc",num2str(nmc),"_mean_",bacName(i),"_m1.mat"),"M","Nt","Xt");
  
  % calculate posterior median and simulate data from it
  M = median(sample);
  p = 10^ M(1);
  Nt = NaN(nmc,1);
  Xt = NaN(nmc,1);
  parfor j = 1:nmc
    [Nt_tmp, Xt_tmp] = mut_bMBP_rev(1, 1, 1, p, chkt);
    Nt(j) = Nt_tmp;
    Xt(j) = Xt_tmp;
  end
  save(strcat("model_selection/simu_nmc",num2str(nmc),"_median_",bacName(i),"_m1.mat"),"M","Nt","Xt");
  
end
