function H = werngren_server2a(dataName,trans_step,initValue, a, N)
  H = 0;
  cd('~/ABC/code_rev')
  % This folder contains the functions needed for fitting GP models
  addpath('~/ABC/code/kernel')
  % This folder contains functions for quick matrix operations
  addpath('~/ABC/code/lightspeed')
  % This folder contains functions for optimization, which is for estimating hyperparameters in GP models
  addpath('~/ABC/code/optim/Matlab')
  
  % Load data
  load(strcat(dataName,'.mat'))
  if initValue == 1
    p_init = MOM;
  end
  
  tp = log(N_t) / a;
  theta_sigma = [20 20 20 20];        % sd of prior distribution
  p_range = [-11 -7; -11 -7; 0.1*tp 0.9*tp];            % range for the three parameters 
  dt_range = [log10(0.5) log10(2)];
  
  % model specifications
  model_spec.num_training_theta = [1500; 10];    % Number of settings to train and refine GP model 
  model_spec.ksi = 0.35;                         % The threshold of incorrect decision to refine model
  model_spec.eps = 1e-8;                         % Discrepancy allowed between the true data and simulated data
  model_spec.M = 100;                            % Number of samples drawn from the surrogate model
  model_spec.N = N;                          % Total number of samples to be drawn from the posterior
  model_spec.a = a;                              % Hyperparameter of the geo dist, known, same as the one to generate data
  model_spec.chkt = tp;                          % Plating time, known, same as the one to generate data
  model_spec.Z0 = 1;                              % Number of cultures in one sample, 1 culture per sample in simulation.
  model_spec.constraint = 0;                     % If theta1 smaller than theta2, 1 if yes.
  model_spec.saveSample = 0;
  model_spec.num_rep = 5;                       % The number of replications at each setting 
  model_spec.time_update = 500;                 % The number to update process onto the screen
  
  %kern_param = [0.01,0.4];
  model_spec.init_param = [1 1 1 1 1]';            % Hyperparameter for GP model, in the order of lengthscale parameters and the nugget parameter 
  model_spec.bounds.upper = [4 4 4 4];             % Upper bounds for the three lengthscale parameters
  model_spec.bounds.lower = [0.1 0.1 0.1 0.1];       % Lower bounds for the three lengthscale parameters

  model_spec.trans_step = trans_step;          % The transition steps for H37Rv

  model_spec.theta_old = [log10(0.8) log10(p_init) log10(p_init) tp/2];    % t_d = 4  % The initial value of the three parameters

  model_spec.init = 0;

  % obtain posterior samples
  POOL = parpool(10);
  theta_mu = [log10(0.8) log10(p_init) log10(p_init) tp/2];    % t_d = 4  % The initial value of the three parameters
  [sample, runningTime, initTime] = ABC_mu2a(theta_mu, theta_sigma, dt_range, p_range, obs_X, model_spec);
  delete(POOL)

  % calculate the acceptance rate.
  acc_rate = sum(diff(sample)~=0) / model_spec.N;

  save(strcat('/data/ZChenLab/Lu/ABC/werngren/sample_a',num2str(a),"_",dataName,'_',num2str(initValue),'.mat'),'model_spec','acc_rate','theta_mu','theta_sigma',...
  'sample','MOM','MLE','runningTime','dt_range','p_range');
  H = 1;