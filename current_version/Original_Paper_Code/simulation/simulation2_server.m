% This function runs one replicate of simulation study 2 for GPS-ABC estimator
% 100 datasets have already been generated in simulation2.m file
% Input: seed: the random seed to find the dataset
%        caseNo: =1, p1=1e-8, p2=1e-8; =2, p1=1e-9, p2=5e-8
% repeat calling this function with different setups by varying seed and caseNo on server using the following line:
% (Use swarm to run them simultaneously.)
% matlab -nodisplay -nodesktop  -nosplash -r 'cd ~/ABC_mutation_rate/simulation; simulation2_server(seed, caseNo); exit;'

function H = simulation2_server(seed,caseNo)
  H = 0;
  cd('~/ABC_mutation_rate')
  % This folder contains the functions needed for fitting GP models
  addpath('~/ABC/code/kernel')
  % This folder contains functions for quick matrix operations
  addpath('~/ABC/code/lightspeed')
  % This folder contains functions for optimization, which is for estimating hyperparameters in GP models
  addpath('~/ABC/code/optim/Matlab')
  
  dataPath = '/data/Lu/ABC/data/simu2';
  resultPath = '/data/Lu/ABC/result/simu2';
  
  %%%%%%%%% Step 1: Model specifications for the ABC estimator %%%%%%%%%%%%%%%%%%%
  theta_sigma = [20 20 20 20];        % sd of prior distribution
  p_range = [-11 -7; -11 -7; 4 18];            % range for the three parameters 
  dt_range = [log10(0.5) log10(2)];

  % model specifications
  model_spec.num_training_theta = [1500; 10];    % Number of settings to train and refine GP model 
  model_spec.ksi = 0.35;                         % The threshold of incorrect decision to refine model
  model_spec.eps = 1e-8;                         % Discrepancy allowed between the true data and simulated data
  model_spec.M = 100;                            % Number of samples drawn from the surrogate model
  model_spec.N = 20000;                          % Total number of samples to be drawn from the posterior
  model_spec.a = 1;                              % Hyperparameter of the geo dist, known, same as the one to generate data
  model_spec.chkt = 20;                          % Plating time, known, same as the one to generate data
  model_spec.Z0 = 1;                              % Number of cultures in one sample, 1 culture per sample in simulation.
  model_spec.constraint = 0;                     % If theta1 smaller than theta2, 1 if yes.
  model_spec.saveSample = 1;
  model_spec.num_rep = 5;                       % The number of replications at each setting 
  model_spec.time_update = 500;                 % The number to update process onto the screen

  %kern_param = [0.01,0.4];
  model_spec.init_param = [1 1 1 1 1]';            % Hyperparameter for GP model, in the order of lengthscale parameters and the nugget parameter 
  model_spec.bounds.upper = [4 4 4 4];             % Upper bounds for the three lengthscale parameters
  model_spec.bounds.lower = [0.1 0.1 0.1 0.1];       % Lower bounds for the three lengthscale parameters
  % case 1
  model_spec.trans_step = [0.8 3.5 3.5 4];         % The transition steps of the MCMC algorithm 
  model_spec.theta_old = [log10(0.8) -9 -8 14];    % t_d = 4  % The initial value of the three parameters
  % case 2
  %model_spec.trans_step = [0.6 1.3 1.3 1.8];         % The transition steps of the MCMC algorithm 
  %model_spec.theta_old = [log10(0.8) -9 log10(5e-8) 14];    % t_d = 4  % The initial value of the three parameters

  model_spec.init = 'training/init_2a.mat';
  %%%%%%%%% Step 2: Piece-wise constant mutation rates estimation %%%%%%%%%%%%%%%%%%%

  POOL = parpool('local',10);                  % always keep this as ABC_mu* functions contain parallel computations for fitting GP model.
    load(strcat(dataPath, '/Y',num2str(caseNo),'_',num2str(seed),'.mat'))
    theta_mu = [0 log10(MOM) log10(MOM)+1 14];   % Initial value corresponding to MOM estimator`
    [sample, runningTime, initTime] = ABC_mu2a(theta_mu, theta_sigma, dt_range, p_range, obs_X, model_spec);
    acc_rate = sum(diff(sample)~=0) / model_spec.N;
    % save the sample with its model specification, acceptance rate and running time.
    save(strcat(resultPath, '/sample',num2str(caseNo),'_',num2str(seed),'.mat'), 'model_spec','theta_mu','theta_sigma','sample','runningTime','dt_range','p_range');
  delete(POOL);
  H = 1;