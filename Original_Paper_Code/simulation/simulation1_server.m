% This function runs one replicate of simulation study 1 for GPS-ABC estimator
% 100 datasets have already been generated in simulation1.m file
% Input: seed: the random seed to find the dataset
%        p: 4,5,6,7,or8, is -log10(mutation prob.)
%        Jcase: =1, 10 parallel cultures; =2, 50 cultures; =3, 100 cultures
% repeat calling this function with different setups by varying seed, p and Jcase on server using the following line:
% Use swarm to run them simultaneously.
% matlab -nodisplay -nodesktop  -nosplash -r 'cd ~/ABC_mutation_rate/simulation; simulation1_server(seed,p,Jcase); exit;'

function H = simulation1_server(seed, p, Jcase)
  % including folders containing functions to be used
  % The main folder contains the functions to be directly used
  cd('~/ABC_mutation_rate')
  % This folder contains the functions needed for fitting GP models
  addpath('~/ABC/code/kernel')
  % This folder contains functions for quick matrix operations
  addpath('~/ABC/code/lightspeed')
  % This folder contains functions for optimization, which is for estimating hyperparameters in GP models
  addpath('~/ABC/code/optim/Matlab')
  
  % setup data path
  dataPath = '/data/Lu/ABC/data/simu1';
  % setup result path
  resultPath = '/data/Lu/ABC/result/simu1';
  %%%%%%%%% Step 1: Model specifications for the ABC estimator %%%%%%%%%%%%%%%%%%%
  theta_sigma = 20;        % sd of prior distribution
  if p == 8
    p_range = [-10 -6];
    model_spec.trans_step = 0.35;
  elseif p == 7
    model_spec.trans_step = 0.3;
    p_range = [-9 -5];
  elseif p == 6
    model_spec.trans_step = 0.3;
    p_range = [-8 -4];
  elseif p == 5
    model_spec.trans_step = 0.25;
    p_range = [-6 -4];
  else
    model_spec.trans_step = 0.20;
    p_range = [-5 -3];
  end
  % model specifications
  model_spec.num_training_theta = [1500; 10];    % Number of settings to train and refine GP model 
  model_spec.ksi = 0.25;                         % The threshold of incorrect decision to refine model
  model_spec.eps = 0;                         % Discrepancy allowed between the true data and simulated data
  model_spec.M = 100;                            % Number of samples drawn from the surrogate model
  model_spec.N = 30000;                          % Total number of samples to be drawn from the posterior
  model_spec.a = 1;                              % Hyperparameter of the geo dist, known, same as the one to generate data
  model_spec.Z0 = 1;                              % Number of cultures in one sample, 1 culture per sample in simulation.
  model_spec.saveSample = 0;
  model_spec.num_rep = 10;                       % The number of replications at each setting 
  model_spec.time_update = 500;                 % The number to update process onto the screen
  model_spec.constraint = 0;
  %kern_param = [0.01,0.4];
  model_spec.init_param = [1 1]';            % Hyperparameter for GP model, in the order of lengthscale parameters and the nugget parameter 
  model_spec.bounds.upper = 4;             % Upper bounds for the three lengthscale parameters
  model_spec.bounds.lower = 0.1;       % Lower bounds for the three lengthscale parameters
  
  %model_spec.trans_step = 0.35;         % The transition steps of the MCMC algorithm 
  model_spec.theta_old = -p;    % t_d = 4  % The initial value of the three parameters
    
  model_spec.init = strcat('training/simu1/init1_p', num2str(p), '.mat');
%%%%%%%%% Step 2: Piece-wise constant mutation rates estimation %%%%%%%%%%%%%%%%%%%

POOL = parpool('local',10);                  % always keep this as ABC_mu* functions contain parallel computations for fitting GP model

load(strcat(dataPath, '/p',num2str(p), ...
    '/Y',num2str(Jcase),'_',num2str(seed),'.mat'))
model_spec.chkt = tp;
theta_mu = log10(MOM);   % Initial value corresponding to MOM estimator`
[sample, runningTime, initTime] = ABC_mu1(theta_mu, theta_sigma, 1, p_range, sqrt(obs_X), model_spec);
acc_rate = sum(diff(sample(:,1))~=0) / model_spec.N;
% save the sample with its model specification, acceptance rate and running time.
% each mutation rate has its own folder to save the result under resultPath.
save(strcat(resultPath,'/p',num2str(p),'/sample', num2str(Jcase),...
      '_',num2str(seed),'.mat'), 'model_spec','theta_mu','theta_sigma','sample','runningTime','p_range');
delete(POOL);
H = 1;
