% This function runs one replicate of simulation study 1 for ABC-MCMC estimator
% 100 datasets have already been generated in simulation1.m file
% Input: seed: the random seed to find the dataset
%        p: 4,5,6,7,or8, is -log10(mutation prob.)
%        Jcase: =1, 10 parallel cultures; =2, 50 cultures; =3, 100 cultures
% repeat calling this function with different setups by varying seed, p and Jcase on server using the following line:
% Use swarm to run them simultaneously.
% matlab -nodisplay -nodesktop  -nosplash -r 'cd ~/ABC_mutation_rate/simulation; simulation1_MCMC_server(seed,p,Jcase); exit;'

function H = simulation1_MCMC_server(seed, p, Jcase)
  % including folders containing functions to be used
  % The main folder contains the functions to be directly used
  cd('~/ABC_mutation_rate')
  % This folder contains the functions needed for fitting GP models
  addpath('~/ABC/code/kernel')
  % This folder contains functions for quick matrix operations
  addpath('~/ABC/code/lightspeed')
  % This folder contains functions for optimization, which is for estimating hyperparameters in GP models
  addpath('~/ABC/code/optim/Matlab')

  %%%%%%%%% Step 1: Model specifications for the ABC estimator %%%%%%%%%%%%%%%%%%%
  theta_sigma = 20;        % sd of prior distribution
  if p == 8
    p_range = [-10 -6];
    model_spec.trans_step = 0.20;
  elseif p == 7
    model_spec.trans_step = 0.2;
    p_range = [-9 -5];
  elseif p == 6
    model_spec.trans_step = 0.2;
    p_range = [-8 -4];
  elseif p == 5
    model_spec.trans_step = 0.2;
    p_range = [-6 -4];
  else
    model_spec.trans_step = 0.2;
    p_range = [-5 -3];
  end
  % model specifications
  model_spec.eps = 0;                         % Discrepancy allowed between the true data and simulated data
  model_spec.M = 100;                            % Number of samples drawn from the surrogate model
  model_spec.N = 30000;                          % Total number of samples to be drawn from the posterior
  model_spec.a = 1;                              % Hyperparameter of the geo dist, known, same as the one to generate data
  model_spec.Z0 = 1;                              % Number of cultures in one sample, 1 culture per sample in simulation.
  model_spec.num_rep = 10;                       % The number of replications at each setting 
  model_spec.time_update = 500;                 % The number to update process onto the screen
  
  %model_spec.trans_step = 0.35;         % The transition steps of the MCMC algorithm 
  model_spec.theta_old = -p;    % t_d = 4  % The initial value of the three parameters
  %%%%%%% Step 2: ABC-MCMC for constant mutation rate estimation%%%%%%%%%%%%%%%%

POOL = parpool('local',20);                  % always keep this as ABC_mu* functions contain parallel computations for fitting GP model

load(strcat('/data/ZChenLab/Lu/ABC/data/simu1/p',num2str(p), ...
    '/Y',num2str(Jcase),'_',num2str(seed),'.mat'))
model_spec.chkt = tp;
theta_mu = log10(MOM);   % Initial value corresponding to MOM estimator`
[sample, runningTime, nacc] = ABC_MCMC(theta_mu, theta_sigma, 1, p_range, sqrt(obs_X), model_spec);
acc_rate = nacc / model_spec.N;
% save the sample with its model specification, acceptance rate and running time.
save(strcat('/data/ZChenLab/Lu/ABC/result/simu1/MCMC/p',num2str(p),'/sample', num2str(Jcase),...
      '_',num2str(seed),'.mat'), 'model_spec','theta_mu','theta_sigma',...
      'acc_rate', 'sample','runningTime','p_range');
delete(POOL);
H = 1;
