%%%%%%%%%%%%%%%%%%%%%%%%% Part 1: Generate data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%% Step 1: Set the hyperparameters %%%%%%%%%%%%%%%%%%%
  % Exponential rate parameter a, typically 1 
  a = 1;
  a_dt = 0.8; % the mutated cells are resistant to antibiotics and live longer
  % Two-stage mutations rates
  mu_vec = 10.^-7 .* [0.01 0.1]; % case 1
  mu_vec = 10.^-7 .* [0.01 0.5]; % case 2
  t_d = 14;
  % Plating time:
  chkt = 20;
  % Total number of cultures
  L = 1;
  nmc = 100;
  % setup data path and result path
  dataPath = '/data/Lu/ABC/data/simu2';
  resultPath = '/data/Lu/ABC/result/simu2';
%%%%%%%%%%%%%%%%%%%%%%%%% Step 2: Generate fluctuation data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Get multiple local workers
  POOL = parpool('local',20);
  nSimu = 100;
% Set seeds
  for seed = 1:nSimu;
    startTime = tic;
    s = RandStream('mt19937ar','Seed',seed);
    RandStream.setGlobalStream(s);
  % Initialize number of total cells and number of mutants
    Nt = NaN(nmc,1); 
    Xt = NaN(nmc,1);
  % Initialize the vector for saving observed data: the statistics
    obs_X = NaN(nmc,1);

    parfor j = 1 : nmc
      if mod(j, 5) == 0
        fprintf(['j=', int2str(j),'\n']); % print  the process
      end
      [tmp1, tmp2] = mut2stage_bMBP_rev(L, a, a_dt, mu_vec(1), mu_vec(2), t_d, chkt);
      Nt(j,1) = tmp1;%%
      Xt(j,1) = tmp2;
      obs_X(j,1) = sqrt(sqrt(tmp2/tmp1));%% Statistic is the 4th root of mutant proportion
    end
% calculate the moment-based estimator
    MOM = 0.5*(1-log(mean(Nt)-mean(Xt))/log(mean(Nt)));
% save simulated data, including Nt, Xt, obs_X and MOM
    duration = toc(startTime);
    save(strcat(dataPath,'/Y2_',num2str(seed),'.mat'),'Nt','Xt','obs_X','MOM','duration');
  end
  delete(POOL);

%%%%%%%%%%%%%%%%%% Part 2: Estimating piecewise constant mutation rates %%%%%%%%%%%%%%%%
  % including folders containing functions to be used
% The main folder contains the functions to be directly used
clear;
cd('~/ABC/code_rev')
% This folder contains the functions needed for fitting GP models
addpath('~/ABC/code/kernel')
% This folder contains functions for quick matrix operations
addpath('~/ABC/code/lightspeed')
% This folder contains functions for optimization, which is for estimating hyperparameters in GP models
addpath('~/ABC/code/optim/Matlab')

%%%%%%%%% Step 1: Model specifications for the ABC estimator %%%%%%%%%%%%%%%%%%%
  load('~/ABC/simulation/data/varyP_varyA/post/Y1_98.mat')
  theta_sigma = [20 20 20 20];        % sd of prior distribution
  p_range = [-11 -7; -11 -7; 4 18];            % range for the three parameters 
  dt_range = [log10(0.5) log10(2)];

  % model specifications
    model_spec.num_training_theta = [1500; 10];    % Number of settings to train and refine GP model 
    model_spec.ksi = 0.35;                         % The threshold of incorrect decision to refine model
    model_spec.eps = 1e-8;                         % Discrepancy allowed between the true data and simulated data
    model_spec.M = 100;                            % Number of samples drawn from the surrogate model
    model_spec.N = 3000;                          % Total number of samples to be drawn from the posterior
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
    %model_spec.trans_step = [0.8 3.5 3.5 4];         % The transition steps of the MCMC algorithm 
    %model_spec.theta_old = [log10(0.8) -9 -8 14];    % t_d = 4  % The initial value of the three parameters
    % case 2
    model_spec.trans_step = [0.6 1.3 1.3 1.8];         % The transition steps of the MCMC algorithm 
    model_spec.theta_old = [log10(0.8) -9 log10(5e-8) 14];    % t_d = 4  % The initial value of the three parameters
    
    model_spec.init = 'training/init_2a.mat';
%%%%%%%%% Step 2: Piece-wise constant mutation rates estimation %%%%%%%%%%%%%%%%%%%

%POOL = parpool('local',20);                  % always keep this as ABC_mu* functions contain parallel computations for fitting GP model.
for seed = 2:100                             % change seed range to run less cases, each takes around 40 minutes.
    load(strcat(dataPath, '/Y2_',num2str(seed),'.mat'))
    theta_mu = [0 log10(MOM) log10(MOM)+1 14];   % Initial value corresponding to MOM estimator`
    [sample, runningTime, initTime] = ABC_mu2a(theta_mu, theta_sigma, dt_range, p_range, obs_X, model_spec);
    acc_rate = sum(diff(sample)~=0) / model_spec.N;
    % save the sample with its model specification, acceptance rate and running time.
    save(strcat(resultPath,'/sample1_',num2str(seed),'.mat'), 'model_spec','theta_mu','theta_sigma','sample','runningTime','dt_range','p_range');
end
delete(POOL);
