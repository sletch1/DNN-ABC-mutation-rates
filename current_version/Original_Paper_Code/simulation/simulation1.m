%%%%%%%%%%%%%%%%%%%%%%%%% Part 1: Generate data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%% Step 1: Set the hyperparameters %%%%%%%%%%%%%%%%%%%
  % Exponential rate parameter a, typically 1 
    a = 1; % may change a to other values
    p = 1e-4; % change it to other values such as 1e-5,1e-6, 1e-7 and 1e-8.
    c = 100;
    Z0 = 1; % may change Z0 to other values
    % Calculate the plating time for the current setting to obtain the expected 
    % number of mutants as 20.
    myfun = @(t, Z0, a, p, c) Z0 * (exp(a * t) - exp(a * t * (1 - 2 * p))) - c;
    fun = @(t) myfun(t, Z0, a, p, c);
    tp = fzero(fun, 20);
    L = 1;
    nmcs=[10 50 100];
%%%%%%%%%%%%%%%%%%%%%%%%% Step 2: Generate fluctuation data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the current directory to the folder containing simulators and esimators.
cd('~/ABC_mutation_rate');
% Set the directory for saving data and estimation results.
dataPath = '/data/Lu/ABC/data/simu1';
resultPath = '/data/Lu/ABC/result/simu1';

  % Get multiple local workers
POOL = parpool('local',10);
nSimu = 100;
% Set seeds
for seed = 1:nSimu;
  if mod(seed, 5) == 0
        fprintf(['seed=', int2str(seed),'\n']); % print  the process
      end
  for k = 1:3
    nmc = nmcs(k);
    startTime = tic;
    s = RandStream('mt19937ar','Seed',seed);
    RandStream.setGlobalStream(s);
    % Initialize number of total cells and number of mutants
    Nt = NaN(nmc,1); 
    Xt = NaN(nmc,1);
    % Initialize the vector for saving observed data: the statistics
    obs_X = NaN(nmc,1);

    parfor j = 1 : nmc
      [tmp1, tmp2] = mut_bMBP_rev(Z0, a, 1, p, tp);
      Nt(j,1) = tmp1;%%
      Xt(j,1) = tmp2;
      obs_X(j,1) = sqrt(tmp2/tmp1);%% Statistic is the 4th root of mutant proportion
    end
    % calculate the moment-based estimator
    [MOM MLE] = MOMMLE_fluc_exp1(Nt, Xt);
    
    % save simulated data, including Nt, Xt, obs_X and MOM
    % Each mutation rate has its own folder to save the data
    % J = 10, 50 and 100 corresponds to Y1, Y2 and Y3, respectively.
    % Yk_i(k = 1,2,3) is the i th Yk generated from seed i, here we have i = 1:100
    duration = toc(startTime);
      save(strcat(dataPath,'/p4/Y',num2str(k),'_', ...
      num2str(seed),'.mat'),'Nt','Xt','obs_X','MOM','MLE','tp','duration');
  end
end
delete(POOL);

%%%%%%%%%%%%%%%%%% Part 2: Constant mutation rate estimation %%%%%%%%%%%%%%%%
%%%%% Use these codes to run a short chain for determining the stepwidth.
%%%%% And save the initial training set for other replicates under the same setting.
  % including folders containing functions to be used
% The main folder contains the functions to be directly used
clear;
cd('~/ABC_mutation_rate');
% This folder contains the functions needed for fitting GP models
addpath('~/ABC/code/kernel')
% This folder contains functions for quick matrix operations
addpath('~/ABC/code/lightspeed')
% This folder contains functions for optimization, which is for estimating hyperparameters in GP models
addpath('~/ABC/code/optim/Matlab')

%%%%%%%%% Step 1: Model specifications for the ABC estimator %%%%%%%%%%%%%%%%%%%
theta_sigma = 20;        % sd of prior distribution
p_range = [-10 -6];

% model specifications
model_spec.num_training_theta = [1500; 10];    % Number of settings to train and refine GP model 
model_spec.ksi = 0.25;                         % The threshold of incorrect decision to refine model
model_spec.eps = 0;                         % Discrepancy allowed between the true data and simulated data
model_spec.M = 100;                            % Number of samples drawn from the surrogate model
model_spec.N = 3000;                          % Total number of samples to be drawn from the posterior
model_spec.a = 1;                              % Hyperparameter of the geo dist, known, same as the one to generate data
model_spec.Z0 = 1;                              % Number of cultures in one sample, 1 culture per sample in simulation.
model_spec.saveSample = 1;
model_sepc.saveSamplePath = strcat('training/simu1/init1_p',num2str(-log10(p)),'.mat');
model_spec.num_rep = 10;                       % The number of replications at each setting 
model_spec.time_update = 500;                 % The number to update process onto the screen
model_spec.constraint = 0;
%kern_param = [0.01,0.4];
    model_spec.init_param = [1 1]';            % Hyperparameter for GP model, in the order of lengthscale parameters and the nugget parameter 
    model_spec.bounds.upper = 4;             % Upper bounds for the three lengthscale parameters
    model_spec.bounds.lower = 0.1;       % Lower bounds for the three lengthscale parameters
    % case 1
    %model_spec.trans_step = [0.8 3.5 3.5 4];         % The transition steps of the MCMC algorithm 
    %model_spec.theta_old = [log10(0.8) -9 -8 14];    % t_d = 4  % The initial value of the three parameters
    % case 2
    model_spec.trans_step = 0.35;         % The transition steps of the MCMC algorithm 
    model_spec.theta_old = -8;    % t_d = 4  % The initial value of the three parameters
    
    model_spec.init = 0;
%%%%%%%%% Step 2: Piece-wise constant mutation rates estimation %%%%%%%%%%%%%%%%%%%

POOL = parpool('local',10);                  % always keep this as ABC_mu* functions contain parallel computations for fitting GP model.
%for seed = 2:100                             % change seed range to run less cases, each takes around 40 minutes.
    seed = 1;
     % Y1 corresponds to 10 parallel cultures, change to Y2 for 50 cultures or to Y3 for 100 cultures
    load(strcat(dataPath,'/p',num2str(-log10(p)),'/Y1_',num2str(seed),'.mat'))
    model_spec.chkt = tp;
    theta_mu = log10(MOM);   % Initial value corresponding to MOM estimator`
    [sample, runningTime, initTime] = ABC_mu1(theta_mu, theta_sigma, 1, p_range, sqrt(obs_X), model_spec);
    % If acc_rate is not smaller than 20%, shorten the step width; If it is larger than 40%, increase the step width.
    acc_rate = sum(diff(sample(:,1))~=0) / model_spec.N;
    % save the sample with its model specification, acceptance rate and running time.
    save(strcat(resultPath,'/p',num2str(-log10(p)),'/sample1_',num2str(seed),'.mat'), 'model_spec','theta_mu','theta_sigma','sample','runningTime','dt_range','p_range');
%end
delete(POOL);
