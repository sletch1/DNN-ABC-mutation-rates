%%%%%%%%%%%%%%%%%%%%%%%%% Part 1: Generate data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Step 1: Set the hyperparameters %%%%%%%%%%%%%%%%%%%
% Exponential rate parameter a, typically 1 
a = 1;
% Two-stage mutations rates
mu_vec = 10.^[-5 -3];
% Traistion time:
t_d = 4; 
% Plating time:
chkt = 11;
% Total number of cultures
nmc = 100;

%%%%%%%%%%%%%%%%%%%%%%%%% Step 2: Generate fluctuation data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get multiple local workers
POOL = parpool('local',20);
% Set seeds
for seed = 1:100;
s = RandStream('mt19937ar','Seed',seed);
    RandStream.setGlobalStream(s);
% Initialize number of total cells and number of mutants
Nt = NaN(nmc,1); 
Xt = NaN(nmc,1);
% Initialize the vector for saving observed data: the statistics
obs_X = NaN(nmc,1);

parfor j = 1 : nmc
    [temp1, temp2] = countsizeBDtree2(a, mu_vec,t_d, chkt);%%
    Nt(j,1) = temp1;%%
    Xt(j,1) = temp2;
    obs_X(j,1) = sqrt(sqrt(temp2./temp1));%% Statistic is the 4th root of mutant proportion
end
% calculate the moment-based estimator
MOM = 0.5*(1-log(mean(Nt)-mean(Xt))/log(mean(Nt)));
% save simulated data, including Nt, Xt, obs_X and MOM
save(strcat('~/ABC/simulation/data/t4/nmc100/Y',num2str(seed),'.mat'),'Nt','Xt','obs_X','MOM');
end
delete(POOL);

%%%%%%%%%%%%%%%%%% Part 2: Estimating piecewise constant mutation rates %%%%%%%%%%%%%%%%
% including folders containing functions to be used
% The main folder contains the functions to be directly used
cd('~/ABC/code')
% This folder contains the functions needed for fitting GP models
addpath('~/ABC/code/kernel')
% This folder contains functions for quick matrix operations
addpath('~/ABC/code/lightspeed')
% This folder contains functions for optimization, which is for estimating hyperparameters in GP models
addpath('~/ABC/code/optim/Matlab')

%%%%%%%%% Step 1: Model specifications for the ABC estimator %%%%%%%%%%%%%%%%%%%

theta_sigma = [20 20 100];                     % sd of prior distribution
param_range = [-9,-1;-9,-1; 0, 11];            % range for the three parameters 

% model specifications
model_spec.num_training_theta = [1000; 10];    % Number of settings to train and refine GP model 
model_spec.ksi = 0.25;                         % The threshold of incorrect decision to refine model
model_spec.eps = 1e-8;                         % Discrepancy allowed between the true data and simulated data
model_spec.M = 100;                            % Number of samples drawn from the surrogate model
model_spec.N = 50000;                          % Total number of samples to be drawn from the posterior
model_spec.a = 1;                              % Hyperparameter of the geo dist, known, same as the one to generate data
model_spec.chkt = 11;                          % Plating time, known, same as the one to generate data
model_spec.L = 1;                              % Number of cultures in one sample, 1 culture per sample in simulation.
model_spec.constraint = 1;                     % If theta1 smaller than theta2, 1 if yes.

model_spec.num_rep = 30;                       % The number of replications at each setting 
model_spec.time_update = 1000;                 % The number to update process onto the screen

%kern_param = [0.01,0.4];
model_spec.init_param = [6 6 1 1]';            % Hyperparameter for GP model, in the order of lengthscale parameters and the nugget parameter 
model_spec.bounds.upper = [8 8 2];             % Upper bounds for the three lengthscale parameters
model_spec.bounds.lower = [0.1 0.1 0.1];       % Lower bounds for the three lengthscale parameters

model_spec.model = 2;                          % which model to be used: 2 is the piece-wise constant model
model_spec.trans_step = [0.3 0.3 0.4];         % The transition steps of the MCMC algorithm 
model_spec.theta_old = [-4 -3 4.5]; % t_d = 4  % The initial value of the three parameters

%%%%%%%%% Step 2: Piece-wise constant mutation rates estimation %%%%%%%%%%%%%%%%%%%

POOL = parpool('local',20);                  % always keep this as ABC_mu2 function contains parallel computations for fitting GP model.
for seed = 1:100                             % change seed range to run less cases, each takes around 40 minutes.
    load(strcat('~/ABC/simulation/data/t4/nmc100/Y',num2str(seed),'.mat'))
    theta_mu = [log10(MOM)-1 log10(MOM)+1 4.5];   % Initial value corresponding to MOM estimator`
    tStart = tic;
    sample = ABC_mu2(theta_mu, theta_sigma, param_range,obs_X, model_spec);
    tEnd = toc(tStart);
    acc_rate = sum(diff(sample)~=0) / model_spec.N;
    % save the sample with its model specification, acceptance rate and running time.
    save(strcat('~/ABC/simulation/results/t4/sample',num2str(seed),'_nmc100.mat'), 'sample','acc_rate','model_spec','tEnd', 'param_range');
end
delete(POOL);
