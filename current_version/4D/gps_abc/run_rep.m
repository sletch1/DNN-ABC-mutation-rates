% One GPS-ABC replicate, the authors' ABC_mu2a on one of OUR observed datasets, with the PAPER's settings
% (xi = 0.3, tau box [0.1,19.9], priors TN(log10 MOM,20,.) x2, TN(10,20,.), TN(0,20,.)); everything the paper
% does not state is taken from simulation2_server.m. Theta order is the code's: (log10 delta, log10 p1, log10 p2, tau).
%   matlab -batch "run_rep(1, 7, 10, 20000)"      % case, seed(dataset index), workers, N
function run_rep(caseNo, seed, nworkers, N)
  maxNumCompThreads(2);   % GP refits are single-threaded; run many replicates side by side instead
  here = pwd; root = fullfile(here, '../../Original_Paper_Code');
  addpath(genpath([root '/kernel'])); addpath(genpath([root '/lightspeed'])); addpath(genpath([root '/optim/Matlab'])); addpath(root); addpath(fullfile(here,'patched'));
  c = parcluster('local'); jd = tempname; mkdir(jd); c.JobStorageLocation = jd; parpool(c, nworkers);
  rng(1000*caseNo + seed);
  S = load(sprintf('../data/mat/obs_4d_case%d_%d.mat', caseNo, seed)); obs_X = S.obs_X; MOM = S.MOM;
  theta_sigma = [20 20 20 20];
  p_range = [-11 -7; -11 -7; 0.1 19.9]; dt_range = [log10(0.5) log10(2)];
  ms.num_training_theta = [1500; 10]; ms.ksi = 0.3; ms.eps = 1e-8; ms.M = 100; ms.N = N; ms.a = 1; ms.chkt = 20; ms.Z0 = 1;
  ms.constraint = 0; ms.saveSample = 0; ms.num_rep = 5; ms.time_update = 250;
  ms.init_param = [1 1 1 1 1]'; ms.bounds.upper = [4 4 4 4]; ms.bounds.lower = [0.1 0.1 0.1 0.1];
  ms.trans_step = [0.8 3.5 3.5 4]; ms.theta_old = [log10(0.8) -9 -8 14];
  ms.init = fullfile(here, 'init_2a_paperbox.mat');
  if ~exist('out','dir'), mkdir('out'); end
  ms.ckpt = fullfile(here, sprintf('out/ckpt_case%d_%d.mat', caseNo, seed));
  lm = log10(max(MOM, 1e-11)); theta_mu = [0 lm lm 10];
  [sample, runningTime, initTime] = ABC_mu2a(theta_mu, theta_sigma, dt_range, p_range, obs_X, ms);
  acc_rate = sum(diff(sample) ~= 0) / ms.N; %#ok<NASGU>
  if ~exist('out','dir'), mkdir('out'); end
  save(sprintf('out/gps4d_case%d_%d.mat', caseNo, seed), 'sample','runningTime','theta_mu','ms','acc_rate');
  fprintf('DONE case %d seed %d: %.1f s, acc %.3f\n', caseNo, seed, runningTime, acc_rate); delete(gcp('nocreate'));
end
