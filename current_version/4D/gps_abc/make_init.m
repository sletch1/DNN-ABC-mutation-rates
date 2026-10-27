% One-off: build the GPS-ABC initial training set on the PAPER box (1,500 Latin-hypercube points x 5 reps)
% with the authors' own getIniX2a / mleHomGP, exactly as ABC_mu2a does when model_spec.init == 0, and save it
% for all replicates (the authors also share one init file across replicates).
%   matlab -batch "make_init(10)"
function make_init(nworkers)
  addpath(genpath('../../Original_Paper_Code/kernel')); addpath(genpath('../../Original_Paper_Code/lightspeed'));
  addpath(genpath('../../Original_Paper_Code/optim/Matlab')); addpath('../../Original_Paper_Code'); addpath(fullfile(pwd,'patched'));   % patched copy must come FIRST on the path
  c = parcluster('local'); jd = tempname; mkdir(jd); c.JobStorageLocation = jd; parpool(c, nworkers);
  rng(1);
  a = 1; dt_range = [log10(0.5) log10(2)];
  % paper box, THETA ORDER OF THE CODE: (log10 delta, log10 p1, log10 p2, tau)
  p_range = [-11 -7; -11 -7; 0.1 19.9]; param_range = [dt_range; p_range];
  init_param = [1 1 1 1 1]'; bounds.upper = [4 4 4 4]; bounds.lower = [0.1 0.1 0.1 0.1];
  tic; [theta_list, X] = getIniX2a(a, dt_range, p_range, 20, 1500, 5, 500, 1, 0);
  theta_list_s = (theta_list - repmat(param_range(:,1)',size(theta_list,1),1)) ./ repmat((param_range(:,2)-param_range(:,1))',size(theta_list,1),1);
  [param, model] = mleHomGP(theta_list_s, mean(X,3), init_param, bounds);
  [param_sd, model_sd] = mleHomGP(theta_list_s, std(X,[],3), init_param, bounds);
  initTime = toc;
  save('init_2a_paperbox.mat','theta_list','theta_list_s','X','initTime','param','model','param_sd','model_sd');
  fprintf('init done in %.1f s\n', initTime); delete(gcp('nocreate'));
end
