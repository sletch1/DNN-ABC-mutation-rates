%% This function is for generating MCMC posterior samples of 
%% piece-wise constant mutation rates and changing time point based on bMBP model
% Input: theta_mu: prior for theta, log scale
%        theta_sigma: prior variance for theta, 0 if parameter is not
%                     estimated in this case.
%        param_range; a structure storing range of each parameter
%        obs_X: observed data
%        model_spec: model specifications, including: 
  %                     num_training_theta: vector of 2, the number of training data points 
%                                         for initial training and additional trainings.
%                     eps: accuracy threshold, 
%                     psi: error threshold
%                     N: sample size
%                     chkt: time checking point
%                     a: parameter for birth-death process
%                     num_rep: number of replicates at each training point
%                     time_update: for showing progress
%                     init_param: initial values of GP's hyperparameters.
%                     theta_old: the initial values of parameters to be estimated.
%                     model: the model version, "2" indicates the model with piece-wise constant rates.
%                     bounds: bounds for the parameters to be estimated
%                     trans_step: stepwidth for proposal distribution
function [Sample, runningTime, initTime] = ABC_mu1a(theta_mu, theta_sigma, dt_range, p_range, obs_X, model_spec)

    S0 = model_spec.num_training_theta(1,:);
    delta = model_spec.num_training_theta(2,:);
    p = length(theta_mu);   %length(theta_mu)

    ksi = model_spec.ksi;
    eps = model_spec.eps ;
    M = model_spec.M;
    N = model_spec.N;
    a = model_spec.a;
    chkt = model_spec.chkt;
    Z0 = model_spec.Z0;
    constraint = model_spec.constraint;

    num_rep = model_spec.num_rep;
    time_update = model_spec.time_update;

    %kern_param = [0.01,0.4];
    init_param = model_spec.init_param;
    bounds = model_spec.bounds;

    %trans_step = [0.08,0.08,0.08];
    trans_step = model_spec.trans_step;
    param_range = [dt_range; p_range];
    
    Y_m = mean(obs_X);
    Y_sd = std(obs_X);
    Feat = [Y_m, Y_sd];
    
    if model_spec.init == 0
      startTime = tic;
      [theta_list, X] = getIniX1a(a, dt_range,p_range,chkt,S0, num_rep,time_update, Z0);
      theta_list_s = (theta_list - repmat(param_range(:,1)',size(theta_list,1),1)) ./...
              repmat((param_range(:,2) - param_range(:,1))',size(theta_list,1),1);
      [param, model] = mleHomGP(theta_list_s,mean(X,3),init_param,bounds);
      [param_sd, model_sd] = mleHomGP(theta_list_s, std(X,[],3), init_param,bounds);
      initTime = toc(startTime);
      if model_spec.saveSample == 1
        save('post/init1a.mat','theta_list', 'theta_list_s', "X", "initTime",'param','model','param_sd','model_sd');
      end
    else
      load(model_spec.init);
      initTime == 0.0;
    end
    
    J = size(Feat,2); % number of the features
    
    theta_old = model_spec.theta_old;

    Sample = NaN(N,p);
    
    startTime = tic;
    %h = waitbar(0,'Please wait...');
    for ss =  1:N
        if mod(ss, time_update) == 0 % screen print to show progress.
            fprintf(['ss=', int2str(ss),'\n']);
        end
    [theta_new,dt_range_new,p_range_new] = get_theta1a(trans_step,theta_old,dt_range,p_range);
    theta = [theta_new;theta_old];
    theta_s = (theta - repmat(param_range(:,1)',2,1)) ./...
              repmat((param_range(:,2) - param_range(:,1))',2,1);
    
    for iter = 1:5000
        alpha = NaN(M,J);  
       for j = 1:J
            if j == 1
                [Mu_cond, Var_cond,nugs] = GPHomPrediction(theta_s,model,param);
            elseif j == 2
                [Mu_cond, Var_cond,nugs] = GPHomPrediction(theta_s,model_sd,param_sd);
            end
            Yj = Feat(:,j);
%             Muj = Mu_cond(:,j);
%             Sigmaj = diag(Var_cond(:,j));
%             sample = randmvn(Muj,Sigmaj,M,eps)';
            %             alpha(:,j) = -0.5 * (Yj - sample(:,1)).^2./(nugs(1)/num_rep+eps^2) + 0.5 * (Yj - sample(:,2)).^2./(nugs(2)/num_rep+eps^2);
            sample1 = randn(M,1)*sqrt(Var_cond(1))+repmat(Mu_cond(1),M,1);
            sample2 = randn(M,1)*sqrt(Var_cond(2))+repmat(Mu_cond(2),M,1);
            alpha(:,j) = -0.5 * (Yj - sample1).^2./(nugs(1)/num_rep+eps^2) + 0.5 * (Yj - sample2).^2./(nugs(2)/num_rep+eps^2);
       end
            log_prior_dens = NaN(size(param_range,1),2);
            for p = 1:size(param_range,1)
              log_prior_dens(p,:) = tnorm([theta_new(p),theta_old(p)], theta_mu(p), theta_sigma(p), param_range(p,:));
            end
            
            log_prior_ratio = sum(log_prior_dens(:,1) - log_prior_dens(:,2));
            %log_prior_ratio = 0;
            log_like_ratio = sum(alpha,2);
            
            log_trans_dens_new = NaN(size(param_range,1),2);
            log_trans_dens_old = NaN(size(param_range,1),2);
            for p = 1:size(param_range,1)
              log_trans_dens_new(p,:) = tnorm(theta_new(p), theta_old(p), trans_step(p), param_range(p,:));
              log_trans_dens_old(p,:) = tnorm(theta_old(p), theta_new(p), trans_step(p), param_range(p,:));
            end
            
            log_trans_ratio = sum(log_trans_dens_old - log_trans_dens_new); % all cancell out except the jacobian.
            
            Alpha =min([ones(M,1) exp(log_prior_ratio + log_like_ratio + log_trans_ratio)],[],2);
            [Eps,Tau] = unconditionError(Alpha,0.001);
            if Eps <= ksi
              break
            else
              [n_training,~,num_rep] = size(X);
              [Theta_new, X_delta] = getIniX1a(a,dt_range_new,p_range_new,chkt,delta, num_rep,time_update,Z0);
              [n_training_delta,~,num_rep] = size(X_delta); 
              feat_delta = NaN(n_training_delta + n_training,1,num_rep);
              for k = 1:num_rep
                feat_delta((n_training + 1):(n_training_delta + n_training),:,k) = X_delta(:,:,k);
                feat_delta(1:n_training,:,k) = X(:,:,k);
              end
              X = feat_delta;
              theta_list_new = [theta_list;Theta_new];
              theta_list = theta_list_new;
              Theta_new_s = (Theta_new - repmat(param_range(:,1)',delta,1)) ./...
                repmat((param_range(:,2) - param_range(:,1))',delta,1);
              theta_list_new_s = [theta_list_s; Theta_new_s];
              theta_list_s = theta_list_new_s;
              [param, model] = mleHomGP(theta_list_s,mean(X,3),init_param,bounds);
              [param_sd, model_sd] = mleHomGP(theta_list_s,std(X,[],3),init_param,bounds);
              for j = 1:size(p_range_new,1)
                  p_range_new(j,:) = [p_range_new(j,1) - 0.001*iter, p_range_new(j,2)+0.001*iter];
              end
              if length(dt_range_new) == 2
                dt_range_new = [dt_range_new(1) - 0.001*iter, dt_range_new(2)+0.001*iter];
              end
            end
       end
       if rand(1) < Tau
         Sample(ss,:) = theta_new;
       else
         Sample(ss,:) = theta_old;
       end
       theta_old = Sample(ss,:);
         %h = waitbar(ss/N,h,['remaining sample =',num2str(N-ss),'samples']);
    end
    runningTime = toc(startTime);
    %if model_spec.saveSample == 1
    % save('init_1a_final.mat','theta_list', 'theta_list_s','X');
    %end
         close(gcf)