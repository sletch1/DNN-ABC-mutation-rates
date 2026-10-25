%% This function is for generating MCMC posterior samples of 
%% constant the mutation rate based on bMBP model using ABC-MCMC estimator
% Input: theta_mu: prior for theta, log scale
%        theta_sigma: prior standard deviation for theta
%        a_dt: the relative growth rate, a fixed number.
%        p_range; a vector storing range of the mutation rate (in log scale)
%        obs_X: observed data (4th root of the mutation freq)
%        model_spec: model specifications, including: 
%                    M: number of samples each proposed 
%                    num_training_theta: vector of 2, the number of training data points 
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
%                     bounds: bounds for the parameters to be estimated
%                     trans_step: stepwidth for proposal distribution
function [Sample, runningTime, nacc] = ABC_MCMC(theta_mu, theta_sigma, a_dt, p_range, obs_X, model_spec)

    M = model_spec.M;
    N = model_spec.N;
    a = model_spec.a;
    eps = model_spec.eps;
    
    chkt = model_spec.chkt;
    Z0 = model_spec.Z0;

    num_rep = model_spec.num_rep;
    time_update = model_spec.time_update;

    %trans_step = [0.08,0.08,0.08];
    trans_step = model_spec.trans_step;
    
    % parameter range
    param_range = p_range;
    
    Y_m = mean(obs_X);
    Y_sd = std(obs_X);
    Feat = [Y_m, Y_sd];
    
    J = size(Feat,2); % number of the features
    theta_old = model_spec.theta_old;

    Sample = NaN(N,1);
    
    startTime = tic;
    %h = waitbar(0,'Please wait...');
    nacc = 0;
    for ss =  1:N
        if mod(ss, time_update) == 0 % screen print to show progress.
            fprintf(['ss=', int2str(ss),'\n']);
        end
        [theta_new,~] = get_theta1(trans_step,theta_old,p_range);
    
        alpha = NaN(M,J);  
        feat_new = NaN(M,2);
        feat_old = NaN(M,2);
        parfor m = 1:M
            [~,m_new,s_new] = MCMC_sample1(a, a_dt, theta_new, p_range,chkt, num_rep, Z0);
            feat_new(m,:) = [m_new, s_new];
            [~,m_old,s_old] = MCMC_sample1(a, a_dt, theta_old, p_range,chkt, num_rep, Z0);
            feat_old(m,:) = [m_old, s_old];
        end
        for j = 1:J
            Yj = Feat(:,j);
            alpha(:,j) = -0.5 * (Yj - feat_new(:,j)).^2./(1e-7 + eps^2) +...
            0.5 * (Yj - feat_old(:,j)).^2./(1e-7 + eps^2);
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
        Tau = median(Alpha);
      
       if rand(1) < Tau
         Sample(ss) = theta_new;
         nacc = nacc + 1;
       else
         Sample(ss) = theta_old;
       end
       theta_old = Sample(ss);
         %h = waitbar(ss/N,h,['remaining sample =',num2str(N-ss),'samples']);
    end
    runningTime = toc(startTime);
