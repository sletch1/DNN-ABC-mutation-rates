% This is the function for proposing new theta 
% Input: 
  %   model: 1. constant mutation model; 2. piece-wise constant mutation rates
%   trans_step: step width of proposal distribution
%   theta_old: theta from last step
%   param_range: the range for parameters
% Output:
  %   theta: the new proposed theta
%   param_range_new: the new parameter range

function [theta, dt_range_new, p_range_new] = get_theta2a(trans_step,theta_old,dt_range,p_range,constraint)
    p_par = size(p_range,1);
    pd = makedist('Normal','mu',theta_old(1),'sigma',trans_step(1));
    t = truncate(pd,dt_range(1), dt_range(2));
    theta  = random(t,1,1);
    dt_range_new = [min(theta,theta_old(1))-1e-8,max(theta,theta_old(1))+1e-8];
    
    theta_tmp = [1 0 0];
    p_range_new = NaN(p_par,2);
    if constraint == 1
      while theta_tmp(2) > theta_tmp(1)
        for i = 1:2
          mu_temp = theta_old(i+1);
          sig_temp = trans_step(i+1);
          upper_temp = p_range(i,2);
          lower_temp = p_range(i,1);
          pd = makedist('Normal','mu',mu_temp,'sigma',sig_temp);
          t = truncate(pd,lower_temp, upper_temp);
          theta_tmp(i) = random(t,1,1);
          p_range_new(i,:) = [min(theta_tmp(i),mu_temp)-1e-8,max(theta_tmp(i),mu_temp)+1e-8];
        end
      end
      mu_temp = theta_old(4);
      sig_temp = trans_step(4);
      upper_temp = p_range(3,2);
      lower_temp = p_range(3,1);
      pd = makedist('Normal','mu',mu_temp,'sigma',sig_temp);
      t = truncate(pd,lower_temp, upper_temp);
      theta_tmp(3) = random(t,1,1);
      p_range_new(3,:) = [min(theta_tmp(3),mu_temp)-1e-8,max(theta_tmp(3),mu_temp)+1e-8];
      theta = [theta theta_tmp];
    elseif constraint == 0
      for i = 1:p_par
        mu_temp = theta_old(i+1);
        sig_temp = trans_step(i+1);
        upper_temp = p_range(i,2);
        lower_temp = p_range(i,1);
        pd = makedist('Normal','mu',mu_temp,'sigma',sig_temp);
        t = truncate(pd,lower_temp, upper_temp);
        theta_tmp(i) = random(t,1,1);
        p_range_new(i,:) = [min(theta_tmp(i),mu_temp)-1e-8,max(theta_tmp(i),mu_temp)+1e-8];
      end
      theta = [theta theta_tmp];
    else
      error('constraint can only be 0 or 1. 1: mu1 smaller than mu2; 0: no constraint');
    end
end