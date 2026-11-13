% This is the function for proposing new theta 
% Input: 
  %   trans_step: step width of proposal distribution
  %   theta_old: theta from last step
  %   p_range: the range for parameters
  %   constraint: 0, no constraint; 1, stage 1 mutation rate smaller than stage 2
% Output:
  %   theta: the new proposed theta
  %   p_range_new: the new range of mutation rates for refinement

function [theta,p_range_new] = get_theta2(trans_step,theta_old, p_range,constraint)
  p_par = size(p_range,1);
  theta = [1 0 0];
  p_range_new = NaN(p_par,2);
  
  if constraint == 1
    while theta(2) > theta(1)
      for i = 1:2
        mu_temp = theta_old(i);
        sig_temp = trans_step(i);
        upper_temp = p_range(i,2);
        lower_temp = p_range(i,1);
        pd = makedist('Normal','mu',mu_temp,'sigma',sig_temp);
        t = truncate(pd,lower_temp, upper_temp);
        theta(i) = random(t,1,1);
        p_range_new(i,:) = [min(theta(i),mu_temp)-1e-8,max(theta(i),mu_temp)+1e-8];
      end
    end
    mu_temp = theta_old(3);
    sig_temp = trans_step(3);
    upper_temp = p_range(3,2);
    lower_temp = p_range(3,1);
    pd = makedist('Normal','mu',mu_temp,'sigma',sig_temp);
    t = truncate(pd,lower_temp, upper_temp);
    theta(3) = random(t,1,1);
    p_range_new(3,:) = [min(theta(3),mu_temp)-1e-8,max(theta(3),mu_temp)+1e-8];
  elseif constraint == 0
    for i = 1:p_par
      mu_temp = theta_old(i);
      sig_temp = trans_step(i);
      upper_temp = p_range(i,2);
      lower_temp = p_range(i,1);
      pd = makedist('Normal','mu',mu_temp,'sigma',sig_temp);
      t = truncate(pd,lower_temp, upper_temp);
      theta(i) = random(t,1,1);
      p_range_new(i,:) = [min(theta(i),mu_temp)-1e-8,max(theta(i),mu_temp)+1e-8];
    end
  else
    error('constraint can only be 0 or 1. 1: mu1 smaller than mu2; 0: no constraint');
  end
end