% This is the function for proposing new theta 
% Input: 
%   trans_step: step width of proposal distribution
%   theta_old: theta from last step
%   p_range: the range for mutation rate
% Output:
%   theta: the new proposed theta
%   p_range_new: the new parameter range

function [theta, p_range_new] = get_theta1(trans_step,theta_old,p_range)
  p_par = size(p_range,1);
  pd = makedist('Normal','mu',theta_old,'sigma',trans_step);
  t = truncate(pd,p_range(1), p_range(2));
  theta  = random(t,1,1);
  p_range_new = [min(theta,theta_old)-1e-8,max(theta,theta_old)+1e-8];
end