% This is the function for proposing new theta 
% Input: 
%   trans_step: step width of proposal distribution
%   theta_old: theta from last step
%   dt_range: the range for relative fitness
%   p_range: the range for mutation rate
% Output:
%   theta: the new proposed theta
%   dt_range_new: the new parameter range of relative fitness for refinement
%   p_range_new: the new parameter range of mutation rate for refinement

function [theta, dt_range_new, p_range_new] = get_theta1a(trans_step,theta_old,dt_range,p_range)
  p_par = size(p_range,1);
  
  pd = makedist('Normal','mu',theta_old(1),'sigma',trans_step(1));
  t = truncate(pd,dt_range(1), dt_range(2));
  theta  = random(t,1,1);
  dt_range_new = [min(theta,theta_old(1))-1e-8,max(theta,theta_old(1))+1e-8];
  
  pd = makedist('Normal','mu',theta_old(2),'sigma',trans_step(2));
  t = truncate(pd,p_range(1), p_range(2));
  theta_tmp = random(t,1,1);
  p_range_new = [min(theta_tmp,theta_old(2))-1e-8,max(theta_tmp,theta_old(2))+1e-8];
  theta = [theta theta_tmp];
end