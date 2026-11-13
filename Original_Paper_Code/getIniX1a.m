% This function is for getting training theta list as the input for bMBP model with piece-wise constant mutation rates.
% Input: a: rate parameter of the bMBP model.
%        param_range; a structure storing range of each parameter
%        chkt: time checking point
%        num_training: vector of 2, the number of training data points for initial training and additional trainings. 
%        num_rep: number of replicates at each training point             
%        time_update: for showing progress 
%        Z0: number of bacteria in each culture
% Output: theta_list: training points
%         Y: Output from model

function [theta_list,Y] = getIniX1a(a, dt_range, p_range,chkt,num_training, num_rep,time_update, Z0)
    param_range = [dt_range;p_range];
    
    p_par = size(p_range,1);
    
    [theta_list] = getThetaList(num_training, param_range);
    n_theta = size(theta_list,1);
    Y = NaN(n_theta,1,num_rep);
    parfor i = 1:n_theta
        theta_temp = theta_list(i,:);
        a_dt = 10^theta_temp(1);
        mu_par_temp = 10^(theta_temp(2));
                
        if mod(i, time_update/50) == 0 % screen print to show progress.
            fprintf(['i=', int2str(i),'\n']);
        end
        
        for j = 1:num_rep
            [Z, X] = mut_bMBP_rev(Z0, a, a_dt, mu_par_temp, chkt);
            Y(i,1,j) = sqrt(sqrt(X/Z));
        end
    end
end

function[theta_list] = getThetaList(num_training, param_range)
    lhs = lhsdesign(num_training,size(param_range,1));
    theta_list = repmat(param_range(:,1)',num_training,1) + ...
            lhs.*repmat((param_range(:,2) - param_range(:,1))',num_training,1);
end
