% This function is for getting training theta list as the input for bMBP model with piece-wise constant mutation rates.
% Input: a: rate parameter of the bMBP model.
%        dt_range: the range of the ratio between mutant growth rate and normal growth rate.
%        p_range; range of the 2-stage mutation rate parameters
%        chkt: time checking point
%        num_training: vector of 2, the number of training data points for initial training and additional trainings. 
%        num_rep: number of replicates at each training point             
%        time_update: for showing progress 
%        Z0: number of bacteria in each culture
% Output: theta_list: training points
%         Y: Output from model

function [theta_list,Y] = getIniX2a(a, dt_range,p_range,chkt,num_training, num_rep,time_update, Z0, constraint)
    param_range = [dt_range;p_range];
    
    p_par = size(p_range,1);
    
    [theta_list] = getThetaList(num_training, param_range, constraint);
    n_theta = size(theta_list,1);
    Y = NaN(n_theta,1,num_rep);
    parfor i = 1:n_theta
        theta_temp = theta_list(i,:);
        a_dt = 10^theta_temp(1);
        mu_vec_temp = [10^(theta_temp(2)) 10^(theta_temp(3))];
        t_temp = theta_temp(4);
        mu_par_temp =[mu_vec_temp t_temp];
                
        if mod(i, time_update/50) == 0 % screen print to show progress.
            fprintf(['i=', int2str(i),'\n']);
        end
        
        for j = 1:num_rep
            [Z, X] = mut2stage_bMBP_rev(Z0, a, a_dt, mu_par_temp(1), mu_par_temp(2), mu_par_temp(3), chkt);
            while Z == 0
              [Z, X] = mut2stage_bMBP_rev(Z0, a, a_dt, mu_par_temp(1), mu_par_temp(2), mu_par_temp(3), chkt);  
            end
            Y(i,1,j) = sqrt(sqrt(X/Z));
        end
    end
%     end
end

function[theta_list] = getThetaList(num_training, param_range, constraint)
    lhs = lhsdesign(num_training,size(param_range,1));
    theta_list = repmat(param_range(:,1)',num_training,1) + ...
            lhs.*repmat((param_range(:,2) - param_range(:,1))',num_training,1);
    if constraint == 1
        idx = theta_list(:,3) > theta_list(:,2);
        theta_list = theta_list(idx,:);
    end
    
end
