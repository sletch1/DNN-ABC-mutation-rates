function[Eps,Tau] = unconditionError(Alpha,delta)
% This function is used to calculate the unconditional error of making
% decisions using Monte Carlo method
%Input: 
%       Alpha:The acceptance ratio
%       delta: small interval to calculate integral
%Output: 
%        Eps: unconditional error: Definition found in the paper: GPS-ABC
%        Tau: The threshold used to make decision
Tau = median(Alpha);
M = 1/delta;
unif = linspace(0,1,M);
larger = sum(sign(repmat(Alpha,1,M)-repmat(unif,size(Alpha,1),1)) + 1,1)/2/size(Alpha,1);
smaller = 1-larger;
Eps = delta *(((sign(unif-Tau)+1)/2)*larger' + ((-sign(unif-Tau)+1)/2)*smaller');
        