function Knm = se_kern_fast(theta, X, Xu)
%function Knm = covfuncCompute(logtheta, X, Xu)
%
%Description:  It computes the covariance function between 
%              two set of inputs points: X and Xu.  
%

%
%Supported covariance functions:  RBF and ARD kernel.   

% Zhu: This function compute the squared expeontial kernal between two sets
% of events X and Xu.
% Input:
% theta: K(x,xu)=theta(1)^2 exp(-||x-xu||^2/2/theta(2)^2) 
% X:     N by d matrix, each row is one event location.
% Xu:    M by d matrix.
% Output:
% Knm:   A N by M matrix of Kernel evaluated at the points X, Xu.
%jitter = 1e-9; 

n = size(X,1);
X = X./repmat(sqrt(theta'),n,1);

% X_min = min(X);
% X_max = max(X);

%X = (X-repmat(X_min,n,1))./repmat(X_max-X_min,n,1);
if nargin == 3
   m = size(Xu,1);   
   Xu = Xu./repmat(sqrt(theta'),m,1);
   %Xu = (Xu-repmat(X_min,n,1))./repmat(X_max-X_min,m,1);
   Knm = -2*X*Xu' + repmat(sum(X.*X,2),1,m) + repmat(sum(Xu.*Xu,2)',n,1);
   %Knm = theta(1)^2*exp(-0.5*Knm);
   Knm = exp(-Knm);
else
   Knm=repmat(sum(X.*X,2),1,n);
   %Knm = -2*(X*X') + repmat(sum(X.*X,2)',n,1) + repmat(sum(X.*X,2),1,n);
   Knm = -2*X*X' + Knm + Knm'; %#ok<MHERM>
   %Knm = theta(1)^2*exp(-0.5*Knm);
   Knm = exp(-Knm);
   Knm=(Knm+Knm')/2; % the trick (A+A')/2 simply make the matrix symmetric. 
end
