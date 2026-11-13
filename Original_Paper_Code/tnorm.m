% This script evaluates the log density of truncated log normal
% distribution.
function  logdens= tnorm(x, mu1, sig1, trun)
% Input: 
% x is a vector of values to be caculated for the truncated log normal
% distribution. (Not in log scale)
% mu1: the mean of the log normal distribution (the log scale mean) 
% sig1: the std of the log normal distribution (the log scale std). 
% trun: a 1 by 2 vector, the lower and upper bound of the truncations. 
% Output:
% the log density of the truncated log normal evaluated at value x. 
if sig1 == 0
   logdens = zeros(1,length(x));
else
   probdist = makedist('Normal','mu', mu1, 'sigma',sig1);
   trun1 = truncate(probdist,trun(1),trun(2));
   logdens=log(pdf(trun1, x));
end

% Testing code by zhu.
% xlist = linspace(0,60,100);
% f1=pdf(probdist, xlist);
% figure()
% plot(xlist, f1)
% xlist = linspace(0,60,100);
% f2=pdf(trun1, xlist);
% figure()
% plot(xlist, f2)
% trapz(xlist,f2)

% xlist = linspace(1e-4,0.5,100);
% f1=pdf(probdist, xlist);
% figure()
% plot(xlist, f1)
% xlist = linspace(1e-4,0.5,100);
% f2=pdf(trun1, xlist);
% figure()
% plot(xlist, f2)
%trapz(xlist,f2)

% xlist = linspace(1e-4,100,100);
% f1=pdf(probdist, xlist);
% figure()
% plot(xlist, f1)
% xlist = linspace(1e-4,100,100);
% f2=pdf(trun1, xlist);
% figure()
% plot(xlist, f2)
% trapz(xlist,f2)