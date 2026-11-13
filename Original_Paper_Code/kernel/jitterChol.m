function [L,er] = jitterChol(K,eps)
% function [L er] = jitterChol(K)
%
% Description:  Computing Choleski decomposition by adding jitter  
%              when the matrix is semipositive definite  
%

jitter = eps ;
m = size(K,1); 
[L,er] = chol(K);
if er > 0 % add jitter
   %warning('Jitter added'); 
   %K = K + jitter*mean(diag(K))*eye(m);
   K=bsxfun(@plus,K,sparse(1:m,1:m,jitter*mean(diag(K)))); % much faster than line 13
   [L,er] = chol(K);
end