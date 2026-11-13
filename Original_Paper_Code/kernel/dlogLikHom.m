function [tmp1, tmp2] = dlogLikHom(X0, Z0, Z, mult, theta, g,eps)
    N = length(Z);
    n = size(X0,1);
    
    C = se_kern_fast(theta, X0);
    Ki = jitterChol(C + (g/mult)*eye(n),eps);
    Ki = solve_triu(Ki, eye(n));
    Ki = Ki*Ki';
    
    beta0 = sum(Ki) * Z0 /sum(sum(Ki));
    Z0 = Z0 - beta0;
    Z = Z - beta0;
    KiZ0 = Ki *Z0;
    
    psi = Z0' * KiZ0;
    
    tmp1 = NaN(length(theta),1);
    if(length(theta)==1) %isotropy
        dC_dthetak = se_dist_fast(X0)/(theta^2).*C;
        tmp1 = (N/2 * KiZ0' * dC_dthetak * KiZ0)/((Z'*Z-(mult*Z0)'*Z0)/g +psi)- 1/2*sum(sum(Ki.*dC_dthetak',2));
    else
        for i=1:length(theta)
            dC_dthetak = se_dist_fast(X0(:,i))/(theta(i))^2 .*C;
            tmp1(i) = (N/2 * KiZ0' * dC_dthetak * KiZ0)/((Z'*Z-(mult*Z0)'*Z0)/g +psi)- 1/2*sum(sum(Ki.*dC_dthetak',2));
        end
    end
    
    tmp2 = N/2*((Z'*Z - (mult*Z0)' * Z0)/(g^2) + sum(KiZ0.^2/mult)) / ((Z'*Z-(mult*Z0)'*Z0)/g +psi) - (N-n)/(2*g)-1/2*sum(diag(Ki)/mult);
