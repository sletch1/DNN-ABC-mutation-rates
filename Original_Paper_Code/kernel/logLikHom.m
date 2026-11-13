function loglik = logLikHom(X0,Z0,Z,mult,theta,g,eps)
n = size(X0,1);
N = length(Z);

C = se_kern_fast(theta, X0);
Ki = jitterChol(C + (g/mult)*eye(n),eps);
ldetKi = -2 * sum(log(diag(Ki)));

Ki = solve_triu(Ki, eye(n));
Ki = Ki*Ki';

beta0 = sum(Ki) * Z0 /sum(sum(Ki));

psi_0 = (Z0-beta0)' * Ki * (Z0-beta0);

psi = 1/N * (((Z-beta0)' *(Z-beta0) - mult*(Z0-beta0)'*(Z0-beta0))/g+psi_0);

loglik = -N/2 * log(2*pi)-N/2*log(psi)+1/2*ldetKi-(N-n)/2*log(g)-1/2*sum(log(mult))-N/2;

 