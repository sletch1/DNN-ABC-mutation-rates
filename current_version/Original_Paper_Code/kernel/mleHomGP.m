function [param, model] = mleHomGP(theta_list,Z,init_param,bounds)
    Z0 = mean(Z,3);
    mult = size(Z,3);
    n = size(theta_list,1);
    Z = reshape(Z,n,mult);
    Z = reshape(Z',numel(Z),1);
    
    model.Z0 = Z0;
    model.X0 = theta_list;
    model.Z = Z;
    
    eps = 1e-8;
    g_min = eps;
    g_max = 1e2;
    
    lower = [bounds.lower,g_min]';
    upper = [bounds.upper,g_max]';
    
    N = length(Z); 
    
    fcn{1} = @(x)ll(x,theta_list,Z0,Z,mult,eps);
    fcn{2} = @(x)dll(x,theta_list,Z0,Z,mult,eps);
    opts.x0 = init_param;
    opt.printEvery=1000;
    
    [par,f] = lbfgsb( fcn, lower, upper, opts);
    param.theta = par(1:(length(par)-1));
    param.g = par(length(par));
    model.f = f;
    
    C = se_kern_fast(param.theta, theta_list);
    Ki = jitterChol(C + (param.g/mult)*eye(n),eps);
    Ki = solve_triu(Ki, eye(n));
    Ki = Ki*Ki';
    model.Ki = Ki;
    
    beta0 = sum(Ki) * Z0 /sum(sum(Ki));
    model.beta0 = beta0;
    
    psi_0 = (Z0-beta0)' * Ki * (Z0-beta0);
    param.tau = 1/N * (((Z-beta0)' *(Z-beta0) - mult*(Z0-beta0)'*(Z0-beta0))/param.g+psi_0);
end
function fn = ll(x,X0,Z0,Z,mult,eps)
    idx = 1;
    theta = x(idx:size(X0,2));
    idx = idx+size(X0,2);
    g = x(idx);
    fn = -logLikHom(X0,Z0,Z,mult,theta,g,eps);
    end
    function gr = dll(x,X0,Z0,Z,mult,eps)
    idx = 1;
    theta = x(idx:size(X0,2));
    idx = idx+size(X0,2);
    g = x(idx);
    [gr_theta, gr_g] = dlogLikHom(X0, Z0, Z, mult, theta, g,eps);
    gr = [-gr_theta', -gr_g]';
    end

