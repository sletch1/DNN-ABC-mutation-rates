function [mu, sd2, nugs] = GPHomPrediction(x,model,param)
    model.Ki = model.Ki/param.tau;
    kx = param.tau * se_kern_fast(param.theta, model.X0, x);
    nugs = repmat(param.tau*param.g,size(x,1),1);
    mu = model.beta0+kx'*(model.Ki *(model.Z0-model.beta0));
    sd2 = param.tau - sum(kx'.*(model.Ki * kx)',2)+(1-sum(model.Ki,2)'*kx)'.^2/sum(sum(model.Ki));
     