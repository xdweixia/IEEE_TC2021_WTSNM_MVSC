function [x,objV] = wshrinkObj_weight_lp(x, rho, sX, isWeight, mode, p)
%rho是一个列向量，他的第i个元素对应的是第i个奇异值的权重
if isWeight == 1
    %     C = 2*sqrt(2)*sqrt(sX(3)*sX(2));
    C = sqrt(sX(3)*sX(2));
end

if ~exist('mode','var')
    % mode = 1是采用lateral slice的方法
    % mode = 2是采用front slice的方法
    % mode = 3是采用top slice的方法
    mode = 1;
end

X=reshape(x,sX);

% 将三阶的矩阵按照一定的模方向进行旋转

if mode == 1
    Y=X2Yi(X,3);
elseif mode == 3
    Y=shiftdim(X, 1);
else
    Y = X;
end

Yhat = fft(Y,[],3);
objV = 0;

% 取一定模的方向的个数
if mode == 1
    n3 = sX(2);
elseif mode == 3
    n3 = sX(1);
else
    n3 = sX(3);
end

if isinteger(n3/2)
    endValue = int16(n3/2+1);
    for i = 1:endValue
        [uhat,shat,vhat] = svd(full(Yhat(:,:,i)),'econ');
        
        if isWeight
            weight = C./(diag(shat) + eps);
            tau = rho*weight;
            shat = soft(shat,diag(tau));
        else
            tau = rho;
            shat=diag(shat);
            shat = solve_Lp_w(shat, tau, p);
            shat=diag(shat);
        end
        
        objV = objV + sum(shat(:));
        Yhat(:,:,i) = uhat*shat*vhat';
        if i > 1
            Yhat(:,:,n3-i+2) = conj(uhat)*shat*conj(vhat)';
            objV = objV + sum(shat(:));
        end
    end
    [uhat,shat,vhat] = svd(full(Yhat(:,:,endValue+1)),'econ');
    if isWeight
        weight = C./(diag(shat) + eps);
        tau = rho*weight;
        shat = soft(shat,diag(tau));
    else
        tau = rho;
        shat=diag(shat);
        shat = solve_Lp_w(shat, tau, p);
        shat=diag(shat);
    end
    objV = objV + sum(shat(:));
    Yhat(:,:,endValue+1) = uhat*shat*vhat';
else
    endValue = int16(n3/2+1);
    for i = 1:endValue
        [uhat,shat,vhat] = svd(full(Yhat(:,:,i)),'econ');
        if isWeight
            weight = C./(diag(shat) + eps);
            tau = rho*weight;
            shat = soft(shat,diag(tau));
        else
            tau=rho;
            shat=diag(shat);
            shat = solve_Lp_w(shat, tau, p);
            shat=diag(shat);
        end
        objV = objV + sum(shat(:));
        Yhat(:,:,i) = uhat*shat*vhat';
        if i > 1
            Yhat(:,:,n3-i+2) = conj(uhat)*shat*conj(vhat)';
            objV = objV + sum(shat(:));
        end
    end
end

%% 原来的逆变化
Y = ifft(Yhat,[],3);

if mode == 1
    X = Yi2X(Y,3);
elseif mode == 3
    X = shiftdim(Y, 2);
else
    X = Y;
end

x = X(:);

end
