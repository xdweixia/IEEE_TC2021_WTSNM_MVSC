% Reference:
%
%	Wei Xia, Xiangdong Zhang, Quanxue Gao, Xiaochuang Shu, Jungong Han,
%   Xinbo Gao, "Multiview Subspace Clustering by an Enhanced Tensor
%   Nuclear Norm," IEEE Transactions on Cybernetics (TCYB), 2021.
%
%   version 1.0 --May./2020
%
%   Written by Wei Xia (xd.weixia AT gmail.com)

clear;
clc

folder_now = pwd;
addpath([folder_now, '\funs']);
addpath([folder_now, '\dataset']);
%% ==================== Load Datatset and Normalization ===================
load('dataset\yale');
cls_num=length(unique(gt));
nV = length(X);
for v=1:nV
    X{v}=X{v}';
end
for v=1:nV
    [X{v}]=NormalizeData(X{v});
end
fid=fopen('Log.txt','a');

%% ========================== Parameters Setting ==========================
beta = [0.5 5 100]'; % the weights, size: 1*nV, nV is the number of view
lambda = 0.1;
alpha = 1e-7;
p = 0.5;             % the p-value of tensor Schatten p-norm

%% ============================ Optimization ==============================
[Z, E, F, Z_hat, converge_Z, converge_Z_G] = solve_WTSNM_MVSC(X, cls_num, alpha, lambda, beta, p);
Predicted = SpectralClustering(Z_hat, cls_num);
result =  ClusteringMeasure(gt, Predicted)
fprintf(fid,'lambda: %.3f\n', lambda);
fprintf(fid,'alpha: %.8f\n', alpha);
fprintf(fid,'p: %f\n',p);
fprintf(fid,'Weights: %.1f %.1f %.1f\n', beta);
fprintf(fid,'result: %.4g %.4g %.4g %.4g %.4g %.4g \n', result);
