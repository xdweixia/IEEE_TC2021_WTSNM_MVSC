function [Z,E,F,Z_hat,converge_Z,converge_Z_G] = solve_WTSNM_MVSC(X,nC,alpha,lambda,beta,mi)
% X is a cell data, each cell is a matrix in size of d_v *N,
% each column is a sample 
% min {lambda||E||_2,1 + ||\mathcal Z||_w,* + 2*alpha*tr(F^T*L_\hat Z*F)}
% s.t. X^v = X^v*Z^v + E^v,v=1,2,....,m, F^T*F=I
% \mathcal Z = rotate {tensor(Z^1,....Z^m)} in size of N*m*N
% E=[E^1;....;E^m];
% \hat Z = {sum_v=1 ^m {(|Z^v|+|Z^v|^T)/2}}/m
% beta: is a weighted vector of the tensor Schatten p-norm 

nV = length(X);
N = size(X{1},2);
tem_Z = zeros(N,N);

%% ============================ Initialization ============================
for k=1:nV
    Z{k} = zeros(N,N); %Z{2} = zeros(N,N);
    W{k} = zeros(N,N);
    J{k} = zeros(N,N);
    E{k} = zeros(size(X{k},1),N); %E{2} = zeros(size(X{k},1),N);
    Y{k} = zeros(size(X{k},1),N); %Y{2} = zeros(size(X{k},1),N);
    tem_Z = tem_Z + (abs(Z{k})+(abs(Z{k}))')./2;
end
Z_hat = tem_Z./nV;
P = zeros(N,N);
F = zeros(N,nC);
w = zeros(N*N*nV,1);
j = zeros(N*N*nV,1);
sX = [N, N, nV];

Isconverg = 0;epson = 1e-7;

iter = 0;
mu = 10e-5; max_mu = 10e10; pho_mu = 2;
rho = 0.0001; max_rho = 10e12; pho_rho = 2;

converge_Z=[];
converge_Z_G=[];

%% ================================ Upadate ===============================
while(Isconverg == 0)

%% ============================== Upadate Z^k =============================
      for i=1:N
          for l= 1:N
              P(l,i) = (norm(F(l,:)-F(i,:),2))^2;
              if Z_hat(l,i)>0
                  tem_P(l,i) = P(l,i);
              elseif Z_hat(l,i)==0
                    tem_P(l,i) = 0;  
                  else 
                    tem_P(l,i) = -P(l,i);
              end
          end
      end
      clear i l
      
      temZ = zeros(N,N);
       temp_E =[];
       
      for k =1:nV
          tmp = X{k}'*Y{k} + mu*X{k}'*X{k} +  rho*J{k}- mu*X{k}'*E{k} - W{k} -(alpha/mu)*tem_P ;
          Z{k}=inv(rho*eye(N,N)+ mu*X{k}'*X{k})*tmp;
          temZ = temZ + (abs(Z{k})+(abs(Z{k}))')./2;
          temp_E=[temp_E;X{k}-X{k}*Z{k}+Y{k}/mu];
      end
      Z_hat = temZ./nV;
      clear k 

%% =========================== Upadate E^k, Y^k ===========================
       [Econcat] = solve_l1l2(temp_E,lambda/mu);
       ro_b =0;
       
       E{1} =  Econcat(1:size(X{1},1),:);
       Y{1} = Y{1} + mu*(X{1}-X{1}*Z{1}-E{1});
       ro_end = size(X{1},1);
       for i=2:nV
           ro_b = ro_b + size(X{i-1},1);
           ro_end = ro_end + size(X{i},1);
           E{i} =  Econcat(ro_b+1:ro_end,:);
           Y{i} = Y{i} + mu*(X{i}-X{i}*Z{i}-E{i});
       end

%% ============================= Upadate J^k ==============================
        Z_tensor = cat(3, Z{:,:});
        W_tensor = cat(3, W{:,:});
        z = Z_tensor(:);
        w = W_tensor(:);
        [j, objV] = wshrinkObj_weight_lp(z + 1/rho*w,beta./rho,sX, 0,3,mi);
        J_tensor = reshape(j, sX);

%% ============================== Upadate W ===============================
        w = w + rho*(z - j);
        
%% ============================== Upadate F ===============================
        Z_hat = (Z_hat+Z_hat')/2; 
        D = diag(sum(Z_hat));
        L = D-Z_hat;
       DN = diag( 1./sqrt(sum(Z_hat)+eps) );
      LapN = speye(N) - DN * Z_hat * DN;

     [~,~,vN] = svd(LapN);
     kerN = vN(:,N-nC+1:N);
     for i = 1:N
        F(i,:) = kerN(i,:) ./ norm(kerN(i,:)+eps);
     end

%% ============================== Recording ===============================
    objV = objV + 2*alpha*trace(F'*L*F);
    history.objval(iter+1)   =  objV;

%% ====================== Checking Coverge Condition ======================
    max_Z=0;
    max_Z_G=0;
    Isconverg = 1;
    for k=1:nV
        if (norm(X{k}-X{k}*Z{k}-E{k},inf)>epson)
            history.norm_Z = norm(X{k}-X{k}*Z{k}-E{k},inf);
            Isconverg = 0;
            max_Z=max(max_Z,history.norm_Z );
        end
        
        J{k} = J_tensor(:,:,k);
        W_tensor = reshape(w, sX);
        W{k} = W_tensor(:,:,k);
        if (norm(Z{k}-J{k},inf)>epson)
            history.norm_Z_G = norm(Z{k}-J{k},inf);
            Isconverg = 0;
            max_Z_G=max(max_Z_G, history.norm_Z_G);
        end
    end
    converge_Z=[converge_Z max_Z];
    converge_Z_G=[converge_Z_G max_Z_G];
   
    if (iter>200)
        Isconverg  = 1;
    end
    iter = iter + 1;
    mu = min(mu*pho_mu, max_mu);
    rho = min(rho*pho_rho, max_rho);
end
