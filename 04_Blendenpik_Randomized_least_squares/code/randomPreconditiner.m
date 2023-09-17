function [x_optimal, resvec]  = randomPreconditiner(A, params, b)
% Output:
%   R - the upper triangular factor of the preconditioner
%rng(11);
[m, n] = size(A);
htimes =  params.preprocess_steps; % number of pre-processed steps
tol = params.tolerance;
maxite = params.maxit; %maximum iteration steps for LSQR and MINRES
mm = ceil(size(A, 1)/1000)*1000;

fprintf('mm value: %d \n', mm);
if mm > m
   M = [A; zeros(mm-size(A, 1), size(A, 2) )]; 
else
   M = A;
end

%fprintf('size of matrix M: %d, %d \n', size(M));

% Row mixing step by using dct
for i = 1:htimes
    D = diag( sign(randn(mm, 1) ) ); %diagonal matrix with +- on its diagonal with equal probability
    M = dct(D*M); % discrete cosine transformation        
end

t = params.gamma * n / mm;
s = rand(mm, 1);
%fprintf('size of vector s : %d\n', size(s, 1));
rows = s < t;
%fprintf('size of vector rows : %d\n', size(rows, 1));
S = diag(rows);
S_M = S*M;
S_M = S_M(all(S_M~=0, 2), :); % keep non-zero rows

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% question e)
% sampled_rate = min(1, params.gamma*n/mm);
% sampled_rows = randsample(mm, round(sampled_rate*mm));
% S_M = M(sampled_rows,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Q, R]= qr(S_M , 0); %tiny QR decomposition

fprintf('size of matrix Q : %d, %d\n', size(Q));
fprintf('size of matrix R : %d, %d\n', size(R));

kappa_tilde = rcond(R);

% to check if the condition number of the preconditioner is good
if kappa_tilde > 5*eps
    fprintf('condition kappa_tilde^-1 > 5*eps\n');    
    %%%% LSQR implementation %%%%    
%     B = A*R^-1;    
%     [x_optimal,flag,relres,iter,resvec] = lsqr(B'*B, B'*b,tol, maxite);     
%     x_optimal = R\x_optimal;    
    
    %%%% MINRES implementation 
    %%%% uncoment from 60 to 62 and comment from 53 to 55
    %%%% to use MINRES
    B = A*R^-1;    
    [x_optimal,flag,relres,iter,resvec] = minres(B'*B, B'*b,tol, maxite);     
    x_optimal = R\x_optimal;        
end
end

