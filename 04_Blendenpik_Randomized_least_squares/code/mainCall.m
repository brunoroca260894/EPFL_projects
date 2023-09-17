%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Basic Blendenpik code %%%%%%%
%%% EPFL Spring 2022. MATH-453 %%
%%%%%%%%%%%%%%% bruno rodriguez %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
clc;

epsilon_machine = eps;
tol = 1e-10;

param = struct('gamma', 4, 'preprocess_steps', 1, 'tolerance', tol, 'maxit', 5000);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% question e) %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% gamma_n = [2:1:10];
% time_lsqr = [];
% samples = 1:1:5;
% average_time = zeros(size(samples,2), size(gamma_n,2));
% j = 1;
% 
% for i = samples
%     time_lsqr =  [];
%     for gamma = gamma_n    
%         fprintf('gamma value: %d \n', gamma);
%         param.gamma  = gamma;
%         %incoherent mtrix
% %         U = orth(rand(20000, 400));
% %         S = diag(linspace(1, 1e5, 400));
% %         V = orth(rand(400));
% %         A = U*S*V;
% 
%         %coherent mtrix
%         A = [diag(linspace(1, 1e5, 400)); zeros(19600, 400)];
%         A = A + 1e-8*ones(20000, 400);
% 
%         % random vector b
%         b = rand(20000, 1);
% 
%         tic
%         [m, n] = size(A);    
%         [x_optimal, resvec]  = randomPreconditiner(A, param, b);
%         end_time = toc;
%         time_lsqr = [time_lsqr, end_time];
%         
%         fprintf('------------ \n');
%     end   
%    j = j +1;
%    average_time(i, : ) =time_lsqr;     
%    plot(gamma_n, time_lsqr, 'LineWidth', 0.8, 'Color', 'k','LineStyle','--','Marker', 'x');
%    hold on;        
% end

% plot(gamma_n, mean(average_time, 1), 'LineWidth', 1.5, 'Color', 'b','LineStyle','-','Marker', 'x', 'DisplayName', 'mean time');
% title('coherent matrix');
% xlabel('gamma \gamma');
% ylabel('time (sec)');
% grid on;
% set(findall(gcf,'-property','FontSize'),'FontSize',21);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% question f) %%%%%%%%%%
%%% default solver MINRES, to use 
%%% LSQR, see comments in 
%%% randomPreconditiner.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% incoherent matrix
rng(11);
param.gamma  = 5;
U = orth(rand(20000, 400));
S = diag(linspace(1, 1e5, 400));
V = orth(rand(400));
A = U*S*V';

% random vector b
b = rand(20000, 1);
% 
% exact_sol = A\b;
[m, n] = size(A);    
[x_optimal, residual_vector]  = randomPreconditiner(A, param, b);
total_ite = max(size(residual_vector));
num_iteration = 0:1:total_ite-1;

%plot 1
% figure
semilogy(num_iteration, residual_vector, 'LineWidth', 1.5, 'Marker', 'o', 'DisplayName', 'incoherent matrix');
title('convergence rate');
xlabel('number of iterations');
ylabel('');
grid on;
hold on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% coherent matrix
param.gamma  = 5;
A = [diag(linspace(1, 1e5, 400)); zeros(19600, 400)];
A = A + 1e-8*ones(20000, 400);
%b = rand(20000, 1);

[m, n] = size(A);    
[x_optimal, residual_vector]  = randomPreconditiner(A, param, b);
total_ite = max(size(residual_vector));
num_iteration = 0:1:total_ite-1;

% plot 2
semilogy(num_iteration, residual_vector, 'LineWidth', 1.5, 'Marker', 'o','DisplayName', 'coherent matrix');
legend
set(findall(gcf,'-property','FontSize'),'FontSize',18);