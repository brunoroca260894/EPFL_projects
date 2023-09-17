clear all; colors=get(gca,'ColorOrder'); close all; clc;
set(0,'defaultAxesFontSize',20); set(0,'defaultLineLineWidth',2);
addpath('./adeim-master');

N=500;
FinalTime=200; 
Nsteps=10*FinalTime;
dt=FinalTime/Nsteps;

Ntrials=50;
Ncoupl=15;

n_latent_pod=10; 
n_samples_deim=10;

rng('default');

phi0=linspace(0,1*pi,N).';
%phi0=zeros(N,1);
y0=phi0;

%uniform distribution
%a=0.97; b=1.03; omega_all=a+(b-a)*rand(N,Ntrials); Kc=2/(pi*1/(b-a));
%K=0;
%K=0.0382;
%K=0.15;

%normal distribution
mu=1; sig=0.02; omega_all=mu+sig*normrnd(0,1,[N,Ntrials]); Kc=2/(pi*1/sqrt(2*pi*sig^2)); 
%K=0;
%K=0.0319;
K=0.3;

%cauchy distribution
%omega_0=1; gam=0.1; omega_all = omega_0+gam*tan(pi*(rand(N,Ntrials)-1/2)); Kc=2*gam; 
%K=0;
%K=0.2;
%K=1.8;

omega_all=omega_all-mean(omega_all,1);

[omega_all,ids]=sort(omega_all,1);
%ids=repmat((1:N)',1,Ntrials);

K_all=linspace(0,K,Ncoupl);

%function
%see end of script

%distribution of sampled natural frequencies
figure;
subplot(2,1,1); histogram(omega_all);
subplot(2,1,2); plot(1:N,omega_all);

%% construct training set

data=zeros(Ntrials*Ncoupl*(Nsteps+1),N);
order_train=zeros(Nsteps+1,Ntrials,Ncoupl);

for k=1:Ncoupl
    disp(k);
    for i=1:Ntrials
        [t_train,y_train]=RK3 (@(t,y) rhs_fun(t,y,omega_all(:,i),K_all(k),N),[0 FinalTime], y0(ids(:,i)), dt);
        data((k-1)*(Nsteps+1)*Ntrials+(Nsteps+1)*(i-1)+1:(k-1)*(Nsteps+1)*Ntrials+(Nsteps+1)*i,:)=y_train;
        phi_train=y_train;
        order_train(:,i,k)=abs(mean(exp(1i*phi_train),2));
    end
end

%compute svd
[U,S,~]=svd(data.','econ'); U=U(:,1:n_latent_pod); S=diag(S); 

%check decay of singular values
figure;
semilogy(1:length(S),S/S(1),'color',colors(1,:));
xlabel('$i$','interpreter','latex'); ylabel('$\sigma_i/\sigma_1$','interpreter','latex');

%visualize pod modes
figure;
%plot(1:size(U,1),U.*sqrt(S(1:size(U,2)))');
%xlabel('$i$','interpreter','latex'); ylabel('$\sqrt{\sigma_j} u_j$','interpreter','latex');
plot(1:size(U,1),U);
xlabel('$i$','interpreter','latex'); ylabel('$u_j$','interpreter','latex');

order_train_final=squeeze(order_train(end,:,:));

%visualize distribution of order parameters
figure;
for k=1:Ncoupl
    subplot(ceil(Ncoupl/2),2,k); histogram(order_train_final(:,k),linspace(0,1,100),'Normalization','probability');
    xlabel(['$R(K=' num2str(K_all(k)) ')$'],'interpreter','latex'); ylabel('$freq$','interpreter','latex');
end

%% simulate for the full range of natural frequency and coupling

Pdeim = deim(U);
PTUinv=U(Pdeim,:)\eye(size(U,2));
PTU=U(Pdeim,:);

Ntrials_test=Ntrials;
omega_test=omega_all; 
ids_test=ids;

%Ntrials_test=Ntrials;
%omega_test=a+(b-a)*rand(N,Ntrials_test);
%omega_test=mu+sig*normrnd(0,1,[N,Ntrials_test]);
%omega_test=omega_0+gam*tan(pi*(rand(N,Ntrials_test)-1/2));
%omega_test=omega_test-mean(omega_test,1);
%[omega_test,ids_test]=sort(omega_test,1);
%ids_test=repmat((1:N)',1,Ntrials_test);

Ncoupl_test=10;
K_test=linspace(0,K,Ncoupl_test);


for k=1:size(K_test,2)
    disp(k);
    
    for i=1:size(omega_test,2)
        
        %full model
        [time_full,y_full] = RK3 (@(t,y) rhs_fun(t,y,omega_test(:,i),K_test(k),N),[0 FinalTime], y0(ids_test(:,i)), dt);
        phi_full=y_full;
        order_full=abs(mean(exp(1i*phi_full),2));
        
        %POD
        [time_pod,y_pod_proj] = RK3 (@(t,y) U'*rhs_fun(t,U*y,omega_test(:,i),K_test(k),N),[0 FinalTime], U'*y0(ids_test(:,i)), dt);
        y_pod=y_pod_proj*U';
        phi_pod=y_pod;
        order_pod=abs(mean(exp(1i*phi_pod),2));
        
        %DEIM
        PTomega=omega_test(Pdeim,i);
        [time_deim,y_deim_proj]=RK3(@(t,y) PTUinv*rhs_fun_colloc(t,y,PTomega,K_test(k),N,U,PTU), [0 FinalTime], U'*y0(ids_test(:,i)), dt);
        y_deim=y_deim_proj*U';
        phi_deim=y_deim;
        order_deim=abs(mean(exp(1i*phi_deim),2));
        
        %save quantities
        order_full_rand(:,i,k)=order_full;
        order_pod_rand(:,i,k)=order_pod;
        order_deim_rand(:,i,k)=order_deim;
        
    end
end

%compute mean and variance over samples
mean_full=squeeze(mean(order_full_rand,2)); var_full=squeeze(var(order_full_rand,0,2));
mean_pod=squeeze(mean(order_pod_rand,2)); var_pod=squeeze(var(order_pod_rand,0,2));
mean_deim=squeeze(mean(order_deim_rand,2)); var_deim=squeeze(var(order_deim_rand,0,2));

%visualize confidence interval for order parameter as fcn of time for different values of K
figure;
for k=1:Ncoupl_test
    subplot(ceil(Ncoupl_test/2),2,k);
    plot_confidence_interval(mean_deim(:,k)-1.96*sqrt(var_deim(:,k)/size(omega_test,2)),mean_deim(:,k)+1.96*sqrt(var_deim(:,k)/size(omega_test,2)),time_deim,colors(2,:));hold on;
    plot_confidence_interval(mean_pod(:,k)-1.96*sqrt(var_pod(:,k)/size(omega_test,2)),mean_pod(:,k)+1.96*sqrt(var_pod(:,k)/size(omega_test,2)),time_pod,colors(1,:));
    plot_confidence_interval(mean_full(:,k)-1.96*sqrt(var_full(:,k)/size(omega_test,2)),mean_full(:,k)+1.96*sqrt(var_full(:,k)/size(omega_test,2)),time_full,[0 0 0]);
    legend({'DEIM','POD','FOM'},'interpreter','latex');
    xlabel('$t$','interpreter','latex'); ylabel(['$R(K=' num2str(K_test(k)) ')$'],'interpreter','latex');
end

%asymptotic order parameters
order_full_final=squeeze(order_full_rand(end,:,:));
order_pod_final=squeeze(order_pod_rand(end,:,:));
order_deim_final=squeeze(order_deim_rand(end,:,:));

%visualize distribution of order parameters for different coupling
figure;
for k=1:Ncoupl_test
    subplot(Ncoupl_test,3,(k-1)*3+1); histogram(order_full_final(:,k),linspace(0,1,100),'Normalization','probability','facecolor',[0 0 0]);
    subplot(Ncoupl_test,3,(k-1)*3+2); histogram(order_pod_final(:,k),linspace(0,1,100),'Normalization','probability','facecolor',colors(1,:));
    subplot(Ncoupl_test,3,(k-1)*3+3); histogram(order_deim_final(:,k),linspace(0,1,100),'Normalization','probability','facecolor',colors(2,:));
end

%visualize confidence interval of asymptotic order parameter vs coupling strength
figure;
plot_confidence_interval(mean_deim(end,:)-1.96*sqrt(var_deim(end,:)/size(omega_test,2)),mean_deim(end,:)+1.96*sqrt(var_deim(end,:)/size(omega_test,2)),K_test,colors(2,:));hold on;
plot_confidence_interval(mean_pod(end,:)-1.96*sqrt(var_pod(end,:)/size(omega_test,2)),mean_pod(end,:)+1.96*sqrt(var_pod(end,:)/size(omega_test,2)),K_test,colors(1,:));
plot_confidence_interval(mean_full(end,:)-1.96*sqrt(var_full(end,:)/size(omega_test,2)),mean_full(end,:)+1.96*sqrt(var_full(end,:)/size(omega_test,2)),K_test,[0 0 0]);
legend({'DEIM','POD','FOM'},'interpreter','latex');
xlabel('$K$','interpreter','latex'); ylabel('$R$','interpreter','latex');



%% functions to compute the rhs

function out = rhs_fun(t,y,omega,K,N) 

    out=omega+K/N*(sum(sin(y),1)*cos(y)-sum(cos(y),1)*sin(y));

end


function out = rhs_fun_colloc(t,y,omega,K,N,U,PTU) 

    reconstr=U*y;
    reconstr_loc=PTU*y;

    out=omega+K/N*(sum(sin(reconstr),1)*cos(reconstr_loc)-sum(cos(reconstr),1)*sin(reconstr_loc));

end


function out = rhs_fun_groups(t,y,omega,K,N,Nl) 

    averages=y/sqrt(Nl);

    out=omega+K/N*Nl^2*(sum(sin(averages),1)*cos(averages)-sum(cos(averages),1)*sin(averages));
    out=out/sqrt(Nl);
    
end