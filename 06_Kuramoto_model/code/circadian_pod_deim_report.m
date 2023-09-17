clear all; colors=get(gca,'ColorOrder'); close all; clc;
set(0,'defaultAxesFontSize',20); set(0,'defaultLineLineWidth',2);

N=100;
FinalTime=500; 
Nsteps=2*FinalTime;
dt=FinalTime/Nsteps;
%final time steps (used to compute QoIs)
Nasympt=Nsteps/5;

Ntrials=1;
NcouplK=5; NcouplL0=3;

n_latent_pod=10; %up to N
n_samples_deim=10;

x0=repmat([0.0998 0.2468 2.0151 0.0339].',N,1);

MC_test=false; %true if I want to use 50 iid samples for testing, independently of the training

%fixed parameters
params.nu=[0.7 0.35 NaN 0.35 NaN 0.35 NaN 1]';
params.Ki=[1 1 NaN 1 NaN 1 NaN 1]';
params.k=[NaN NaN 0.7 NaN 0.7 NaN 0.35]';
params.nuc=0.4;
params.Kc=1;
params.omega=2*pi/24;

rng('default');
%uniform distribution
K=0.6; L0=0.02; a=0.8; b=1.2; tau_all=a+(b-a)*rand(N,Ntrials); 

%normal distribution
%K=0.6; L0=0.02; mu=1; sig=0.1; omega_all=mu+sig*normrnd(0,1,[N,Ntrials]);

K_all=linspace(0,K,NcouplK);
L0_all=linspace(0,L0,NcouplL0);

%tau_all=sort(tau_all,1);

%function
%see end of script


%% construct training set
data=zeros(Ntrials*NcouplK*NcouplL0*(Nsteps+1),4*N);
data_rhs=zeros(Ntrials*NcouplK*NcouplL0*(Nsteps+1),4*N);
for k=1:NcouplK
    disp(k);
    for l=1:NcouplL0
        for i=1:Ntrials
            [t_train,x_train]=RK3 (@(t,x) circadian_rhs(t,x,params,tau_all(:,i),K_all(k),L0_all(l)),[0 FinalTime], x0, dt);
            data((k-1)*NcouplL0*(Nsteps+1)*Ntrials+(l-1)*(Nsteps+1)*Ntrials+(Nsteps+1)*(i-1)+1:(k-1)*NcouplL0*(Nsteps+1)*Ntrials+(l-1)*(Nsteps+1)*Ntrials+(Nsteps+1)*i,:)=x_train;
            %data=[data; x_train];
            
            x_rhs=zeros(size(x_train));
            for t=1:length(t_train)
                x_rhs(t,:)=circadian_rhs(t_train(t),x_train(t,:)',params,tau_all(:,i),K_all(k),L0_all(l))';
            end
            data_rhs((k-1)*NcouplL0*(Nsteps+1)*Ntrials+(l-1)*(Nsteps+1)*Ntrials+(Nsteps+1)*(i-1)+1:(k-1)*NcouplL0*(Nsteps+1)*Ntrials+(l-1)*(Nsteps+1)*Ntrials+(Nsteps+1)*i,:)=x_rhs;
        end
    end
end

%compute svd for each variable
for i=1:4
    %compute svd for X,Y,Z,V
    [U{i},S{i},~]=svd(data(:,i:4:end-4+i).','econ'); U{i}=U{i}(:,1:n_latent_pod); S{i}=diag(S{i});
end
%the matrix is blockdiagonal, but I have to reorder the indexes 
ids=[1:4:4*N-3, 2:4:4*N-2, 3:4:4*N-1, 4:4:4*N];
U_pod(ids,:)=(blkdiag(U{:}));

%check decay of singular values
figure;
semilogy(1:length(S{1}),S{1}/S{1}(1),'color',colors(1,:));
xlabel('$i$','interpreter','latex'); ylabel('$\sigma_i/\sigma_1$','interpreter','latex');


%% simulate for one instance of the natural frequency
tau=tau_all(:,1);
K=K_all(end);
L0=L0_all(end);

%full model
tic;
[time_full,x_full] = RK3 (@(t,x) circadian_rhs(t,x,params,tau,K,L0),[0 FinalTime], x0, dt);
elapsed_full=toc;
[synchrony_full,sync_param_full,avg_gene_conc_full,spectral_ampl_fact_full,estimated_period_full,order_param_X_full] = compute_QoIs(time_full,x_full,N,Nasympt,params.omega,L0);

%POD
tic;
[time_pod,x_pod_proj] = RK3 (@(t,x) U_pod'*circadian_rhs(t,U_pod*x,params,tau,K,L0),[0 FinalTime], U_pod'*x0, dt);
x_pod=x_pod_proj*U_pod';
elapsed_pod=toc;
[synchrony_pod,sync_param_pod,avg_gene_conc_pod,spectral_ampl_fact_pod,estimated_period_pod,order_param_X_pod] = compute_QoIs(time_pod,x_pod,N,Nasympt,params.omega,L0);

%DEIM with snapshots
Pdeim=deim(U{4});
Pdeim_ext = 4*(Pdeim-1)+(1:4); Pdeim_ext=sort(Pdeim_ext(:));
PTU=U_pod(Pdeim_ext,:);
PTUinv=U_pod(Pdeim_ext,:)\eye(size(U_pod,2));
meanU_v=mean(U{4},1);
PTtau=tau(Pdeim);

%DEIM with rhs
% for i=1:4
%     [Urhs_loc{i},S{i},~]=svd(data_rhs(:,i:4:end-4+i).','econ'); Urhs_loc{i}=Urhs_loc{i}(:,1:n_samples_deim); Srhs{i}=diag(S{i});
% end
% figure;
% semilogy(1:length(Srhs{1}),Srhs{1}/Srhs{1}(1)); hold on;
% xlabel('$i$','interpreter','latex'); ylabel('$\sigma_i/\sigma_1(RHS)$','interpreter','latex');
% Urhs(ids,:)=(blkdiag(Urhs_loc{:}));
% Pdeim = deim(Urhs_loc{4});
% ids_pod=[1:4:4*n_latent_pod-3, 2:4:4*n_latent_pod-2, 3:4:4*n_latent_pod-1, 4:4:4*n_latent_pod];
% ids_deim=[1:4:4*n_samples_deim-3, 2:4:4*n_samples_deim-2, 3:4:4*n_samples_deim-1, 4:4:4*n_samples_deim];
% Pdeim_ext = 4*(Pdeim-1)+(1:4); Pdeim_ext=sort(Pdeim_ext(:));
% for i=1:4, PTUinv_loc{i}=(U{i}'*Urhs_loc{i})*(Urhs_loc{i}(Pdeim,:)\eye(size(Urhs_loc{i},2))); end;  PTUinv(:,ids_deim)=blkdiag(PTUinv_loc{:});
% for i=1:4, PTU_loc{i}=U{i}(Pdeim,:); end; PTU(ids_deim,:)=blkdiag(PTU_loc{:});
% meanU_v=mean(U{4},1);
% PTtau=tau(Pdeim);

tic;
[time_deim,x_deim_proj]=RK3(@(t,x) PTUinv*circadian_rhs_colloc(t,x,params,PTtau,K,L0,PTU,meanU_v), [0 FinalTime], U_pod'*x0, dt);
x_deim=x_deim_proj*U_pod';
elapsed_deim=toc;
[synchrony_deim,sync_param_deim,avg_gene_conc_deim,spectral_ampl_fact_deim,estimated_period_deim,order_param_X_deim] = compute_QoIs(time_deim,x_deim,N,Nasympt,params.omega,L0);


%visualize original and solution every Noscil_plot oscillators
Noscil_plot=floor(N/5);
figure;
subplot(4,3,1); 
plot(time_full,x_full(:,1:4*Noscil_plot:end-3),'color',[0 0 0]); xlabel({'$t$'},'interpreter','latex'); ylabel({'$(X_i)_{FOM}$'},'interpreter','latex'); 
subplot(4,3,4);
%figure;
plot(time_full,x_full(:,2:4*Noscil_plot:end-2),'color',[0 0 0]); xlabel({'$t$'},'interpreter','latex'); ylabel({'$(Y_i)_{FOM}$'},'interpreter','latex'); 
subplot(4,3,7);
%figure;
plot(time_full,x_full(:,3:4*Noscil_plot:end-1),'color',[0 0 0]); xlabel({'$t$'},'interpreter','latex'); ylabel({'$(Z_i)_{FOM}$'},'interpreter','latex'); 
subplot(4,3,10);
%figure;
plot(time_full,x_full(:,4:4*Noscil_plot:end),'color',[0 0 0]); xlabel({'$t$'},'interpreter','latex'); ylabel({'$(V_i)_{FOM}$'},'interpreter','latex'); 
subplot(4,3,2); 
%figure;
plot(time_pod,x_pod(:,1:4*Noscil_plot:end-3),'color',colors(1,:)); xlabel({'$t$'},'interpreter','latex'); ylabel({'$(X_i)_{POD}$'},'interpreter','latex'); 
subplot(4,3,5);
%figure;
plot(time_pod,x_pod(:,2:4*Noscil_plot:end-2),'color',colors(1,:)); xlabel({'$t$'},'interpreter','latex'); ylabel({'$(Y_i)_{POD}$'},'interpreter','latex'); 
subplot(4,3,8);
%figure;
plot(time_pod,x_pod(:,3:4*Noscil_plot:end-1),'color',colors(1,:)); xlabel({'$t$'},'interpreter','latex'); ylabel({'$(Z_i)_{POD}$'},'interpreter','latex'); 
subplot(4,3,11);
%figure;
plot(time_pod,x_pod(:,4:4*Noscil_plot:end),'color',colors(1,:)); xlabel({'$t$'},'interpreter','latex'); ylabel({'$(V_i)_{POD}$'},'interpreter','latex'); 
subplot(4,3,3); 
%figure;
plot(time_deim,x_deim(:,1:4*Noscil_plot:end-3),'color',colors(2,:)); xlabel({'$t$'},'interpreter','latex'); ylabel({'$(X_i)_{DEIM}$'},'interpreter','latex'); 
subplot(4,3,6);
%figure;
plot(time_deim,x_deim(:,2:4*Noscil_plot:end-2),'color',colors(2,:)); xlabel({'$t$'},'interpreter','latex'); ylabel({'$(Y_i)_{DEIM}$'},'interpreter','latex'); 
subplot(4,3,9);
%figure;
plot(time_deim,x_deim(:,3:4*Noscil_plot:end-1),'color',colors(2,:)); xlabel({'$t$'},'interpreter','latex'); ylabel({'$(Z_i)_{DEIM}$'},'interpreter','latex'); 
subplot(4,3,12);
%figure;
plot(time_deim,x_deim(:,4:4*Noscil_plot:end),'color',colors(2,:)); xlabel({'$t$'},'interpreter','latex'); ylabel({'$(V_i)_{DEIM}$'},'interpreter','latex'); 


%visualize synchrony variable F/E(V^2)
figure;
plot(time_full,synchrony_full,'color',[0 0 0]); hold on; 
plot(time_pod,synchrony_pod,'color',colors(1,:)); 
plot(time_deim,synchrony_deim,'color',colors(2,:));
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$t$','interpreter','latex'); ylabel('$Q$','interpreter','latex'); 

%visualize average gene concentration X(t)
figure;
plot(time_full,avg_gene_conc_full,'color',[0 0 0]); hold on;
plot(time_pod,avg_gene_conc_pod,'color',colors(1,:)); 
plot(time_deim,avg_gene_conc_deim,'color',colors(2,:));
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$t$','interpreter','latex'); ylabel('$X$','interpreter','latex');

%visualize order parameter
figure;
plot(time_full(end-Nasympt:end),order_param_X_full,'color',[0 0 0]); hold on;
plot(time_pod(end-Nasympt:end),order_param_X_pod,'color',colors(1,:)); 
plot(time_deim(end-Nasympt:end),order_param_X_deim,'color',colors(2,:));
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$t$','interpreter','latex'); ylabel('$R$','interpreter','latex'); ylim([0 1]);

%display times
fprintf('Elapsed time for full model: %f\n',elapsed_full);
fprintf('Elapsed time for POD model: %f\n',elapsed_pod);
fprintf('Elapsed time for DEIM model: %f\n',elapsed_deim);


%% simulate for the full range of natural frequency, with fixed coupling
K=K_all(end);
L0=L0_all(end);

%deim with snapshots
Pdeim=deim(U{4});
Pdeim_ext = 4*(Pdeim-1)+(1:4); Pdeim_ext=sort(Pdeim_ext(:));
PTU=U_pod(Pdeim_ext,:);
PTUinv=U_pod(Pdeim_ext,:)\eye(size(U_pod,2));
meanU_v=mean(U{4},1);

%deim with rhs
% Pdeim = deim(Urhs_loc{4});
% for i=1:4, PTUinv_loc{i}=(U{i}'*Urhs_loc{i})*(Urhs_loc{i}(Pdeim,:)\eye(size(Urhs_loc{i},2))); end;  PTUinv(:,ids_deim)=blkdiag(PTUinv_loc{:});
% for i=1:4, PTU_loc{i}=U{i}(Pdeim,:); end; PTU(ids_deim,:)=blkdiag(PTU_loc{:});
% meanU_v=mean(U{4},1);

%if true, then I test with 50 iid samples, independent of training
if MC_test
    rng('default'); tau_all=a+(b-a)*rand(N,50);
end

for i=1:size(tau_all,2)
    disp(i);

    %full
    tic;
    [time_full,x_full] = RK3 (@(t,x) circadian_rhs(t,x,params,tau_all(:,i),K,L0),[0 FinalTime], x0, dt);
    elapsed_full=toc;
    [synchrony_full_rand(:,i),sync_param_full_rand(i),avg_gene_conc_full_rand(:,i),spectral_ampl_fact_full_rand(i),estimated_period_full_rand(i),order_param_X_full_rand(:,i)] = compute_QoIs(time_full,x_full,N,Nasympt,params.omega,L0);

    %POD
    tic;
    [time_pod,x_pod_proj] = RK3 (@(t,x) U_pod'*circadian_rhs(t,U_pod*x,params,tau_all(:,i),K,L0),[0 FinalTime], U_pod'*x0, dt);
    x_pod=x_pod_proj*U_pod';
    elapsed_pod=toc;
    [synchrony_pod_rand(:,i),sync_param_pod_rand(i),avg_gene_conc_pod_rand(:,i),spectral_ampl_fact_pod_rand(i),estimated_period_pod_rand(i),order_param_X_pod_rand(:,i)] = compute_QoIs(time_pod,x_pod,N,Nasympt,params.omega,L0);

    %DEIM
    PTtau=tau_all(Pdeim,i);
    tic;
    [time_deim,x_deim_proj]=RK3(@(t,x) PTUinv*circadian_rhs_colloc(t,x,params,PTtau,K,L0,PTU,meanU_v), [0 FinalTime], U_pod'*x0, dt);
    x_deim=x_deim_proj*U_pod';
    elapsed_deim=toc;
    [synchrony_deim_rand(:,i),sync_param_deim_rand(i),avg_gene_conc_deim_rand(:,i),spectral_ampl_fact_deim_rand(i),estimated_period_deim_rand(i),order_param_X_deim_rand(:,i)] = compute_QoIs(time_deim,x_deim,N,Nasympt,params.omega,L0);
    
end

%compute averages
synchrony_full=mean(synchrony_full_rand,2); synchrony_pod=mean(synchrony_pod_rand,2); synchrony_deim=mean(synchrony_deim_rand,2);
avg_gene_conc_full=mean(avg_gene_conc_full_rand,2); avg_gene_conc_pod=mean(avg_gene_conc_pod_rand,2); avg_gene_conc_deim=mean(avg_gene_conc_deim_rand,2);
sync_param_full=mean(sync_param_full_rand,2); sync_param_pod=mean(sync_param_pod_rand,2); sync_param_deim=mean(sync_param_deim_rand,2);
spectral_ampl_fact_full=mean(spectral_ampl_fact_full_rand,2); spectral_ampl_fact_pod=mean(spectral_ampl_fact_pod_rand,2); spectral_ampl_fact_deim=mean(spectral_ampl_fact_deim_rand,2);
estimated_period_full=mean(estimated_period_full_rand,2); estimated_period_pod=mean(estimated_period_pod_rand,2); estimated_period_deim=mean(estimated_period_deim_rand,2);
order_param_X_full=mean(order_param_X_full_rand,2); order_param_X_pod=mean(order_param_X_pod_rand,2); order_param_X_deim=mean(order_param_X_deim_rand,2);

%compute variances
var_synchrony_full=var(synchrony_full_rand,1,2); var_synchrony_pod=var(synchrony_pod_rand,1,2); var_synchrony_deim=var(synchrony_deim_rand,1,2);
var_avg_gene_conc_full=var(avg_gene_conc_full_rand,1,2); var_avg_gene_conc_pod=var(avg_gene_conc_pod_rand,1,2); var_avg_gene_conc_deim=var(avg_gene_conc_deim_rand,1,2);
var_sync_param_full=var(sync_param_full_rand,1,2); var_sync_param_pod=var(sync_param_pod_rand,1,2); var_sync_param_deim=var(sync_param_deim_rand,1,2);
var_spectral_ampl_fact_full=var(spectral_ampl_fact_full_rand,1,2); var_spectral_ampl_fact_pod=var(spectral_ampl_fact_pod_rand,1,2); var_spectral_ampl_fact_deim=var(spectral_ampl_fact_deim_rand,1,2);
var_estimated_period_full=var(estimated_period_full_rand,1,2); var_estimated_period_pod=var(estimated_period_pod_rand,1,2); var_estimated_period_deim=var(estimated_period_deim_rand,1,2);
var_order_param_X_full=var(order_param_X_full_rand,1,2); var_order_param_X_pod=var(order_param_X_pod_rand,1,2); var_order_param_X_deim=var(order_param_X_deim_rand,1,2);

%visualize confidence interval for synchrony variable
figure;
plot_confidence_interval(synchrony_deim-1.96*sqrt(var_synchrony_deim/size(tau_all,2)),synchrony_deim+1.96*sqrt(var_synchrony_deim/size(tau_all,2)),time_deim,colors(2,:)); hold on;
plot_confidence_interval(synchrony_pod-1.96*sqrt(var_synchrony_pod/size(tau_all,2)),synchrony_pod+1.96*sqrt(var_synchrony_pod/size(tau_all,2)),time_pod,colors(1,:));
plot_confidence_interval(synchrony_full-1.96*sqrt(var_synchrony_full/size(tau_all,2)),synchrony_full+1.96*sqrt(var_synchrony_full/size(tau_all,2)),time_full,[0 0 0]);
legend({'DEIM','POD','FOM'},'interpreter','latex');
xlabel('$t$','interpreter','latex'); ylabel('$Q$','interpreter','latex');

%visualize confidence interval for avg gene concentration X(t)
figure;
plot_confidence_interval(avg_gene_conc_deim-1.96*sqrt(var_avg_gene_conc_deim/size(tau_all,2)),avg_gene_conc_deim+1.96*sqrt(var_avg_gene_conc_deim/size(tau_all,2)),time_deim,colors(2,:)); hold on;
plot_confidence_interval(avg_gene_conc_pod-1.96*sqrt(var_avg_gene_conc_pod/size(tau_all,2)),avg_gene_conc_pod+1.96*sqrt(var_avg_gene_conc_pod/size(tau_all,2)),time_pod,colors(1,:));
plot_confidence_interval(avg_gene_conc_full-1.96*sqrt(var_avg_gene_conc_full/size(tau_all,2)),avg_gene_conc_full+1.96*sqrt(var_avg_gene_conc_full/size(tau_all,2)),time_full,[0 0 0]);
legend({'DEIM','POD','FOM'},'interpreter','latex');
xlabel('$t$','interpreter','latex'); ylabel('$X$','interpreter','latex');

%visualize confidence interval for order parameter as fcn of time
figure;
plot_confidence_interval(order_param_X_deim-1.96*sqrt(var_order_param_X_deim/size(tau_all,2)),order_param_X_deim+1.96*sqrt(var_order_param_X_deim/size(tau_all,2)),time_deim(end-Nasympt:end),colors(2,:));hold on;
plot_confidence_interval(order_param_X_pod-1.96*sqrt(var_order_param_X_pod/size(tau_all,2)),order_param_X_pod+1.96*sqrt(var_order_param_X_pod/size(tau_all,2)),time_pod(end-Nasympt:end),colors(1,:));
plot_confidence_interval(order_param_X_full-1.96*sqrt(var_order_param_X_full/size(tau_all,2)),order_param_X_full+1.96*sqrt(var_order_param_X_full/size(tau_all,2)),time_full(end-Nasympt:end),[0 0 0]);
ylim([0 1]);
legend({'DEIM','POD','FOM'},'interpreter','latex');
xlabel('$t$','interpreter','latex'); ylabel('$R$','interpreter','latex');

    
%% simulate for one instance of the random frequency, but varying coupling 
NcouplK=10; NcouplL0=5;
K_all=linspace(0,K,NcouplK);
L0_all=linspace(0,L0,NcouplL0);
tau=tau_all(:,1);

%deim with snapshots
Pdeim=deim(U{4});
Pdeim_ext = 4*(Pdeim-1)+(1:4); Pdeim_ext=sort(Pdeim_ext(:));
PTU=U_pod(Pdeim_ext,:);
PTUinv=U_pod(Pdeim_ext,:)\eye(size(U_pod,2));
meanU_v=mean(U{4},1);
PTtau=tau(Pdeim);

%deim with rhs
% Pdeim = deim(Urhs_loc{4});
% for i=1:4, PTUinv_loc{i}=(U{i}'*Urhs_loc{i})*(Urhs_loc{i}(Pdeim,:)\eye(size(Urhs_loc{i},2))); end;  PTUinv(:,ids_deim)=blkdiag(PTUinv_loc{:});
% for i=1:4, PTU_loc{i}=U{i}(Pdeim,:); end; PTU(ids_deim,:)=blkdiag(PTU_loc{:});
% meanU_v=mean(U{4},1);

for k=1:NcouplK
    disp(k);
    for l=1:NcouplL0
        
        %full
        tic;
        [time_full,x_full] = RK3 (@(t,x) circadian_rhs(t,x,params,tau,K_all(k),L0_all(l)),[0 FinalTime], x0, dt);
        elapsed_time_full(k,l)=toc;
        [synchrony_full(:,k,l),sync_param_full(k,l),avg_gene_conc_full(:,k,l),spectral_ampl_fact_full(k,l),estimated_period_full(k,l),order_param_X_full(:,k,l)] = compute_QoIs(time_full,x_full,N,Nasympt,params.omega,L0_all(l));
        
        %POD
        tic;
        [time_pod,x_pod_proj] = RK3 (@(t,x) U_pod'*circadian_rhs(t,U_pod*x,params,tau,K_all(k),L0_all(l)),[0 FinalTime], U_pod'*x0, dt);
        x_pod=x_pod_proj*U_pod';
        elapsed_time_pod(k,l)=toc;
        [synchrony_pod(:,k,l),sync_param_pod(k,l),avg_gene_conc_pod(:,k,l),spectral_ampl_fact_pod(k,l),estimated_period_pod(k,l),order_param_X_pod(:,k,l)] = compute_QoIs(time_pod,x_pod,N,Nasympt,params.omega,L0_all(l));
        
        %DEIM
        tic;
        [time_deim,x_deim_proj]=RK3(@(t,x) PTUinv*circadian_rhs_colloc(t,x,params,PTtau,K_all(k),L0_all(l),PTU,meanU_v), [0 FinalTime], U_pod'*x0, dt);
        x_deim=x_deim_proj*U_pod';
        elapsed_time_deim(k,l)=toc;
        [synchrony_deim(:,k,l),sync_param_deim(k,l),avg_gene_conc_deim(:,k,l),spectral_ampl_fact_deim(k,l),estimated_period_deim(k,l),order_param_X_deim(:,k,l)] = compute_QoIs(time_deim,x_deim,N,Nasympt,params.omega,L0_all(l));

    end

end

%write points in meshgrid format
[tmp1,tmp2]=meshgrid(K_all,L0_all);
test_K=reshape(tmp1',[],1);
test_L0=reshape(tmp2',[],1);
%visualize quantities
[KK,LL0]=meshgrid(linspace(0,K_all(end),8*length(K_all)),linspace(0,L0_all(end),8*length(L0_all)));
%rho
figure;
subplot(2,2,1); sync_param_plot=griddata(test_K,test_L0,sync_param_full(:),KK,LL0); surf(KK,LL0,sync_param_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0.9 1]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$\rho (FOM)$','interpreter','latex');
subplot(2,2,2); sync_param_plot=griddata(test_K,test_L0,sync_param_pod(:),KK,LL0); surf(KK,LL0,sync_param_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0.9 1]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$\rho (POD)$','interpreter','latex');
subplot(2,2,3); sync_param_plot=griddata(test_K,test_L0,sync_param_deim(:),KK,LL0); surf(KK,LL0,sync_param_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0.9 1]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$\rho (DEIM)$','interpreter','latex');
%spectral amplification factor
figure;
subplot(2,2,1); amplif_fact_plot=griddata(test_K,test_L0,spectral_ampl_fact_full(:),KK,LL0); surf(KK,LL0,amplif_fact_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0 20]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$S (FOM)$','interpreter','latex');
subplot(2,2,2); amplif_fact_plot=griddata(test_K,test_L0,spectral_ampl_fact_pod(:),KK,LL0); surf(KK,LL0,amplif_fact_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0 20]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$S (POD)$','interpreter','latex');
subplot(2,2,3); amplif_fact_plot=griddata(test_K,test_L0,spectral_ampl_fact_deim(:),KK,LL0); surf(KK,LL0,amplif_fact_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0 20]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$S (DEIM)$','interpreter','latex');
%estimated period
figure;
subplot(2,2,1); estimated_period_plot=griddata(test_K,test_L0,estimated_period_full(:),KK,LL0); surf(KK,LL0,estimated_period_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([20 30]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$\bar{T} (FOM)$','interpreter','latex');
subplot(2,2,2); estimated_period_plot=griddata(test_K,test_L0,estimated_period_pod(:),KK,LL0); surf(KK,LL0,estimated_period_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([20 30]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$\bar{T} (POD)$','interpreter','latex');
subplot(2,2,3); estimated_period_plot=griddata(test_K,test_L0,estimated_period_deim(:),KK,LL0); surf(KK,LL0,estimated_period_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([20 30]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$\bar{T} (DEIM)$','interpreter','latex');
%order parameter
figure;
subplot(2,2,1); order_param_plot=griddata(test_K,test_L0,reshape(order_param_X_full(end,:,:),[],1),KK,LL0); surf(KK,LL0,order_param_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0 1]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$R (FOM)$','interpreter','latex');
subplot(2,2,2); order_param_plot=griddata(test_K,test_L0,reshape(order_param_X_pod(end,:,:),[],1),KK,LL0); surf(KK,LL0,order_param_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0 1]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$R (POD)$','interpreter','latex');
subplot(2,2,3); order_param_plot=griddata(test_K,test_L0,reshape(order_param_X_deim(end,:,:),[],1),KK,LL0); surf(KK,LL0,order_param_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0 1]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$R (DEIM)$','interpreter','latex');
%computational cost
figure;
max_cost=max([elapsed_time_full(:); elapsed_time_pod(:); elapsed_time_deim(:)]);
subplot(2,2,1); cost_plot=griddata(test_K,test_L0,elapsed_time_full(:),KK,LL0); surf(KK,LL0,cost_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0 max_cost]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$Cost (FOM)$','interpreter','latex');
subplot(2,2,2); cost_plot=griddata(test_K,test_L0,elapsed_time_pod(:),KK,LL0); surf(KK,LL0,cost_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0 max_cost]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$Cost (POD)$','interpreter','latex');
subplot(2,2,3); cost_plot=griddata(test_K,test_L0,elapsed_time_deim(:),KK,LL0); surf(KK,LL0,cost_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0 max_cost]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$Cost (DEIM)$','interpreter','latex');


%% simulate for all instances of the random frequency AND varying coupling 
%this is similar to generation of the training set and it is a combination
%of the two previous parts that have either K fixed or omega fixed

NcouplK=10; NcouplL0=5;
K_all=linspace(0,K,NcouplK);
L0_all=linspace(0,L0,NcouplL0);

%deim with snapshots
Pdeim=deim(U{4});
Pdeim_ext = 4*(Pdeim-1)+(1:4); Pdeim_ext=sort(Pdeim_ext(:));
PTU=U_pod(Pdeim_ext,:);
PTUinv=U_pod(Pdeim_ext,:)\eye(size(U_pod,2));
meanU_v=mean(U{4},1);

%deim with rhs
% Pdeim = deim(Urhs_loc{4});
% for i=1:4, PTUinv_loc{i}=(U{i}'*Urhs_loc{i})*(Urhs_loc{i}(Pdeim,:)\eye(size(Urhs_loc{i},2))); end;  PTUinv(:,ids_deim)=blkdiag(PTUinv_loc{:});
% for i=1:4, PTU_loc{i}=U{i}(Pdeim,:); end; PTU(ids_deim,:)=blkdiag(PTU_loc{:});
% meanU_v=mean(U{4},1);

%if true, then I test with 50 iid samples, independent of training
if MC_test
    rng('default'); tau_all=a+(b-a)*rand(N,50);
end

for k=1:NcouplK
    disp(-k);
    for l=1:NcouplL0
        disp(-l);
        
        for i=1:size(tau_all,2)
            tau=tau_all(:,i);
            PTtau=tau(Pdeim);
        
            %full
            tic;
            [time_full,x_full] = RK3 (@(t,x) circadian_rhs(t,x,params,tau,K_all(k),L0_all(l)),[0 FinalTime], x0, dt);
            elapsed_time_full(k,l)=toc;
            [synchrony_full_test(:,k,l,i),sync_param_full_test(k,l,i),avg_gene_conc_full_test(:,k,l,i),spectral_ampl_fact_full_test(k,l,i),estimated_period_full_test(k,l,i),order_param_X_full_test(:,k,l,i)] = compute_QoIs(time_full,x_full,N,Nasympt,params.omega,L0_all(l));
            
            %POD
            tic;
            [time_pod,x_pod_proj] = RK3 (@(t,x) U_pod'*circadian_rhs(t,U_pod*x,params,tau,K_all(k),L0_all(l)),[0 FinalTime], U_pod'*x0, dt);
            x_pod=x_pod_proj*U_pod';
            elapsed_time_pod(k,l)=toc;
            [synchrony_pod_test(:,k,l,i),sync_param_pod_test(k,l,i),avg_gene_conc_pod_test(:,k,l,i),spectral_ampl_fact_pod_test(k,l,i),estimated_period_pod_test(k,l,i),order_param_X_pod_test(:,k,l,i)] = compute_QoIs(time_pod,x_pod,N,Nasympt,params.omega,L0_all(l));
            
            %DEIM
            tic;
            [time_deim,x_deim_proj]=RK3(@(t,x) PTUinv*circadian_rhs_colloc(t,x,params,PTtau,K_all(k),L0_all(l),PTU,meanU_v), [0 FinalTime], U_pod'*x0, dt);
            x_deim=x_deim_proj*U_pod';
            elapsed_time_deim(k,l)=toc;
            [synchrony_deim_test(:,k,l,i),sync_param_deim_test(k,l,i),avg_gene_conc_deim_test(:,k,l,i),spectral_ampl_fact_deim_test(k,l,i),estimated_period_deim_test(k,l,i),order_param_X_deim_test(:,k,l,i)] = compute_QoIs(time_deim,x_deim,N,Nasympt,params.omega,L0_all(l));

        end
        
        %compute means (I am not plotting confidence intervals here, so variances are not necessary)
        sync_param_full=mean(sync_param_full_test,3); sync_param_pod=mean(sync_param_pod_test,3); sync_param_deim=mean(sync_param_deim_test,3);
        spectral_ampl_fact_full=mean(spectral_ampl_fact_full_test,3); spectral_ampl_fact_pod=mean(spectral_ampl_fact_pod_test,3); spectral_ampl_fact_deim=mean(spectral_ampl_fact_deim_test,3);
        estimated_period_full=mean(estimated_period_full_test,3); estimated_period_pod=mean(estimated_period_pod_test,3); estimated_period_deim=mean(estimated_period_deim_test,3);
        order_param_X_full=mean(order_param_X_full_test,4); order_param_X_pod=mean(order_param_X_pod_test,4); order_param_X_deim=mean(order_param_X_deim_test,4);
        
    end

end

%write points in meshgrid format
[tmp1,tmp2]=meshgrid(K_all,L0_all);
test_K=reshape(tmp1',[],1);
test_L0=reshape(tmp2',[],1);
%visualize quantities
[KK,LL0]=meshgrid(linspace(0,K_all(end),8*length(K_all)),linspace(0,L0_all(end),8*length(L0_all)));
%rho
figure;
sync_param_plot=griddata(test_K,test_L0,sync_param_full(:),KK,LL0); surf(KK,LL0,sync_param_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0.9 1]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$\rho (FOM)$','interpreter','latex');
figure;
sync_param_plot=griddata(test_K,test_L0,sync_param_pod(:),KK,LL0); surf(KK,LL0,sync_param_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0.9 1]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$\rho (POD)$','interpreter','latex');
figure;
sync_param_plot=griddata(test_K,test_L0,sync_param_deim(:),KK,LL0); surf(KK,LL0,sync_param_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0.9 1]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$\rho (DEIM)$','interpreter','latex');
%spectral amplification factor
figure;
amplif_fact_plot=griddata(test_K,test_L0,spectral_ampl_fact_full(:),KK,LL0); surf(KK,LL0,amplif_fact_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0 20]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$S (FOM)$','interpreter','latex');
figure;
amplif_fact_plot=griddata(test_K,test_L0,spectral_ampl_fact_pod(:),KK,LL0); surf(KK,LL0,amplif_fact_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0 20]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$S (POD)$','interpreter','latex');
figure;
amplif_fact_plot=griddata(test_K,test_L0,spectral_ampl_fact_deim(:),KK,LL0); surf(KK,LL0,amplif_fact_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0 20]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$S (DEIM)$','interpreter','latex');
%estimated period
figure;
estimated_period_plot=griddata(test_K,test_L0,estimated_period_full(:),KK,LL0); surf(KK,LL0,estimated_period_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([20 30]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$\bar{T} (FOM)$','interpreter','latex');
figure;
estimated_period_plot=griddata(test_K,test_L0,estimated_period_pod(:),KK,LL0); surf(KK,LL0,estimated_period_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([20 30]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$\bar{T} (POD)$','interpreter','latex');
figure;
estimated_period_plot=griddata(test_K,test_L0,estimated_period_deim(:),KK,LL0); surf(KK,LL0,estimated_period_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([20 30]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$\bar{T} (DEIM)$','interpreter','latex');
%order parameter
figure;
order_param_plot=griddata(test_K,test_L0,reshape(order_param_X_full(end,:,:),[],1),KK,LL0); surf(KK,LL0,order_param_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0 1]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$R (FOM)$','interpreter','latex');
figure;
order_param_plot=griddata(test_K,test_L0,reshape(order_param_X_pod(end,:,:),[],1),KK,LL0); surf(KK,LL0,order_param_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0 1]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$R (POD)$','interpreter','latex');
figure;
order_param_plot=griddata(test_K,test_L0,reshape(order_param_X_deim(end,:,:),[],1),KK,LL0); surf(KK,LL0,order_param_plot); colormap jet; colorbar; view(0,90); shading interp; caxis([0 1]); xlabel('$K$','interpreter','latex'); ylabel('$L_0$','interpreter','latex'); title('$R (DEIM)$','interpreter','latex');


%% behavior varying reduced dimension using single instances
latent_dim=unique(round(logspace(0,log10(N),20)));

K=K_all(end);
L0=L0_all(end);
tau=tau_all(:,1);

for k=1:length(latent_dim)
    disp(k);
    
    %full
    tic;
    [time_full,x_full] = RK3 (@(t,x) circadian_rhs(t,x,params,tau,K,L0),[0 FinalTime], x0, dt);
    elapsed_time_full_k(k)=toc;
    [synchrony_full_k(:,k),sync_param_full_k(k),avg_gene_conc_full_k(:,k),spectral_ampl_fact_full_k(k),estimated_period_full_k(k),order_param_X_full_k(:,k)] = compute_QoIs(time_full,x_full,N,Nasympt,params.omega,L0);
    
    %quantities for reduced models
    for i=1:4
        [U{i},S{i},~]=svd(data(:,i:4:end-4+i).','econ'); U{i}=U{i}(:,1:latent_dim(k)); S{i}=diag(S{i});
    end
    clear U_pod;
    U_pod(ids,:)=(blkdiag(U{:}));
    
    Pdeim=deim(U{4});
    Pdeim_ext = 4*(Pdeim-1)+(1:4); Pdeim_ext=sort(Pdeim_ext(:));
    PTU=U_pod(Pdeim_ext,:);
    PTUinv=U_pod(Pdeim_ext,:)\eye(size(U_pod,2));
    meanU_v=mean(U{4},1);
    PTtau=tau(Pdeim);
   
    %for i=1:4
    %    [Urhs_loc{i},S{i},~]=svd(data_rhs(:,i:4:end-4+i).','econ'); Urhs_loc{i}=Urhs_loc{i}(:,1:latent_dim(k)); Srhs{i}=diag(S{i});
    %end
    %clear Urhs;
    %clear PTUinv PTU;
    %Urhs(ids,:)=(blkdiag(Urhs_loc{:}));
    %Pdeim = deim(Urhs_loc{4});
    %ids_pod=[1:4:4*n_latent_pod-3, 2:4:4*n_latent_pod-2, 3:4:4*n_latent_pod-1, 4:4:4*n_latent_pod];
    %ids_deim=[1:4:4*latent_dim(k)-3, 2:4:4*latent_dim(k)-2, 3:4:4*latent_dim(k)-1, 4:4:4*latent_dim(k)];
    %Pdeim_ext = 4*(Pdeim-1)+(1:4); Pdeim_ext=sort(Pdeim_ext(:));
    %for i=1:4, PTUinv_loc{i}=(U{i}'*Urhs_loc{i})*(Urhs_loc{i}(Pdeim,:)\eye(size(Urhs_loc{i},2))); end;  PTUinv(:,ids_deim)=blkdiag(PTUinv_loc{:});
    %for i=1:4, PTU_loc{i}=U{i}(Pdeim,:); end; PTU(ids_deim,:)=blkdiag(PTU_loc{:});
    %meanU_v=mean(U{4},1);
    %PTtau=tau(Pdeim);


    %POD
    tic;
    [time_pod,x_pod_proj] = RK3 (@(t,x) U_pod'*circadian_rhs(t,U_pod*x,params,tau,K,L0),[0 FinalTime], U_pod'*x0, dt);
    x_pod=x_pod_proj*U_pod';
    elapsed_time_pod_k(k)=toc;
    [synchrony_pod_k(:,k),sync_param_pod_k(k),avg_gene_conc_pod_k(:,k),spectral_ampl_fact_pod_k(k),estimated_period_pod_k(k),order_param_X_pod_k(:,k)] = compute_QoIs(time_pod,x_pod,N,Nasympt,params.omega,L0);
    
    %DEIM
    tic;
    [time_deim,x_deim_proj]=RK3(@(t,x) PTUinv*circadian_rhs_colloc(t,x,params,PTtau,K,L0,PTU,meanU_v), [0 FinalTime], U_pod'*x0, dt);
    x_deim=x_deim_proj*U_pod';
    elapsed_time_deim_k(k)=toc;
    [synchrony_deim_k(:,k),sync_param_deim_k(k),avg_gene_conc_deim_k(:,k),spectral_ampl_fact_deim_k(k),estimated_period_deim_k(k),order_param_X_deim_k(:,k)] = compute_QoIs(time_deim,x_deim,N,Nasympt,params.omega,L0);

end

figure;
loglog(latent_dim,elapsed_time_full_k,'color',[0 0 0]); hold on;
loglog(latent_dim,elapsed_time_pod_k,'color',colors(1,:)); 
loglog(latent_dim,elapsed_time_deim_k,'color',colors(2,:)); 
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$k/4$','interpreter','latex'); ylabel('$Cost$','interpreter','latex');
figure;
subplot(2,1,1);
semilogx(latent_dim,order_param_X_full_k(end,:),'color',[0 0 0]); hold on;
semilogx(latent_dim,order_param_X_pod_k(end,:),'color',colors(1,:)); 
semilogx(latent_dim,order_param_X_deim_k(end,:),'color',colors(2,:)); 
ylim([0 1]);
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$k/4$','interpreter','latex'); ylabel('$R$','interpreter','latex');
subplot(2,1,2);
loglog(latent_dim,abs(order_param_X_pod_k(end,:)-order_param_X_full_k(end,:))./abs(order_param_X_full_k(end,:))+1e-16,'color',colors(1,:)); hold on;
loglog(latent_dim,abs(order_param_X_deim_k(end,:)-order_param_X_full_k(end,:))./abs(order_param_X_full_k(end,:))+1e-16,'color',colors(2,:)); 
ylim([0 1]);
legend({'POD','DEIM'},'interpreter','latex');
xlabel('$k/4$','interpreter','latex'); ylabel('$|R_{FOM}-R_{ROM}|/|R_{FOM}|$','interpreter','latex');
figure;
loglog(mean(elapsed_time_full_k),1e-16,'o','color',[0 0 0]); hold on;
loglog(elapsed_time_pod_k,abs(order_param_X_pod_k(end,:)-order_param_X_full_k(end,:))./abs(order_param_X_full_k(end,:))+1e-16,'o--','color',colors(1,:)); 
loglog(elapsed_time_deim_k,abs(order_param_X_deim_k(end,:)-order_param_X_full_k(end,:))./abs(order_param_X_full_k(end,:))+1e-16,'o--','color',colors(2,:)); 
xlabel('$Cost$','interpreter','latex'); ylabel('$Err$','interpreter','latex');
legend({'FOM','POD','DEIM'},'interpreter','latex');


%% behavior varying dimension of the full problem

full_dim=[20; 50; 100; 200; 500; 1000];
reduced_dim=10*ones(size(full_dim));

NcouplK=5;
NcouplL0=3;
K_all=linspace(0,K,NcouplK);
L0_all=linspace(0,L0,NcouplL0);

for nn=1:length(full_dim)
    disp(nn);
    
    %construct training set and svd
    %initial condition
    x0=repmat([0.0998 0.2468 2.0151 0.0339].',full_dim(nn),1);
    %natural frequencies (uniform distribution)
    tau_all=a+(b-a)*rand(full_dim(nn),Ntrials); 
    %tau_all=sort(tau_all,1);
    
    data=[];
    data_rhs=[];
    for k=1:NcouplK
        for l=1:NcouplL0
            for i=1:Ntrials
                [t_train,x_train]=RK3 (@(t,x) circadian_rhs(t,x,params,tau_all(:,i),K_all(k),L0_all(l)),[0 FinalTime], x0, dt);
                data=[data; x_train];
                
                x_rhs=zeros(size(x_train));
                for t=1:length(t_train)
                    x_rhs(t,:)=circadian_rhs(t_train(t),x_train(t,:)',params,tau_all(:,i),K_all(k),L0_all(l))';
                end
                data_rhs=[data_rhs; x_rhs];
            end
        end
    end
    for i=1:4
        [U{i},S{i},~]=svd(data(:,i:4:end-4+i).','econ'); U{i}=U{i}(:,1:n_latent_pod); S{i}=diag(S{i});
    end
    clear U_pod;
    ids=[1:4:4*full_dim(nn)-3, 2:4:4*full_dim(nn)-2, 3:4:4*full_dim(nn)-1, 4:4:4*full_dim(nn)];
    U_pod(ids,:)=(blkdiag(U{:}));
    
    
    %simulate models for one instance of the parameters
    K=K_all(end);
    L0=L0_all(end);
    tau=tau_all(:,1);
    
    %full
    tic;
    [time_full,x_full] = RK3 (@(t,x) circadian_rhs(t,x,params,tau,K,L0),[0 FinalTime], x0, dt);
    elapsed_time_full_n(nn)=toc;
    [synchrony_full_n(:,nn),sync_param_full_n(nn),avg_gene_conc_full_n(:,nn),spectral_ampl_fact_full_n(nn),estimated_period_full_n(nn),order_param_X_full_n(:,nn)] = compute_QoIs(time_full,x_full,full_dim(nn),Nasympt,params.omega,L0);

    %quantities for deim model (svd is already done)
    Pdeim=deim(U{4});
    Pdeim_ext = 4*(Pdeim-1)+(1:4); Pdeim_ext=sort(Pdeim_ext(:));
    PTU=U_pod(Pdeim_ext,:);
    PTUinv=U_pod(Pdeim_ext,:)\eye(size(U_pod,2));
    meanU_v=mean(U{4},1);
    PTtau=tau(Pdeim);
    
    %for i=1:4
    %    [Urhs_loc{i},S{i},~]=svd(data_rhs(:,i:4:end-4+i).','econ'); Urhs_loc{i}=Urhs_loc{i}(:,1:reduced_dim(nn)); Srhs{i}=diag(S{i});
    %end
    %clear Urhs PTUinv PTU;
    %Urhs(ids,:)=(blkdiag(Urhs_loc{:}));
    %Pdeim = deim(Urhs_loc{4});
    %ids_pod=[1:4:4*n_latent_pod-3, 2:4:4*n_latent_pod-2, 3:4:4*n_latent_pod-1, 4:4:4*n_latent_pod];
    %ids_deim=[1:4:4*reduced_dim(nn)-3, 2:4:4*reduced_dim(nn)-2, 3:4:4*reduced_dim(nn)-1, 4:4:4*reduced_dim(nn)];
    %Pdeim_ext = 4*(Pdeim-1)+(1:4); Pdeim_ext=sort(Pdeim_ext(:));
    %for i=1:4, PTUinv_loc{i}=(U{i}'*Urhs_loc{i})*(Urhs_loc{i}(Pdeim,:)\eye(size(Urhs_loc{i},2))); end;  PTUinv(:,ids_deim)=blkdiag(PTUinv_loc{:});
    %for i=1:4, PTU_loc{i}=U{i}(Pdeim,:); end; PTU(ids_deim,:)=blkdiag(PTU_loc{:});
    %meanU_v=mean(U{4},1);
    %PTtau=tau(Pdeim);
    
    %POD
    tic;
    [time_pod,x_pod_proj] = RK3 (@(t,x) U_pod'*circadian_rhs(t,U_pod*x,params,tau,K,L0),[0 FinalTime], U_pod'*x0, dt);
    x_pod=x_pod_proj*U_pod';
    elapsed_time_pod_n(nn)=toc;
    [synchrony_pod_n(:,nn),sync_param_pod_n(nn),avg_gene_conc_pod_n(:,nn),spectral_ampl_fact_pod_n(nn),estimated_period_pod_n(nn),order_param_X_pod_n(:,nn)] = compute_QoIs(time_pod,x_pod,full_dim(nn),Nasympt,params.omega,L0);
    
    %DEIM
    tic;
    [time_deim,x_deim_proj]=RK3(@(t,x) PTUinv*circadian_rhs_colloc(t,x,params,PTtau,K,L0,PTU,meanU_v), [0 FinalTime], U_pod'*x0, dt);
    x_deim=x_deim_proj*U_pod';
    elapsed_time_deim_n(nn)=toc;
    [synchrony_deim_n(:,nn),sync_param_deim_n(nn),avg_gene_conc_deim_n(:,nn),spectral_ampl_fact_deim_n(nn),estimated_period_deim_n(nn),order_param_X_deim_n(:,nn)] = compute_QoIs(time_deim,x_deim,full_dim(nn),Nasympt,params.omega,L0);



end

figure;
loglog(full_dim,elapsed_time_full_n,'color',[0 0 0]); hold on;
loglog(full_dim,elapsed_time_pod_n,'color',colors(1,:)); 
loglog(full_dim,elapsed_time_deim_n,'color',colors(2,:)); 
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$n/4$','interpreter','latex'); ylabel('$Cost$','interpreter','latex'); xlim([min(full_dim) max(full_dim)]);
figure;
subplot(2,1,1);
semilogx(full_dim,order_param_X_full_n(end,:),'color',[0 0 0]); hold on;
semilogx(full_dim,order_param_X_pod_n(end,:),'color',colors(1,:)); 
semilogx(full_dim,order_param_X_deim_n(end,:),'color',colors(2,:)); 
ylim([0 1]);
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$n/4$','interpreter','latex'); ylabel('$R$','interpreter','latex'); xlim([min(full_dim) max(full_dim)]);
subplot(2,1,2);
loglog(full_dim,abs(order_param_X_pod_n(end,:)-order_param_X_full_n(end,:))./abs(order_param_X_full_n(end,:))+1e-16,'color',colors(1,:)); hold on;
loglog(full_dim,abs(order_param_X_deim_n(end,:)-order_param_X_full_n(end,:))./abs(order_param_X_full_n(end,:))+1e-16,'color',colors(2,:)); 
ylim([0 1]);
legend({'POD','DEIM'},'interpreter','latex');
xlabel('$n/4$','interpreter','latex'); ylabel('$|R_{FOM}-R_{ROM}|/|R_{FOM}|$','interpreter','latex'); xlim([min(full_dim) max(full_dim)]);
%xlabel('$n/4$','interpreter','latex'); ylabel('$Err$','interpreter','latex'); xlim([min(full_dim) max(full_dim)]);
figure;
loglog(elapsed_time_full_n,1e-16*ones(size(elapsed_time_full_n)),'o--','color',[0 0 0]); hold on;
loglog(elapsed_time_pod_n,abs(order_param_X_pod_n(end,:)-order_param_X_full_n(end,:))./abs(order_param_X_full_n(end,:))+1e-16,'o--','color',colors(1,:)); 
loglog(elapsed_time_deim_n,abs(order_param_X_deim_n(end,:)-order_param_X_full_n(end,:))./abs(order_param_X_full_n(end,:))+1e-16,'o--','color',colors(2,:)); 
xlabel('$Cost$','interpreter','latex'); ylabel('$Err$','interpreter','latex');
legend({'FOM','POD','DEIM'},'interpreter','latex');



%% functions to compute the rhs and quantities of interest

function [out] = circadian_rhs(t,x,params,tau,K,L0)

    %extract variables
    X=x(1:4:end-3,:); Y=x(2:4:end-2,:); Z=x(3:4:end-1,:); V=x(4:4:end,:); F=mean(V,1);
    
    out=zeros(size(x));
    %x_dot
    out(1:4:end-3)=params.nu(1)*params.Ki(1)^4./(params.Ki(1)^4+Z.^4)-params.nu(2)*X./(params.Ki(2)+X)+params.nuc*K*F./(params.Kc+K*F)+L0/2*(1+sin(params.omega*t));
    %y_dot
    out(2:4:end-2)=params.k(3)*X-params.nu(4)*Y./(params.Ki(4)+Y);
    %z_dot
    out(3:4:end-1)=params.k(5)*Y-params.nu(6)*Z./(params.Ki(6)+Z);
    %v_dot
    out(4:4:end)=params.k(7)*X-params.nu(8)*V./(params.Ki(8)+V);
    %divide rhs by frequency of each oscillator (same quantity for all variables)
    out=out./repelem(tau,4,1);

end

function out=circadian_rhs_colloc(t,x,params,tau,K,L0,U,meanU_v)

    %compute reconstruction for deim indexes only
    reconstr=U*x;
    X=reconstr(1:4:end-3,:); Y=reconstr(2:4:end-2,:); Z=reconstr(3:4:end-1,:); V=reconstr(4:4:end,:); 
    
    %coupling term depends on F and can be precomputed
    F=meanU_v*x(3*length(meanU_v)+1:4*length(meanU_v));
    
    %define collocated output
    out=zeros(size(U,1),1);
    out(1:4:end-3)=params.nu(1)*params.Ki(1)^4./(params.Ki(1)^4+Z.^4)-params.nu(2)*X./(params.Ki(2)+X)+params.nuc*K*F./(params.Kc+K*F)+L0/2*(1+sin(params.omega*t));
    out(2:4:end-2)=params.k(3)*X-params.nu(4)*Y./(params.Ki(4)+Y);
    out(3:4:end-1)=params.k(5)*Y-params.nu(6)*Z./(params.Ki(6)+Z);
    out(4:4:end)=params.k(7)*X-params.nu(8)*V./(params.Ki(8)+V);
    out=out./repelem(tau,4,1);
    
end

function [synchrony,sync_param,avg_gene_conc,spectral_ampl_fact,estimated_period,order_param_X] = compute_QoIs(t,x,N,Nasympt,omega,L0)

    %F(t)^2/expectation(V_i(t)^2)
    V=x(:,4:4:end);
    synchrony=mean(V,2).^2./mean(V.^2,2);
    %sqrt of time-averaged synchrony variable
    sync_param=sqrt(mean(synchrony(end-Nasympt:end)));
    
    %expectation(X_i(t))
    X=x(:,1:4:end-3);
    avg_gene_conc=mean(X,2);
    %time avg of exp(-i*omega*t)*avg_gene_conc, absolute value squared and normalized by 4/L0^2
    spectral_ampl_fact=4/L0^2*abs(mean(exp(-1i*omega*t(end-Nasympt:end)).*avg_gene_conc(end-Nasympt:end)))^2;
    %find period of the avg_gene_conc variable
    [peaks,locs]=findpeaks(avg_gene_conc(end-Nasympt:end));
    estimated_period=mean(diff(t(locs)));
    %find period for each oscillator, using the variable X_i
    for i=1:N
        %X_i
        X=x(:,1:4:end-3);
        %estimate period
        [peaks_i,locs_i]=findpeaks(X(end-Nasympt:end,i));
        %phase grows linearly in the period, and is equal to 2*k*pi at peaks
        estimated_phase(:,i)=interp1(t(end-Nasympt+locs_i),2*pi*(0:length(locs_i)-1)',t(end-Nasympt:end),'linear','extrap');
    end
    %kuramoto order parameter for the X_i variables
    order_param_X=abs(mean(exp(1i*estimated_phase),2));
end