clear all; colors=get(gca,'ColorOrder'); close all; clc;
set(0,'defaultAxesFontSize',20); set(0,'defaultLineLineWidth',2);

N=100;
FinalTime=200; 
Nsteps=10*FinalTime;
dt=FinalTime/Nsteps;

Ntrials=1;
Ncoupl=15;

n_latent_pod=10; %up to N
n_samples_deim=10;

MC_test=false; %true if I want to use 50 iid samples for testing, independently of the training

phi0=linspace(0,1*pi,N).';
y0=[cos(phi0); sin(phi0)];

rng('default');
%uniform distribution
K=0.15; a=0.97; b=1.03; omega_all=a+(b-a)*rand(N,Ntrials); Kc=2/(pi*1/(b-a)); Rsqrt=@(K) nan(size(K));
%K=10; a=-1; b=1; omega_all=a+(b-a)*rand(N,Ntrials); Kc=2/(pi*1/(b-a)); Rsqrt=@(K) nan(size(K));

%normal distribution
%K=0.3; mu=1; sig=0.02; omega_all=mu+sig*normrnd(0,1,[N,Ntrials]); Kc=2/(pi*1/sqrt(2*pi*sig^2)); Rsqrt=@(K) sqrt(16/(pi*Kc^3))*sqrt(1/(1/sqrt(2*pi*sig^2)*1/sig^2))*sqrt((K-Kc)/Kc);

%cauchy distribution
%K=0.4; omega_0=1; gam=0.015; omega_all = omega_0+gam*tan(pi*(rand(N,Ntrials)-1/2)); Kc=2*gam; Rsqrt=@(K) sqrt(1-Kc./K);

K_all=linspace(0,K,Ncoupl);
omega_all=omega_all-mean(omega_all,1);

%omega_all=sort(omega_all,1);

%function
%see end of script


%% construct training set
data=zeros(Ntrials*Ncoupl*(Nsteps+1),2*N);
data_rhs=zeros(Ntrials*Ncoupl*(Nsteps+1),2*N);
for k=1:Ncoupl
    disp(k);
    for i=1:Ntrials
        [t_train,y_train]=RK3 (@(t,y) rhs_fun(t,y,omega_all(:,i),K_all(k),N),[0 FinalTime], y0, dt);
        data((k-1)*(Nsteps+1)*Ntrials+(Nsteps+1)*(i-1)+1:(k-1)*(Nsteps+1)*Ntrials+(Nsteps+1)*i,:)=y_train;
        %data=[data; y_train];
        
        y_rhs=zeros(size(y_train));
        for t=1:length(t_train)
            y_rhs(t,:)=rhs_fun(t_train(t),y_train(t,:)',omega_all(:,i),K_all(k),N)';
        end
        data_rhs((k-1)*(Nsteps+1)*Ntrials+(Nsteps+1)*(i-1)+1:(k-1)*(Nsteps+1)*Ntrials+(Nsteps+1)*i,:)=y_rhs;
        %data_rhs=[data_rhs; y_rhs];
    end
end

%compute svd
[U,S,~]=svd(data(:,1:end/2).','econ'); U=U(:,1:n_latent_pod); S=diag(S); 
[V,~,~]=svd(data(:,end/2+1:end).','econ'); V=V(:,1:n_latent_pod); 
U_pod=blkdiag(U,V);

%compute rsvd
%[U,S,~]=rsvd(data(:,1:end/2).',n_latent_pod); U=U(:,1:n_latent_pod); S=diag(S); 
%[V,~,~]=rsvd(data(:,end/2+1:end).',n_latent_pod); V=V(:,1:n_latent_pod); 
%U_pod=blkdiag(U,V);

%check decay of singular values
figure;
semilogy(1:length(S),S/S(1),'color',colors(1,:));
xlabel('$i$','interpreter','latex'); ylabel('$\sigma_i/\sigma_1$','interpreter','latex');

%show distribution of order parameters
%phi_train=unwrap(mod(atan2(data(:,end/2+1:end),data(:,1:end/2)),2*pi),[],1);
%order_train=abs(mean(exp(1i*phi_train),2));
%figure;
%histogram(order_train,'Normalization','probability');


%% simulate for one instance of the natural frequency
omega=omega_all(:,1);
K=K_all(end);

%full model
tic;
[time_full,y_full] = RK3 (@(t,y) rhs_fun(t,y,omega,K,N),[0 FinalTime], y0, dt);
elapsed_full=toc;
phi_full=unwrap(mod(atan2(y_full(:,end/2+1:end),y_full(:,1:end/2)),2*pi),[],1);
order_full=abs(mean(exp(1i*phi_full),2));

%POD
tic;
[time_pod,y_pod_proj] = RK3 (@(t,y) U_pod'*rhs_fun(t,U_pod*y,omega,K,N),[0 FinalTime], U_pod'*y0, dt);
y_pod=y_pod_proj*U_pod';
elapsed_pod=toc;
phi_pod=unwrap(mod(atan2(y_pod(:,end/2+1:end),y_pod(:,1:end/2)),2*pi),[],1);
order_pod=abs(mean(exp(1i*phi_pod),2));

%DEIM with snapshots
Pdeim = deim(U); 
PTUinv=blkdiag(U(Pdeim,:)\eye(size(U,2)),V(Pdeim,:)\eye(size(V,2))); 
PTU=U(Pdeim,:); PTV=V(Pdeim,:);
sumU=sum(U,1); sumV=sum(V,1);
PTomega=omega(Pdeim);

%DEIM with rhs
%[Urhs,Srhs,~]=svd(data_rhs(:,1:end/2).','econ'); Urhs=Urhs(:,1:n_samples_deim); Srhs=diag(Srhs); 
%[Vrhs,~,~]=svd(data_rhs(:,end/2+1:end).','econ'); Vrhs=Vrhs(:,1:n_samples_deim);
%figure;
%semilogy(1:length(Srhs),Srhs/Srhs(1)); hold on;
%xlabel('$i$','interpreter','latex'); ylabel('$\sigma_i/\sigma_1(RHS)$','interpreter','latex');
%Pdeim = deim(Urhs);
%PTUinv=blkdiag( (U'*Urhs)*(Urhs(Pdeim,:)\eye(size(Urhs,2))), (V'*Vrhs)*(Vrhs(Pdeim,:)\eye(size(Vrhs,2))) );
%PTU=U(Pdeim,:); PTV=V(Pdeim,:);
%sumU=sum(U,1); sumV=sum(V,1);
%PTomega=omega(Pdeim);

tic;
[time_deim,y_deim_proj]=RK3(@(t,y) PTUinv*rhs_fun_colloc(t,y,PTomega,K,N,PTU,sumU,U,PTV,sumV,V,Pdeim), [0 FinalTime], U_pod'*y0, dt);
y_deim=y_deim_proj*U_pod';
elapsed_deim=toc;
phi_deim=unwrap(mod(atan2(y_deim(:,end/2+1:end),y_deim(:,1:end/2)),2*pi),[],1);
order_deim=abs(mean(exp(1i*phi_deim),2));

%visualize original and reconstructed solution
Noscil_plot=floor(N/5);
figure;
subplot(3,1,1); plot(time_full,y_full(:,[1:Noscil_plot:N N+1:Noscil_plot:2*N]),'color',[0 0 0]); hold on; xlabel('$t$','interpreter','latex'); ylabel('$(x_i,y_i)_{FOM}$','interpreter','latex');
%ylabel('FOM','interpreter','latex');
subplot(3,1,2); plot(time_pod,y_pod(:,[1:Noscil_plot:N N+1:Noscil_plot:2*N]),'color',colors(1,:)); xlabel('$t$','interpreter','latex'); ylabel('$(x_i,y_i)_{POD}$','interpreter','latex');
%ylabel('POD','interpreter','latex');
subplot(3,1,3); plot(time_deim,y_deim(:,[1:Noscil_plot:N N+1:Noscil_plot:2*N]),'color',colors(2,:)); xlabel('$t$','interpreter','latex'); ylabel('$(x_i,y_i)_{DEIM}$','interpreter','latex');
%ylabel('DEIM','interpreter','latex');

%visualize amplitude
figure;
plot(time_full,sqrt(y_full(:,1:end/2).^2+y_full(:,end/2+1:end).^2)-1,'color',[0 0 0]); hold on;
plot(time_pod,sqrt(y_pod(:,1:end/2).^2+y_pod(:,end/2+1:end).^2)-1,'color',colors(1,:)); 
plot(time_deim,sqrt(y_deim(:,1:end/2).^2+y_deim(:,end/2+1:end).^2)-1,'color',colors(2,:)); 
title({'FOM, POD, DEIM'},'interpreter','latex');
xlabel('$t$','interpreter','latex'); ylabel('$\sqrt{x_i^2+y_i^2}-1$','interpreter','latex');

%visualize original and reconstructed order parameters
figure;
plot(time_full,order_full,'color',[0 0 0]); hold on;
plot(time_pod,order_pod,'color',colors(1,:));
plot(time_deim,order_deim,'color',colors(2,:));
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$t$','interpreter','latex'); ylabel('$R$','interpreter','latex');

%display times
fprintf('Elapsed time for full model: %f\n',elapsed_full);
fprintf('Elapsed time for POD model: %f\n',elapsed_pod);
fprintf('Elapsed time for DEIM model: %f\n',elapsed_deim);


%% simulate for the full range of natural frequency, with fixed coupling
K=K_all(end);

%deim with snapshots
Pdeim = deim(U);
PTUinv=blkdiag(U(Pdeim,:)\eye(size(U,2)),V(Pdeim,:)\eye(size(V,2)));
PTU=U(Pdeim,:); PTV=V(Pdeim,:);
sumU=sum(U,1); sumV=sum(V,1);

%deim with rhs
%Pdeim = deim(Urhs);
%PTUinv=blkdiag( (U'*Urhs)*(Urhs(Pdeim,:)\eye(size(Urhs,2))), (V'*Vrhs)*(Vrhs(Pdeim,:)\eye(size(Vrhs,2))) );
%PTU=U(Pdeim,:); PTV=V(Pdeim,:);
%sumU=sum(U,1); sumV=sum(V,1);


%if true, then I test with 50 iid samples, independent of training
if MC_test
    rng('default'); omega_all=a+(b-a)*rand(N,50); omega_all=omega_all-mean(omega_all,1);
end

for i=1:size(omega_all,2)
    disp(i);

    %full model
    [time_full,y_full] = RK3 (@(t,y) rhs_fun(t,y,omega_all(:,i),K,N),[0 FinalTime], y0, dt);
    phi_full=unwrap(mod(atan2(y_full(:,end/2+1:end),y_full(:,1:end/2)),2*pi),[],1);
    order_full=abs(mean(exp(1i*phi_full),2));

    %POD
    [time_pod,y_pod_proj] = RK3 (@(t,y) U_pod'*rhs_fun(t,U_pod*y,omega_all(:,i),K,N),[0 FinalTime], U_pod'*y0, dt);
    y_pod=y_pod_proj*U_pod';
    phi_pod=unwrap(mod(atan2(y_pod(:,end/2+1:end),y_pod(:,1:end/2)),2*pi),[],1);
    order_pod=abs(mean(exp(1i*phi_pod),2));

    %DEIM
    PTomega=omega_all(Pdeim,i);
    [time_deim,y_deim_proj]=RK3(@(t,y) PTUinv*rhs_fun_colloc(t,y,PTomega,K,N,PTU,sumU,U,PTV,sumV,V,Pdeim), [0 FinalTime], U_pod'*y0, dt);
    y_deim=y_deim_proj*U_pod';
    phi_deim=unwrap(mod(atan2(y_deim(:,end/2+1:end),y_deim(:,1:end/2)),2*pi),[],1);
    order_deim=abs(mean(exp(1i*phi_deim),2));
    
    %save quantities
    order_full_rand(:,i)=order_full;
    order_pod_rand(:,i)=order_pod;
    order_deim_rand(:,i)=order_deim;
    
end

%compute mean and variance
mean_full=mean(order_full_rand,2); var_full=var(order_full_rand,1,2);
mean_pod=mean(order_pod_rand,2); var_pod=var(order_pod_rand,1,2);
mean_deim=mean(order_deim_rand,2); var_deim=var(order_deim_rand,1,2);

%visualize confidence interval for order parameter as fcn of time
figure;
plot_confidence_interval(mean_deim-1.96*sqrt(var_deim/size(omega_all,2)),mean_deim+1.96*sqrt(var_deim/size(omega_all,2)),time_deim,colors(2,:));hold on;
plot_confidence_interval(mean_pod-1.96*sqrt(var_pod/size(omega_all,2)),mean_pod+1.96*sqrt(var_pod/size(omega_all,2)),time_pod,colors(1,:));
plot_confidence_interval(mean_full-1.96*sqrt(var_full/size(omega_all,2)),mean_full+1.96*sqrt(var_full/size(omega_all,2)),time_full,[0 0 0]);
legend({'DEIM','POD','FOM'},'interpreter','latex');
xlabel('$t$','interpreter','latex'); ylabel('$R$','interpreter','latex');

    
%% simulate for one instance of the random frequency, but varying coupling 
Ncoupl=10;
K_all=linspace(0,K,Ncoupl);
omega=omega_all(:,1);

%DEIM with snapshots
Pdeim = deim(U); 
PTUinv=blkdiag(U(Pdeim,:)\eye(size(U,2)),V(Pdeim,:)\eye(size(V,2))); 
PTU=U(Pdeim,:); PTV=V(Pdeim,:);
sumU=sum(U,1); sumV=sum(V,1);
PTomega=omega(Pdeim);

%DEIM with rhs
%Pdeim = deim(Urhs);
%PTUinv=blkdiag( (U'*Urhs)*(Urhs(Pdeim,:)\eye(size(Urhs,2))), (V'*Vrhs)*(Vrhs(Pdeim,:)\eye(size(Vrhs,2))) );
%PTU=U(Pdeim,:); PTV=V(Pdeim,:);
%sumU=sum(U,1); sumV=sum(V,1);
%PTomega=omega(Pdeim);


for k=1:Ncoupl
    disp(k);
    
    %full
    tic;
    [time_full,y_full] = RK3 (@(t,y) rhs_fun(t,y,omega,K_all(k),N),[0 FinalTime], y0, dt);
    elapsed_time_full(k)=toc;
    phi_full=unwrap(mod(atan2(y_full(:,end/2+1:end),y_full(:,1:end/2)),2*pi),[],1);
    order_param_full(:,k)=abs(mean(exp(1i*phi_full),2));
    
    %POD
    tic;
    [time_pod,y_pod_proj] = RK3 (@(t,y) U_pod'*rhs_fun(t,U_pod*y,omega,K_all(k),N),[0 FinalTime], U_pod'*y0, dt);
    y_pod=y_pod_proj*U_pod';
    elapsed_time_pod(k)=toc;
    phi_pod=unwrap(mod(atan2(y_pod(:,end/2+1:end),y_pod(:,1:end/2)),2*pi),[],1);
    order_param_pod(:,k)=abs(mean(exp(1i*phi_pod),2));
    
    %DEIM
    tic;
    [time_deim,y_deim_proj]=RK3(@(t,y) PTUinv*rhs_fun_colloc(t,y,PTomega,K_all(k),N,PTU,sumU,U,PTV,sumV,V,Pdeim), [0 FinalTime], U_pod'*y0, dt);
    y_deim=y_deim_proj*U_pod';
    elapsed_time_deim(k)=toc;
    phi_deim=unwrap(mod(atan2(y_deim(:,end/2+1:end),y_deim(:,1:end/2)),2*pi),[],1);
    order_param_deim(:,k)=abs(mean(exp(1i*phi_deim),2));

end
figure;
plot(K_all,elapsed_time_full,'color',[0 0 0]); hold on;
plot(K_all,elapsed_time_pod,'color',colors(1,:)); 
plot(K_all,elapsed_time_deim,'color',colors(2,:)); 
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$K$','interpreter','latex'); ylabel('$Cost$','interpreter','latex');
figure;
plot(K_all,order_param_full(end,:),'color',[0 0 0]); hold on;
plot(K_all,order_param_pod(end,:),'color',colors(1,:)); 
plot(K_all,order_param_deim(end,:),'color',colors(2,:)); 
%plot(linspace(Kc,max(K_all),100),Rsqrt(linspace(Kc,max(K_all),100)),':','color',[0 0 0]);
ylim([0 1]);
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$K$','interpreter','latex'); ylabel('$R$','interpreter','latex');


%% simulate for all instances of the random frequency AND varying coupling 
%this is similar to generation of the training set and it is a combination
%of the two previous parts that have either K fixed or omega fixed
Ncoupl=10;
K_all=linspace(0,K,Ncoupl);

%DEIM with snapshots
Pdeim = deim(U); 
PTUinv=blkdiag(U(Pdeim,:)\eye(size(U,2)),V(Pdeim,:)\eye(size(V,2))); 
PTU=U(Pdeim,:); PTV=V(Pdeim,:);
sumU=sum(U,1); sumV=sum(V,1);

%DEIM with rhs
%Pdeim = deim(Urhs);
%PTUinv=blkdiag( (U'*Urhs)*(Urhs(Pdeim,:)\eye(size(Urhs,2))), (V'*Vrhs)*(Vrhs(Pdeim,:)\eye(size(Vrhs,2))) );
%PTU=U(Pdeim,:); PTV=V(Pdeim,:);
%sumU=sum(U,1); sumV=sum(V,1);


%if true, then I test with 50 iid samples, independent of training
if MC_test
    rng('default'); omega_all=a+(b-a)*rand(N,50); omega_all=omega_all-mean(omega_all,1);
end

for k=1:Ncoupl
    disp(k);
    
    for i=1:size(omega_all,2)
        omega=omega_all(:,i);
        PTomega=omega(Pdeim);
        
        %full
        tic;
        [time_full,y_full] = RK3 (@(t,y) rhs_fun(t,y,omega,K_all(k),N),[0 FinalTime], y0, dt);
        elapsed_time_full(k)=toc;
        phi_full=unwrap(mod(atan2(y_full(:,end/2+1:end),y_full(:,1:end/2)),2*pi),[],1);
        order_param_full(:,k,i)=abs(mean(exp(1i*phi_full),2));
        
        %POD
        tic;
        [time_pod,y_pod_proj] = RK3 (@(t,y) U_pod'*rhs_fun(t,U_pod*y,omega,K_all(k),N),[0 FinalTime], U_pod'*y0, dt);
        y_pod=y_pod_proj*U_pod';
        elapsed_time_pod(k)=toc;
        phi_pod=unwrap(mod(atan2(y_pod(:,end/2+1:end),y_pod(:,1:end/2)),2*pi),[],1);
        order_param_pod(:,k,i)=abs(mean(exp(1i*phi_pod),2));
        
        %DEIM
        tic;
        [time_deim,y_deim_proj]=RK3(@(t,y) PTUinv*rhs_fun_colloc(t,y,PTomega,K_all(k),N,PTU,sumU,U,PTV,sumV,V,Pdeim), [0 FinalTime], U_pod'*y0, dt);
        y_deim=y_deim_proj*U_pod';
        elapsed_time_deim(k)=toc;
        phi_deim=unwrap(mod(atan2(y_deim(:,end/2+1:end),y_deim(:,1:end/2)),2*pi),[],1);
        order_param_deim(:,k,i)=abs(mean(exp(1i*phi_deim),2));
        
    end

end

%compute averages over natural frequencies
order_param_full_coupl=mean(order_param_full(end,:,:),3);
order_param_full_coupl_var=var(order_param_full(end,:,:),0,3);
order_param_pod_coupl=mean(order_param_pod(end,:,:),3);
order_param_pod_coupl_var=var(order_param_full(end,:,:),0,3);
order_param_deim_coupl=mean(order_param_deim(end,:,:),3);
order_param_deim_coupl_var=var(order_param_full(end,:,:),0,3);

figure;
plot(K_all,order_param_full_coupl,'color',[0 0 0]); hold on;
plot(K_all,order_param_pod_coupl,'color',colors(1,:)); 
plot(K_all,order_param_deim_coupl,'color',colors(2,:)); 
ylim([0 1]);
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$K$','interpreter','latex'); ylabel('$R$','interpreter','latex');

figure;
plot_confidence_interval(order_param_deim_coupl-1.96*sqrt(order_param_deim_coupl_var/size(omega_all,2)),order_param_deim_coupl+1.96*sqrt(order_param_deim_coupl_var/size(omega_all,2)),K_all,colors(2,:)); hold on;
plot_confidence_interval(order_param_pod_coupl-1.96*sqrt(order_param_pod_coupl_var/size(omega_all,2)),order_param_pod_coupl+1.96*sqrt(order_param_pod_coupl_var/size(omega_all,2)),K_all,colors(1,:));
plot_confidence_interval(order_param_full_coupl-1.96*sqrt(order_param_full_coupl_var/size(omega_all,2)),order_param_full_coupl+1.96*sqrt(order_param_full_coupl_var/size(omega_all,2)),K_all,[0 0 0]);
legend({'DEIM','POD','FOM'},'interpreter','latex');
xlabel('$K$','interpreter','latex'); ylabel('$R$','interpreter','latex');

%% behavior varying reduced dimension using single instances
latent_dim=unique(round(logspace(0,log10(N),20)));

K=K_all(end);
omega=omega_all(:,1);

for k=1:length(latent_dim)
    disp(k);
    
    %full
    tic;
    [time_full,y_full] = RK3 (@(t,y) rhs_fun(t,y,omega,K,N),[0 FinalTime], y0, dt);
    elapsed_time_full_k(k)=toc;
    phi_full=unwrap(mod(atan2(y_full(:,end/2+1:end),y_full(:,1:end/2)),2*pi),[],1);
    order_param_full_k(:,k)=abs(mean(exp(1i*phi_full),2));
    amplitude_err_full_k(:,k)=max(abs(sqrt(y_full(:,1:end/2).^2+y_full(:,end/2+1:end).^2)-1),[],2);
    %for i=1:length(time_full), tmp=rhs_fun(time_full(i),y_full(i,:)',omega,K,N); der_y_full_cos(i,:)=tmp(1:end/2); der_y_full_sin(i,:)=tmp(end/2+1:end); end; amplitude_err_full_k(:,k)=max( abs( y_full(:,1:end/2).*der_y_full_cos+y_full(:,end/2+1:end).*der_y_full_sin ) ,[],2);
    
    %quantities for reduced models
    [U,S,~]=svd(data(:,1:end/2).','econ'); U=U(:,1:latent_dim(k)); S=diag(S); 
    [V,~,~]=svd(data(:,end/2+1:end).','econ'); V=V(:,1:latent_dim(k)); 
    U_pod=blkdiag(U,V); 
    
    Pdeim = deim(U); 
    PTUinv=blkdiag(U(Pdeim,:)\eye(size(U,2)),V(Pdeim,:)\eye(size(V,2))); 
    PTU=U(Pdeim,:); PTV=V(Pdeim,:);
    sumU=sum(U,1); sumV=sum(V,1);
    PTomega=omega(Pdeim);
    
    %[Urhs,Srhs,~]=svd(data_rhs(:,1:end/2).','econ'); Urhs=Urhs(:,1:latent_dim(k)); Srhs=diag(Srhs);
    %[Vrhs,~,~]=svd(data_rhs(:,end/2+1:end).','econ'); Vrhs=Vrhs(:,1:latent_dim(k));
    %Pdeim = deim(Urhs);
    %PTUinv=blkdiag( (U'*Urhs)*(Urhs(Pdeim,:)\eye(size(Urhs,2))), (V'*Vrhs)*(Vrhs(Pdeim,:)\eye(size(Vrhs,2))) );
    %PTU=U(Pdeim,:); PTV=V(Pdeim,:);
    %sumU=sum(U,1); sumV=sum(V,1);
    %PTomega=omega(Pdeim);

    %POD
    tic;
    [time_pod,y_pod_proj] = RK3 (@(t,y) U_pod'*rhs_fun(t,U_pod*y,omega,K,N),[0 FinalTime], U_pod'*y0, dt);
    y_pod=y_pod_proj*U_pod';
    elapsed_time_pod_k(k)=toc;
    phi_pod=unwrap(mod(atan2(y_pod(:,end/2+1:end),y_pod(:,1:end/2)),2*pi),[],1);
    order_param_pod_k(:,k)=abs(mean(exp(1i*phi_pod),2));
    amplitude_err_pod_k(:,k)=max(abs(sqrt(y_pod(:,1:end/2).^2+y_pod(:,end/2+1:end).^2)-1),[],2);
    %for i=1:length(time_pod), tmp=U_pod*U_pod'*rhs_fun(time_pod(i),y_pod(i,:)',omega,K,N); der_y_pod_cos(i,:)=tmp(1:end/2); der_y_pod_sin(i,:)=tmp(end/2+1:end); end; amplitude_err_pod_k(:,k)=max( abs( y_pod(:,1:end/2).*der_y_pod_cos+y_pod(:,end/2+1:end).*der_y_pod_sin ) ,[],2);
    
    %DEIM
    tic;
    [time_deim,y_deim_proj]=RK3(@(t,y) PTUinv*rhs_fun_colloc(t,y,PTomega,K,N,PTU,sumU,U,PTV,sumV,V,Pdeim), [0 FinalTime], U_pod'*y0, dt);
    y_deim=y_deim_proj*U_pod';    
    elapsed_time_deim_k(k)=toc;
    phi_deim=unwrap(mod(atan2(y_deim(:,end/2+1:end),y_deim(:,1:end/2)),2*pi),[],1);
    order_param_deim_k(:,k)=abs(mean(exp(1i*phi_deim),2));
    amplitude_err_deim_k(:,k)=max(abs(sqrt(y_deim(:,1:end/2).^2+y_deim(:,end/2+1:end).^2)-1),[],2);
    %for i=1:length(time_deim), tmp=U_pod*PTUinv*rhs_fun_colloc(time_deim(i),y_deim_proj(i,:)',PTomega,K,N,PTU,sumU,U,PTV,sumV,V,Pdeim); der_y_deim_cos(i,:)=tmp(1:end/2); der_y_deim_sin(i,:)=tmp(end/2+1:end); end; amplitude_err_deim_k(:,k)=max( abs( y_deim(:,1:end/2).*der_y_deim_cos+y_deim(:,end/2+1:end).*der_y_deim_sin ) ,[],2);

end

figure;
semilogy(latent_dim,max(amplitude_err_full_k,[],1),'color',[0 0 0]); hold on;
semilogy(latent_dim,max(amplitude_err_pod_k,[],1),'color',colors(1,:)); 
semilogy(latent_dim,max(amplitude_err_deim_k,[],1),'color',colors(2,:)); 
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$k/2$','interpreter','latex'); ylabel('$Err_{A}$','interpreter','latex');

figure;
loglog(latent_dim,elapsed_time_full_k,'color',[0 0 0]); hold on;
loglog(latent_dim,elapsed_time_pod_k,'color',colors(1,:)); 
loglog(latent_dim,elapsed_time_deim_k,'color',colors(2,:)); 
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$k/2$','interpreter','latex'); ylabel('$Cost$','interpreter','latex');
figure;
subplot(2,1,1);
semilogx(latent_dim,order_param_full_k(end,:),'color',[0 0 0]); hold on;
semilogx(latent_dim,order_param_pod_k(end,:),'color',colors(1,:)); 
semilogx(latent_dim,order_param_deim_k(end,:),'color',colors(2,:)); 
ylim([0 1]);
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$k/2$','interpreter','latex'); ylabel('$R$','interpreter','latex');
subplot(2,1,2);
loglog(latent_dim,abs(order_param_pod_k(end,:)-order_param_full_k(end,:))./abs(order_param_full_k(end,:))+1e-16,'color',colors(1,:)); hold on;
loglog(latent_dim,abs(order_param_deim_k(end,:)-order_param_full_k(end,:))./abs(order_param_full_k(end,:))+1e-16,'color',colors(2,:)); 
ylim([0 1]);
legend({'POD','DEIM'},'interpreter','latex');
xlabel('$k/2$','interpreter','latex'); ylabel('$|R_{FOM}-R_{ROM}|/|R_{FOM}|$','interpreter','latex');
figure;
loglog(mean(elapsed_time_full_k),1e-16,'o','color',[0 0 0]); hold on;
loglog(elapsed_time_pod_k,abs(order_param_pod_k(end,:)-order_param_full_k(end,:))./abs(order_param_full_k(end,:))+1e-16,'o--','color',colors(1,:));
loglog(elapsed_time_deim_k,abs(order_param_deim_k(end,:)-order_param_full_k(end,:))./abs(order_param_full_k(end,:))+1e-16,'o--','color',colors(2,:)); 
xlabel('$Cost$','interpreter','latex'); ylabel('$Err$','interpreter','latex');
legend({'FOM','POD','DEIM'},'interpreter','latex');


%% behavior varying dimension of the full problem
full_dim=[20; 50; 100; 200; 500; 1000; 2000; 4000];
reduced_dim=10*ones(size(full_dim));

Ncoupl=15;
K_all=linspace(0,K,Ncoupl);

for nn=1:length(full_dim)
    disp(nn);
    
    %construct training set and svd
    %initial condition
    phi0=linspace(0,1*pi,full_dim(nn)).'; y0=[cos(phi0); sin(phi0)];
    %natural frequencies (uniform distribution)
    omega_all=a+(b-a)*rand(full_dim(nn),Ntrials); omega_all=omega_all-mean(omega_all,1);
    %omega_all=sort(omega_all,1);
    
    data=[];
    data_rhs=[];
    for k=1:Ncoupl
        for i=1:Ntrials
            [t_train,y_train]=RK3 (@(t,y) rhs_fun(t,y,omega_all(:,i),K_all(k),full_dim(nn)),[0 FinalTime], y0, dt);
            data=[data; y_train];
            
            y_rhs=zeros(size(y_train));
            for t=1:length(t_train)
                y_rhs(t,:)=rhs_fun(t_train(t),y_train(t,:)',omega_all(:,i),K_all(k),full_dim(nn))';
            end
            data_rhs=[data_rhs; y_rhs];
        end
    end
    [U,S,~]=svd(data(:,1:end/2).','econ'); U=U(:,1:reduced_dim(nn)); S=diag(S); 
    [V,~,~]=svd(data(:,end/2+1:end).','econ'); V=V(:,1:reduced_dim(nn)); 
    U_pod=blkdiag(U,V);

    %simulate models for one instance of the parameters
    K=K_all(end);
    omega=omega_all(:,1);
    
    %full
    tic;
    [time_full,y_full] = RK3 (@(t,y) rhs_fun(t,y,omega,K,full_dim(nn)),[0 FinalTime], y0, dt);
    elapsed_time_full_n(nn)=toc;
    phi_full=unwrap(mod(atan2(y_full(:,end/2+1:end),y_full(:,1:end/2)),2*pi),[],1);
    order_param_full_n(:,nn)=abs(mean(exp(1i*phi_full),2));
    
    %quantities for deim model (svd is already done)
    
    Pdeim = deim(U); 
    PTUinv=blkdiag(U(Pdeim,:)\eye(size(U,2)),V(Pdeim,:)\eye(size(V,2))); 
    PTU=U(Pdeim,:); PTV=V(Pdeim,:);
    sumU=sum(U,1); sumV=sum(V,1);
    PTomega=omega(Pdeim);
    
    %[Urhs,Srhs,~]=svd(data_rhs(:,1:end/2).','econ'); Urhs=Urhs(:,1:reduced_dim(nn)); Srhs=diag(Srhs);
    %[Vrhs,~,~]=svd(data_rhs(:,end/2+1:end).','econ'); Vrhs=Vrhs(:,1:reduced_dim(nn));
    %Pdeim = deim(Urhs);
    %PTUinv=blkdiag( (U'*Urhs)*(Urhs(Pdeim,:)\eye(size(Urhs,2))), (V'*Vrhs)*(Vrhs(Pdeim,:)\eye(size(Vrhs,2))) );
    %PTU=U(Pdeim,:); PTV=V(Pdeim,:);
    %sumU=sum(U,1); sumV=sum(V,1);
    %PTomega=omega(Pdeim);

    %POD
    tic;
    [time_pod,y_pod_proj] = RK3 (@(t,y) U_pod'*rhs_fun(t,U_pod*y,omega,K,full_dim(nn)),[0 FinalTime], U_pod'*y0, dt);
    y_pod=y_pod_proj*U_pod';
    elapsed_time_pod_n(nn)=toc;
    phi_pod=unwrap(mod(atan2(y_pod(:,end/2+1:end),y_pod(:,1:end/2)),2*pi),[],1);
    order_param_pod_n(:,nn)=abs(mean(exp(1i*phi_pod),2));
    
    %DEIM
    tic;
    [time_deim,y_deim_proj]=RK3(@(t,y) PTUinv*rhs_fun_colloc(t,y,PTomega,K,full_dim(nn),PTU,sumU,U,PTV,sumV,V,Pdeim), [0 FinalTime], U_pod'*y0, dt);
    y_deim=y_deim_proj*U_pod';
    elapsed_time_deim_n(nn)=toc;
    phi_deim=unwrap(mod(atan2(y_deim(:,end/2+1:end),y_deim(:,1:end/2)),2*pi),[],1);
    order_param_deim_n(:,nn)=abs(mean(exp(1i*phi_deim),2));


end

figure;
loglog(full_dim,elapsed_time_full_n,'color',[0 0 0]); hold on;
loglog(full_dim,elapsed_time_pod_n,'color',colors(1,:)); 
loglog(full_dim,elapsed_time_deim_n,'color',colors(2,:)); 
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$n/2$','interpreter','latex'); ylabel('$Cost$','interpreter','latex'); xlim([min(full_dim) max(full_dim)]);
figure;
subplot(2,1,1);
semilogx(full_dim,order_param_full_n(end,:),'color',[0 0 0]); hold on;
semilogx(full_dim,order_param_pod_n(end,:),'color',colors(1,:)); 
semilogx(full_dim,order_param_deim_n(end,:),'color',colors(2,:)); 
ylim([0 1]);
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$n/2$','interpreter','latex'); ylabel('$R$','interpreter','latex'); xlim([min(full_dim) max(full_dim)]);
subplot(2,1,2);
loglog(full_dim,abs(order_param_pod_n(end,:)-order_param_full_n(end,:))./abs(order_param_full_n(end,:))+1e-16,'color',colors(1,:)); hold on;
loglog(full_dim,abs(order_param_deim_n(end,:)-order_param_full_n(end,:))./abs(order_param_full_n(end,:))+1e-16,'color',colors(2,:)); 
ylim([0 1]);
legend({'POD','DEIM'},'interpreter','latex');
xlabel('$n/2$','interpreter','latex'); ylabel('$|R_{FOM}-R_{ROM}|/|R_{FOM}|$','interpreter','latex'); xlim([min(full_dim) max(full_dim)]);
%xlabel('$n/2$','interpreter','latex'); ylabel('$Err$','interpreter','latex'); xlim([min(full_dim) max(full_dim)]);
figure;
loglog(elapsed_time_full_n,1e-16*ones(size(elapsed_time_full_n)),'o--','color',[0 0 0]); hold on;
loglog(elapsed_time_pod_n,abs(order_param_pod_n(end,:)-order_param_full_n(end,:))./abs(order_param_full_n(end,:))+1e-16,'o--','color',colors(1,:));
loglog(elapsed_time_deim_n,abs(order_param_deim_n(end,:)-order_param_full_n(end,:))./abs(order_param_full_n(end,:))+1e-16,'o--','color',colors(2,:)); 
xlabel('$Cost$','interpreter','latex'); ylabel('$Err$','interpreter','latex');
legend({'FOM','POD','DEIM'},'interpreter','latex');


%% functions to compute the rhs

function out = rhs_fun(t,y,omega,K,N) 

    rhs=omega+K/N*(sum(y(end/2+1:end),1)*y(1:end/2)-sum(y(1:end/2),1)*y(end/2+1:end));
    out=[-y(end/2+1:end); y(1:end/2)].*repmat(rhs,2,1);

    %amplitude=sqrt(y(1:end/2).^2+y(end/2+1:end).^2);
    %rhs=omega+K/N*(sum(y(end/2+1:end)./amplitude,1)*y(1:end/2)./amplitude-sum(y(1:end/2)./amplitude,1)*y(end/2+1:end)./amplitude);
    %out=[-y(end/2+1:end); y(1:end/2)].*repmat(rhs,2,1);
end


function out = rhs_fun_colloc(t,y,omega,K,N,U,sumU,Uall,V,sumV,Vall,P) 
    
    reconstr1=U*y(1:end/2);
    reconstr2=V*y(end/2+1:end);
    rhs=omega + K/N*( (sumV*y(end/2+1:end))*reconstr1-(sumU*y(1:end/2))*reconstr2 );
    out=[-reconstr2; reconstr1].*repmat(rhs,2,1);
    
    %reconstr1=Uall*y(1:end/2);
    %reconstr2=Vall*y(end/2+1:end);
    %amplitude=sqrt(reconstr1.^2+reconstr2.^2);
    %coupl_cos=sum(reconstr1./amplitude);
    %coupl_sin=sum(reconstr2./amplitude);
    %rhs=omega+K/N*(coupl_sin*(U*y(1:end/2))./amplitude(P)-coupl_cos*(V*y(end/2+1:end))./amplitude(P));
    %out=[-reconstr2(P); reconstr1(P)].*repmat(rhs,2,1);
    
end