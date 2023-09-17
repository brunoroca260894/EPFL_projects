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

MC_test=false; %true if I want to use a given number iid samples for testing, independently of the training
how_many=50;

rng('default');

phi0=linspace(0,1*pi,N).';
%phi0=linspace(-pi,pi,N).';
%phi0=2*pi*rand(N,1);
y0=phi0;

%uniform distribution
K=0.15; a=0.97; b=1.03; omega_all=a+(b-a)*rand(N,Ntrials); Kc=2/(pi*1/(b-a)); Rsqrt=@(K) nan(size(K));
%K=10; a=-1; b=1; omega_all=a+(b-a)*rand(N,Ntrials); Kc=2/(pi*1/(b-a)); Rsqrt=@(K) nan(size(K));

%normal distribution
%K=0.3; mu=1; sig=0.02; omega_all=mu+sig*normrnd(0,1,[N,Ntrials]); Kc=2/(pi*1/sqrt(2*pi*sig^2)); Rsqrt=@(K) sqrt(16/(pi*Kc^3))*sqrt(1/(1/sqrt(2*pi*sig^2)*1/sig^2))*sqrt((K-Kc)/Kc);

%cauchy distribution
%K=0.4; omega_0=1; gam=0.015; omega_all = omega_0+gam*tan(pi*(rand(N,Ntrials)-1/2)); Kc=2*gam; Rsqrt=@(K) sqrt(1-Kc./K);

K_all=linspace(0,K,Ncoupl);
omega_all=omega_all-mean(omega_all,1);

%[omega_all,ids]=sort(omega_all,1);
ids=repmat((1:N)',1,max(Ntrials,how_many));

%function
%see end of script


%% construct training set
data=zeros(Ntrials*Ncoupl*(Nsteps+1),N);
data_rhs=zeros(Ntrials*Ncoupl*(Nsteps+1),N);
for k=1:Ncoupl
    disp(k);
    for i=1:Ntrials
        [t_train,y_train]=RK3 (@(t,y) rhs_fun(t,y,omega_all(:,i),K_all(k),N),[0 FinalTime], y0(ids(:,i)), dt);
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
[U,S,~]=svd(data.','econ'); U=U(:,1:n_latent_pod); S=diag(S); 

%check decay of singular values
figure;
semilogy(1:length(S),S/S(1),'color',colors(1,:));
xlabel('$i$','interpreter','latex'); ylabel('$\sigma_i/\sigma_1$','interpreter','latex');

%energy
for k=1:length(S), energy(k)=norm(S(k:end))^2/norm(S)^2; end
figure;
semilogy(1:length(S),energy,'color',colors(1,:));
xlabel('$i$','interpreter','latex'); ylabel('Energy','interpreter','latex');

%show distribution of order parameters
%phi_train=data;
%order_train=abs(mean(exp(1i*phi_train),2));
%figure;
%histogram(order_train,'Normalization','probability');


%% simulate for one instance of the natural frequency
omega=omega_all(:,1);
%rng(0,'philox'); omega=a+(b-a)*rand(N,1); omega=omega-mean(omega); rng('default');
%rng('default'); shift=1; omega_tmp=a+(b-a)*rand(N,Ntrials+10); omega=omega_tmp(:,Ntrials+shift); omega=omega-mean(omega);

K=K_all(end);

%full model
tic;
[time_full,y_full] = RK3 (@(t,y) rhs_fun(t,y,omega,K,N),[0 FinalTime], y0(ids(:,1)), dt);
elapsed_full=toc;
phi_full=y_full;
order_full=abs(mean(exp(1i*phi_full),2));

%optimal projection
time_optproj=time_full;
y_optproj=y_full*(U*U');
phi_optproj=y_optproj;
order_optproj=abs(mean(exp(1i*phi_optproj),2));

%POD
tic;
[time_pod,y_pod_proj] = RK3 (@(t,y) U'*rhs_fun(t,U*y,omega,K,N),[0 FinalTime], U'*y0(ids(:,1)), dt);
y_pod=y_pod_proj*U';
elapsed_pod=toc;
phi_pod=y_pod;
order_pod=abs(mean(exp(1i*phi_pod),2));

%DEIM with snapshots
Pdeim = deim(U); 
PTUinv=U(Pdeim,:)\eye(size(U,2)); 
PTU=U(Pdeim,:);
PTomega=omega(Pdeim);

%DEIM with rhs
%[Urhs,~,~]=svd(data_rhs.','econ'); Urhs=Urhs(:,1:n_samples_deim); 
%Pdeim = deim(Urhs);
%PTUinv=(U'*Urhs)*(Urhs(Pdeim,:)\eye(size(Urhs,2)));
%PTU=U(Pdeim,:);
%PTomega=omega(Pdeim);

tic;
[time_deim,y_deim_proj]=RK3(@(t,y) PTUinv*rhs_fun_colloc(t,y,PTomega,K,N,U,PTU), [0 FinalTime], U'*y0(ids(:,1)), dt);
y_deim=y_deim_proj*U';
elapsed_deim=toc;
phi_deim=y_deim;
order_deim=abs(mean(exp(1i*phi_deim),2));

%projection exploiting averages
%setting n_clus=N and n_per_clus=1 we get the full model
%n_per_clus=2; n_clus=N/2;
%if mod(n_per_clus,1)~=0, error('Asymmetric clusters'); end
%p=cell(n_clus,1); for kk=1:n_clus, p{kk}=1/sqrt(n_per_clus)*ones(n_per_clus,1); end; Pavg=blkdiag(p{:});
%omega_sum=sum(reshape(omega,n_per_clus,n_clus),1)';
%avg_y0=1/sqrt(n_per_clus)*sum(reshape(y0,n_per_clus,n_clus),1)';
%tic;
%[time_avg,y_avg_proj]=RK3 (@(t,y) rhs_fun_groups(t,y,omega_sum,K,N,n_per_clus),[0 FinalTime], avg_y0, dt);
%y_avg=y_avg_proj*Pavg';
%elapsed_avg=toc;
%phi_avg=y_avg;
%order_avg=abs(mean(exp(1i*phi_avg),2));     

%visualize original and reconstructed solution
Noscil_plot=floor(N/5);
figure;
subplot(3,1,1); plot(time_full,y_full(:,1:Noscil_plot:end),'color',[0 0 0]); hold on; xlabel('$t$','interpreter','latex'); ylabel('$\phi_{FOM}$','interpreter','latex');
%plot(time_optproj,y_optproj(:,1:Noscil_plot:end),'--','color',[0 0 0]);
subplot(3,1,2); plot(time_pod,y_pod(:,1:Noscil_plot:end),'color',colors(1,:)); xlabel('$t$','interpreter','latex'); ylabel('$\phi_{POD}$','interpreter','latex');
subplot(3,1,3); plot(time_deim,y_deim(:,1:Noscil_plot:end),'color',colors(2,:)); xlabel('$t$','interpreter','latex'); ylabel('$\phi_{DEIM}$','interpreter','latex');
%subplot(4,1,4); plot(time_avg,y_avg,'color',colors(3,:)); xlabel('$t$','interpreter','latex'); ylabel('$\phi_{AVG}$','interpreter','latex');

%visualize original and reconstructed solution in space-time
figure; 
subplot(3,1,1); surf(time_full,1:N,y_full'); shading interp; view(0,90); colormap jet; xlabel('$t$','interpreter','latex'); ylabel('$i$','interpreter','latex'); title('$\phi_{FOM}$','interpreter','latex'); colorbar;
subplot(3,1,2); surf(time_pod,1:N,y_pod'); shading interp; view(0,90); colormap jet; xlabel('$t$','interpreter','latex'); ylabel('$i$','interpreter','latex'); title('$\phi_{POD}$','interpreter','latex'); colorbar;
subplot(3,1,3); surf(time_deim,1:N,y_deim'); shading interp; view(0,90); colormap jet; xlabel('$t$','interpreter','latex'); ylabel('$i$','interpreter','latex'); title('$\phi_{DEIM}$','interpreter','latex'); colorbar;

%visualize original and reconstructed order parameters
figure;
plot(time_full,order_full,'color',[0 0 0]); hold on;
%plot(time_optproj,order_optproj,'--','color',[0 0 0]); 
plot(time_pod,order_pod,'color',colors(1,:));
plot(time_deim,order_deim,'color',colors(2,:));
%plot(time_avg,order_avg,'color',colors(3,:));
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$t$','interpreter','latex'); ylabel('$R$','interpreter','latex');

%display times
fprintf('Elapsed time for full model: %f\n',elapsed_full);
fprintf('Elapsed time for POD model: %f\n',elapsed_pod);
fprintf('Elapsed time for DEIM model: %f\n',elapsed_deim);
%fprintf('Elapsed time for AVG model: %f\n',elapsed_avg);


%% simulate for the full range of natural frequency, with fixed coupling
K=K_all(end);

Pdeim = deim(U);
PTUinv=U(Pdeim,:)\eye(size(U,2));
PTU=U(Pdeim,:);

%Pdeim = deim(Urhs);
%PTUinv=(U'*Urhs)*(Urhs(Pdeim,:)\eye(size(Urhs,2)));
%PTU=U(Pdeim,:);

%if true, then I test with a given number of iid samples, independent of training
if MC_test
    rng('default'); omega_all=a+(b-a)*rand(N,how_many); omega_all=omega_all-mean(omega_all,1);
end

for i=1:size(omega_all,2)
    disp(i);

    %full model
    [time_full,y_full] = RK3 (@(t,y) rhs_fun(t,y,omega_all(:,i),K,N),[0 FinalTime], y0(ids(:,i)), dt);
    phi_full=y_full;
    order_full=abs(mean(exp(1i*phi_full),2));
    
    %optimal projection
    time_optproj=time_full;
    y_optproj=y_full*(U*U');
    phi_optproj=y_optproj;
    order_optproj=abs(mean(exp(1i*phi_optproj),2));

    %POD
    [time_pod,y_pod_proj] = RK3 (@(t,y) U'*rhs_fun(t,U*y,omega_all(:,i),K,N),[0 FinalTime], U'*y0(ids(:,i)), dt);
    y_pod=y_pod_proj*U';
    phi_pod=y_pod;
    order_pod=abs(mean(exp(1i*phi_pod),2));

    %DEIM
    PTomega=omega_all(Pdeim,i);
    [time_deim,y_deim_proj]=RK3(@(t,y) PTUinv*rhs_fun_colloc(t,y,PTomega,K,N,U,PTU), [0 FinalTime], U'*y0(ids(:,i)), dt);
    y_deim=y_deim_proj*U';
    phi_deim=y_deim;
    order_deim=abs(mean(exp(1i*phi_deim),2));
    
    %save quantities
    order_full_rand(:,i)=order_full;
    order_optproj_rand(:,i)=order_optproj;
    order_pod_rand(:,i)=order_pod;
    order_deim_rand(:,i)=order_deim;
    
end

%compute mean and variance
mean_full=mean(order_full_rand,2); var_full=var(order_full_rand,0,2);
mean_optproj=mean(order_optproj_rand,2); var_optproj=var(order_optproj_rand,0,2);
mean_pod=mean(order_pod_rand,2); var_pod=var(order_pod_rand,0,2);
mean_deim=mean(order_deim_rand,2); var_deim=var(order_deim_rand,0,2);

%visualize confidence interval for order parameter as fcn of time
figure;
plot_confidence_interval(mean_deim-1.96*sqrt(var_deim/size(omega_all,2)),mean_deim+1.96*sqrt(var_deim/size(omega_all,2)),time_deim,colors(2,:));hold on;
plot_confidence_interval(mean_pod-1.96*sqrt(var_pod/size(omega_all,2)),mean_pod+1.96*sqrt(var_pod/size(omega_all,2)),time_pod,colors(1,:));
%plot_confidence_interval(mean_optproj-1.96*sqrt(var_optproj/size(omega_all,2)),mean_optproj+1.96*sqrt(var_optproj/size(omega_all,2)),time_optproj,colors(3,:));
plot_confidence_interval(mean_full-1.96*sqrt(var_full/size(omega_all,2)),mean_full+1.96*sqrt(var_full/size(omega_all,2)),time_full,[0 0 0]);
legend({'DEIM','POD','FOM'},'interpreter','latex');
xlabel('$t$','interpreter','latex'); ylabel('$R$','interpreter','latex');

    
%% simulate for one instance of the random frequency, but varying coupling 
Ncoupl=10;
K_all=linspace(0,K,Ncoupl);
omega=omega_all(:,1);

Pdeim = deim(U); 
PTUinv=U(Pdeim,:)\eye(size(U,2)); 
PTU=U(Pdeim,:); 
PTomega=omega(Pdeim);

%Pdeim = deim(Urhs);
%PTUinv=(U'*Urhs)*(Urhs(Pdeim,:)\eye(size(Urhs,2)));
%PTU=U(Pdeim,:);
%PTomega=omega(Pdeim);

for k=1:Ncoupl
    disp(k);
    
    %full
    tic;
    [time_full,y_full] = RK3 (@(t,y) rhs_fun(t,y,omega,K_all(k),N),[0 FinalTime], y0(ids(:,1)), dt);
    elapsed_time_full(k)=toc;
    phi_full=y_full;
    order_param_full(:,k)=abs(mean(exp(1i*phi_full),2));
   
    %optimal projection
    time_optproj=time_full;
    y_optproj=y_full*(U*U');
    phi_optproj=y_optproj;
    order_param_optproj(:,k)=abs(mean(exp(1i*phi_optproj),2));
    
    %POD
    tic;
    [time_pod,y_pod_proj] = RK3 (@(t,y) U'*rhs_fun(t,U*y,omega,K_all(k),N),[0 FinalTime], U'*y0(ids(:,1)), dt);
    y_pod=y_pod_proj*U';
    elapsed_time_pod(k)=toc;
    phi_pod=y_pod;
    order_param_pod(:,k)=abs(mean(exp(1i*phi_pod),2));
    
    %DEIM
    tic;
    [time_deim,y_deim_proj]=RK3(@(t,y) PTUinv*rhs_fun_colloc(t,y,PTomega,K_all(k),N,U,PTU), [0 FinalTime], U'*y0(ids(:,1)), dt);
    y_deim=y_deim_proj*U';
    elapsed_time_deim(k)=toc;
    phi_deim=y_deim;
    order_param_deim(:,k)=abs(mean(exp(1i*phi_deim),2));
    
    %AVG
    %omega_sum=sum(reshape(omega,n_per_clus,n_clus),1)';
    %avg_y0=1/sqrt(n_per_clus)*sum(reshape(y0,n_per_clus,n_clus),1)';
    %tic;
    %[time_avg,y_avg_proj]=RK3 (@(t,y) rhs_fun_groups(t,y,omega_sum,K_all(k),N,n_per_clus),[0 FinalTime], avg_y0, dt);
    %y_avg=y_avg_proj*Pavg';
    %elapsed_time_avg(k)=toc;
    %phi_avg=y_avg;
    %order_param_avg(:,k)=abs(mean(exp(1i*phi_avg),2));

end
figure;
plot(K_all,elapsed_time_full,'color',[0 0 0]); hold on;
plot(K_all,elapsed_time_pod,'color',colors(1,:)); 
plot(K_all,elapsed_time_deim,'color',colors(2,:)); 
%plot(K_all,elapsed_time_avg,'color',colors(3,:)); 
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$K$','interpreter','latex'); ylabel('$Cost$','interpreter','latex');
figure;
plot(K_all,order_param_full(end,:),'color',[0 0 0]); hold on;
%plot(K_all,order_param_optproj(end,:),'--','color',[0 0 0]);
plot(K_all,order_param_pod(end,:),'color',colors(1,:)); 
plot(K_all,order_param_deim(end,:),'color',colors(2,:)); 
%plot(K_all,order_param_avg(end,:),'color',colors(3,:)); 
%plot(linspace(Kc,max(K_all),100),Rsqrt(linspace(Kc,max(K_all),100)),':','color',[0 0 0]);
ylim([0 1]);
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$K$','interpreter','latex'); ylabel('$R$','interpreter','latex');


%% simulate for all instances of the random frequency AND varying coupling 
%this is similar to generation of the training set and it is a combination
%of the two previous parts that have either K fixed or omega fixed

%if true, then I test with a given number of iid samples, independent of training
if MC_test
    rng('default'); omega_all=a+(b-a)*rand(N,how_many); omega_all=omega_all-mean(omega_all,1);
end

Ncoupl=10;
K_all=linspace(0,K,Ncoupl);

Pdeim = deim(U); 
PTUinv=U(Pdeim,:)\eye(size(U,2)); 
PTU=U(Pdeim,:); 
%PTomega=omega(Pdeim);

%Pdeim = deim(Urhs);
%PTUinv=(U'*Urhs)*(Urhs(Pdeim,:)\eye(size(Urhs,2)));
%PTU=U(Pdeim,:);
%PTomega=omega(Pdeim);

for k=1:Ncoupl
    disp(k);
    
    for i=1:size(omega_all,2)
        omega=omega_all(:,i);
        PTomega=omega(Pdeim);
        
        %full
        tic;
        [time_full,y_full] = RK3 (@(t,y) rhs_fun(t,y,omega,K_all(k),N),[0 FinalTime], y0(ids(:,i)), dt);
        elapsed_time_full(k)=toc;
        phi_full=y_full;
        order_param_full(:,k,i)=abs(mean(exp(1i*phi_full),2));
        
        %POD
        tic;
        [time_pod,y_pod_proj] = RK3 (@(t,y) U'*rhs_fun(t,U*y,omega,K_all(k),N),[0 FinalTime], U'*y0(ids(:,i)), dt);
        y_pod=y_pod_proj*U';
        elapsed_time_pod(k)=toc;
        phi_pod=y_pod;
        order_param_pod(:,k,i)=abs(mean(exp(1i*phi_pod),2));
        
        %DEIM
        tic;
        [time_deim,y_deim_proj]=RK3(@(t,y) PTUinv*rhs_fun_colloc(t,y,PTomega,K_all(k),N,U,PTU), [0 FinalTime], U'*y0(ids(:,i)), dt);
        y_deim=y_deim_proj*U';
        elapsed_time_deim(k)=toc;
        phi_deim=y_deim;
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
    phi_full=y_full;
    order_param_full_k(:,k)=abs(mean(exp(1i*phi_full),2));
    
    %quantities for reduced models
    [U,S,~]=svd(data.','econ'); U=U(:,1:latent_dim(k)); S=diag(S); 
    
    Pdeim = deim(U); 
    PTUinv=U(Pdeim,:)\eye(size(U,2)); 
    PTU=U(Pdeim,:); 
    PTomega=omega(Pdeim);
    
    %[Urhs,~,~]=svd(data_rhs.','econ'); Urhs=Urhs(:,1:latent_dim(k));
    %Pdeim = deim(Urhs);
    %PTUinv=(U'*Urhs)*(Urhs(Pdeim,:)\eye(size(Urhs,2)));
    %PTU=U(Pdeim,:);
    %PTomega=omega(Pdeim);
    
    %optimal projection
    time_optproj=time_full;
    y_optproj=y_full*(U*U');
    phi_optproj=y_optproj;
    order_param_optproj_k(:,k)=abs(mean(exp(1i*phi_optproj),2));

    %POD
    tic;
    [time_pod,y_pod_proj] = RK3 (@(t,y) U'*rhs_fun(t,U*y,omega,K,N),[0 FinalTime], U'*y0, dt);
    y_pod=y_pod_proj*U';
    elapsed_time_pod_k(k)=toc;
    phi_pod=y_pod;
    order_param_pod_k(:,k)=abs(mean(exp(1i*phi_pod),2));
    
    %DEIM
    tic;
    [time_deim,y_deim_proj]=RK3(@(t,y) PTUinv*rhs_fun_colloc(t,y,PTomega,K,N,U,PTU), [0 FinalTime], U'*y0, dt);
    y_deim=y_deim_proj*U';
    elapsed_time_deim_k(k)=toc;
    phi_deim=y_deim;
    order_param_deim_k(:,k)=abs(mean(exp(1i*phi_deim),2));

end

figure;
loglog(latent_dim,elapsed_time_full_k,'color',[0 0 0]); hold on;
loglog(latent_dim,elapsed_time_pod_k,'color',colors(1,:)); 
loglog(latent_dim,elapsed_time_deim_k,'color',colors(2,:)); 
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$k$','interpreter','latex'); ylabel('$Cost$','interpreter','latex');
figure;
subplot(2,1,1);
semilogx(latent_dim,order_param_full_k(end,:),'color',[0 0 0]); hold on;
%semilogx(latent_dim,order_param_optproj_k(end,:),'--','color',[0 0 0]);
semilogx(latent_dim,order_param_pod_k(end,:),'color',colors(1,:)); 
semilogx(latent_dim,order_param_deim_k(end,:),'color',colors(2,:)); 
ylim([0 1]);
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$k$','interpreter','latex'); ylabel('$R$','interpreter','latex');
subplot(2,1,2);
loglog(latent_dim,abs(order_param_pod_k(end,:)-order_param_full_k(end,:))./abs(order_param_full_k(end,:))+1e-16,'color',colors(1,:)); hold on;
loglog(latent_dim,abs(order_param_deim_k(end,:)-order_param_full_k(end,:))./abs(order_param_full_k(end,:))+1e-16,'color',colors(2,:)); 
ylim([0 1]);
legend({'POD','DEIM'},'interpreter','latex');
xlabel('$k$','interpreter','latex'); ylabel('$|R_{FOM}-R_{ROM}|/|R_{FOM}|$','interpreter','latex');
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
    phi0=linspace(0,1*pi,full_dim(nn)).'; y0=phi0;
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
    [U,S,~]=svd(data.','econ'); U=U(:,1:reduced_dim(nn)); S=diag(S);

    %simulate models for one instance of the parameters
    K=K_all(end);
    omega=omega_all(:,1);
    
    %full
    tic;
    [time_full,y_full] = RK3 (@(t,y) rhs_fun(t,y,omega,K,full_dim(nn)),[0 FinalTime], y0, dt);
    elapsed_time_full_n(nn)=toc;
    phi_full=y_full;
    order_param_full_n(:,nn)=abs(mean(exp(1i*phi_full),2));
    
    %quantities for deim model (svd is already done)
    Pdeim = deim(U); 
    PTUinv=U(Pdeim,:)\eye(size(U,2)); 
    PTU=U(Pdeim,:); 
    PTomega=omega(Pdeim);
    
    %[Urhs,~,~]=svd(data_rhs.','econ'); Urhs=Urhs(:,1:reduced_dim(nn));
    %Pdeim = deim(Urhs);
    %PTUinv=(U'*Urhs)*(Urhs(Pdeim,:)\eye(size(Urhs,2)));
    %PTU=U(Pdeim,:);
    %PTomega=omega(Pdeim);
    
    %optimal projection
    time_optproj=time_full;
    y_optproj=y_full*(U*U');
    phi_optproj=y_optproj;
    order_param_optproj_n(:,nn)=abs(mean(exp(1i*phi_optproj),2));

    %POD
    tic;
    [time_pod,y_pod_proj] = RK3 (@(t,y) U'*rhs_fun(t,U*y,omega,K,full_dim(nn)),[0 FinalTime], U'*y0, dt);
    y_pod=y_pod_proj*U';
    elapsed_time_pod_n(nn)=toc;
    phi_pod=y_pod;
    order_param_pod_n(:,nn)=abs(mean(exp(1i*phi_pod),2));
    
    %DEIM
    tic;
    [time_deim,y_deim_proj]=RK3(@(t,y) PTUinv*rhs_fun_colloc(t,y,PTomega,K,full_dim(nn),U,PTU), [0 FinalTime], U'*y0, dt);
    y_deim=y_deim_proj*U';
    elapsed_time_deim_n(nn)=toc;
    phi_deim=y_deim;
    order_param_deim_n(:,nn)=abs(mean(exp(1i*phi_deim),2));


end

figure;
loglog(full_dim,elapsed_time_full_n,'color',[0 0 0]); hold on;
loglog(full_dim,elapsed_time_pod_n,'color',colors(1,:)); 
loglog(full_dim,elapsed_time_deim_n,'color',colors(2,:)); 
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$n$','interpreter','latex'); ylabel('$Cost$','interpreter','latex'); xlim([min(full_dim) max(full_dim)]);
figure;
subplot(2,1,1);
semilogx(full_dim,order_param_full_n(end,:),'color',[0 0 0]); hold on;
%semilogx(full_dim,order_param_optproj_n(end,:),'--','color',[0 0 0]);
semilogx(full_dim,order_param_pod_n(end,:),'color',colors(1,:)); 
semilogx(full_dim,order_param_deim_n(end,:),'color',colors(2,:)); 
ylim([0 1]);
legend({'FOM','POD','DEIM'},'interpreter','latex');
xlabel('$n$','interpreter','latex'); ylabel('$R$','interpreter','latex'); xlim([min(full_dim) max(full_dim)]);
subplot(2,1,2);
loglog(full_dim,abs(order_param_pod_n(end,:)-order_param_full_n(end,:))./abs(order_param_full_n(end,:))+1e-16,'color',colors(1,:)); hold on;
loglog(full_dim,abs(order_param_deim_n(end,:)-order_param_full_n(end,:))./abs(order_param_full_n(end,:))+1e-16,'color',colors(2,:)); 
ylim([0 1]);
legend({'POD','DEIM'},'interpreter','latex');
xlabel('$n$','interpreter','latex'); ylabel('$|R_{FOM}-R_{ROM}|/|R_{FOM}|$','interpreter','latex'); xlim([min(full_dim) max(full_dim)]);
figure;
loglog(elapsed_time_full_n,1e-16*ones(size(elapsed_time_full_n)),'o--','color',[0 0 0]); hold on;
loglog(elapsed_time_pod_n,abs(order_param_pod_n(end,:)-order_param_full_n(end,:))./abs(order_param_full_n(end,:))+1e-16,'o--','color',colors(1,:)); 
loglog(elapsed_time_deim_n,abs(order_param_deim_n(end,:)-order_param_full_n(end,:))./abs(order_param_full_n(end,:))+1e-16,'o--','color',colors(2,:)); 
xlabel('$Cost$','interpreter','latex'); ylabel('$Err$','interpreter','latex');
legend({'FOM','POD','DEIM'},'interpreter','latex');


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