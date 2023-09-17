% function [time,out] = RK3 (fun, tspan, y0, dt, params)
% t=tspan(1);
% Niter=ceil(abs(tspan(2)-tspan(1))/dt)*sign(tspan(2)-tspan(1));
% time=zeros(Niter+1,1); out=zeros(Niter+1,length(y0)); 
% time(1)=tspan(1); out(1,:)=y0;
% 
% for i=1:Niter
%     
%     if t+dt>=tspan(2), dt=tspan(2)-t; end
%     sol=out(i,:)';
%     
%     k1=fun(t,sol,params);
%     k2=fun(t+1*dt,sol+1*dt*k1,params);
%     k3=fun(t+0.5*dt,sol+0.25*dt*k1+0.25*dt*k2,params);
%     next=(sol+dt/6*(k1+k2+4*k3))'; 
%         
%     out(i+1,:)=next;
%     time(i+1,:)=t+dt;
%     t=t+dt;
% end
%     
% end

function [time,out] = RK3 (fun, tspan, y0, dt)
t=tspan(1);
Niter=ceil(abs(tspan(2)-tspan(1))/dt)*sign(tspan(2)-tspan(1));
time=zeros(Niter+1,1); out=zeros(Niter+1,length(y0)); 
time(1)=tspan(1); out(1,:)=y0;

for i=1:Niter
    
    if t+dt>=tspan(2), dt=tspan(2)-t; end
    sol=out(i,:).';
    
    k1=fun(t,sol);
    k2=fun(t+1*dt,sol+1*dt*k1);
    k3=fun(t+0.5*dt,sol+0.25*dt*k1+0.25*dt*k2);
    next=(sol+dt/6*(k1+k2+4*k3)).'; 
        
    out(i+1,:)=next;
    time(i+1,:)=t+dt;
    t=t+dt;
end
    
end