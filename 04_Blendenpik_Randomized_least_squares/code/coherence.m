clear all;
clc;
n=1;

coh=zeros(n,1);

for i=1:n
    A = rand(1000, 50);
    [Q, R] = qr(A, 0);
    coh(i) = max(sum(Q.^2, 2));
end
display('coherence: ')
mean(coh)

%high coherence
m = 10;
n = 5;
B = rand(m,n);
mask_rows = 1:1:m; % for rows
coh_rows = zeros(m, 1);
j = 1;

for i= 1:m
    B = rand(m,n);
    B(mask_rows, j) = 0;
    B(i,j) = rand();
    [Q, R] = qr(B, 0);
    coh(i) = max(sum(Q.^2, 2));
    j = j+1;
end

% mask_columns = 1:1:n; % for rows
% coh_col = zeros(n, 1);
% for i= 1:n
%     B = rand(m,n);
%     B(1, mask_columns) = 0;
%     B(1,i) = rand()
%     [Q, R] = qr(B, 0);
%     coh_col(i) = max(sum(Q.^2, 2));
% end


