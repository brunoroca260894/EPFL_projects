function average_coherence 

m = 1000; n = 50;
coh_A = zeros(m,1);

%for loop to generate 1000 random matrice
for i = 1 : m
   %random matrix of size 1000x50
   A = rand(m,n); 
   %compute the QR decomposition
   [Q,R] = qr(A,0);
   %compute the coherence of the matrix A(i)
   coh_A(i) = max(sum(Q.^2,2));
end

%compute the average coverange of the 1000 random matrices
avg = mean(coh_A)

end