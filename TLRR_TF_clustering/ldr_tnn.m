function [L,D,R] = ldr_tnn(X,rnk,lam,max_inner)
[n1,n2,n3] = size(X);
r = rnk;
lambda = lam;

L1=zeros(n1,r,n3);
R1=zeros(r,n2,n3);

nh3 = ceil((1+n3)/2);
k = 0;

X1 = fft(X,[],3);

% update L and R using CSVD-QR
for k=1:nh3
    R{k} = rand(r,n2);
    i = 0;
    while i<max_inner
        i = i+1;
        [L11,L12] = qr(X1(:,:,k)*(R{k}'));
        L{k}=L11(:,1:r); 
        [R11,R12]=qr(X1(:,:,k)'*L{k});
        R{k}=R11(:,1:r)';
        %D{k}=R12(1:r,1:r)';
    end
end
for k=(nh3+1):n3
    L{k}=conj(L{n3-k+2});
    %D{k}=D{n3-k+2};
    R{k}=conj(R{n3-k+2});
end

for k=1:n3
    L1(:,:,k)=L{k};
    %D1(:,:,k)=D{k};
    R1(:,:,k)=R{k};
end

% update D using nuclear-norm minimization
L1=ifft(L1,[],3);
R1=ifft(R1,[],3);

Y1 = tprod(tprod(tran(L1),X),tran(R1));
Y1 = fft(Y1,[],3);

for i=1:n3
    [Y1(:,:,i),~] = prox_nuclear(Y1(:,:,i), lambda);
end

Y1 = ifft(Y1,[],3);

L = L1;
R = R1;
D = Y1;

end