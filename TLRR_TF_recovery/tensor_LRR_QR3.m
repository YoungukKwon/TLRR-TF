function [Z,E,N] = tensor_LRR_QR3(X,A,init)
[n1,n2,n3] = size(X);
[~,n4,~] = size(A);

%% initialize parameters 
r = init.r; % rank
beta = init.beta; % for L1-norm (E)
gamma = init.gamma; % for L21-norm (N)
lambda = init.lambda; % for nuclear norm
mu = init.mu;  % lagrange multiplier
mu_max = init.mu_max;
rho = init.rho;
tol = init.tol;
max_iter = init.max_iter;

%% initialize tensors
% Z=J=Y1: n1 X n4 X n3;
Z = zeros(n4,n2,n3);
J = Z;
Y1 = Z;

% N=E=Y2: n1 X n2 X n3
E = zeros(n1,n2,n3);
Y2 = E;
N = E;

% L: n4 x r x n3 / D: r x r x n3 / R: r x n2 x n3
L = zeros(n4,r,n3);
D = zeros(r,r,n3);
R = zeros(r,n2,n3);

% pre compute
Ain = t_inverse(A);
AT = tran(A);

iter = 0;
nh3 = ceil((1+n3)/2);
max_inner = 1;

while iter < max_iter
    iter = iter + 1;

    % update J
    J_pre = J;
    Q1 = Z+Y1/mu;
    Q2 = X-E-N+Y2/mu;
    J = tprod(Ain, Q1+tprod(AT,Q2));

    % update Z
    Z_pre = Z;
    LDR = abs(tprod(tprod(L,D),R));
    Z = (1/(1+mu))*(LDR + mu*J - Y1);

    % update L, R, and D
    [L,D,R] = ldr_tnn(Z,r,lambda,max_inner);

    % update E
    E_pre = E;
    tmp = X - tprod(A,J) + Y2/mu;
    R2 = tmp - N;
    E = prox_l1(R2,beta/mu);

    % update N
    N_pre = N;
    R3 = tmp - E;
    Qtran=permute(R3,[1,3,2]);
    Qm=zeros(n1*n3,n2);
    for k=1:n2
        Qk=Qtran(:,:,k);
        Qm(:,k)=Qk(:);
    end        
    QmE=L21_solver(Qm,gamma/mu);
    Et=zeros(n1,n3,n2);
    for k=1:n2
        Et(:,:,k)=reshape(QmE(:,k),n1,n3);
    end
    N=permute(Et,[1,3,2]);

    % check convergence
    leq1 = Z-J;
    leq2 = X-tprod(A,J)-E-N;
    leqm1 = max(abs(leq1(:)));
    leqm2 = max(abs(leq2(:)));
    
    difJ = max(abs(J(:)-J_pre(:)));
    difE = max(abs(E(:)-E_pre(:)));
    difZ = max(abs(Z(:)-Z_pre(:)));
    difN = max(abs(N(:)-N_pre(:)));

    err = max([leqm1,leqm2,difJ,difZ,difE,difN]);

    disp(['iter ', num2str(iter), ' err=', num2str(err)]);

    if err < tol
        break
    end

    % update Y1 and Y2
    Y1 = Y1 + mu*leq1;
    Y2 = Y2 + mu*leq2;

    % update mu
    mu = min(rho*mu, mu_max);   
end

end