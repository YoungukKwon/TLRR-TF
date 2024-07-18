clear

addpath(genpath('proximal_operator'));
addpath(genpath('tSVD'));
addpath(genpath('utils'));

% Revised code from https://github.com/shiqiangdu/TLRSR-and-ETLRR
% increase n2, fix n1=n3=60

% Generate Data
n1 = 60;
n3 = 60;
n2 = 300;
k = 5; % number of clusters
m = round(n2/k); % number of samples in each cluster
%q = 0.2;
%r = ceil(min(n1,n3)*q); % tubal rank of each Z_i 
r = 6;

t=0.01; % for ncut clustering
label_m=ones(m,1)*(1:k);
label=label_m(:);

nsim = 1;
res_TPCA = zeros(nsim,1); % Time
res_TLRR = zeros(nsim, 4); % ACC, NMI, PUR, Time
res_ETLRR = zeros(nsim, 4);
res_TLRRTF = zeros(nsim,4);

for s=1:nsim
rng(s+199);

Data=rand(n1,n2,n3);  % rand tenor
U=tsvd(Data); %  U'*U=I;  U is a orthogonal tensor
D = U(:,1:(r*k),:);
Ai=cell(k,1);
Zi=Ai;
Xi=Ai;
Z = zeros(m*k, m*k, n3);
for i=1:k
    Aind = (i-1)*r+1:i*r;
    Ai{i} = D(:,Aind,:);
    Zind = (i-1)*m+1:i*m;
    Zi{i} = randn(r,m,n3);
    Z(Aind,Zind,:)=Zi{i};
    Xi{i}=tprod(Ai{i},Zi{i});
end

X0 = cat(2,Xi{:});
A = cat(2,Ai{:});

% add impulse uniform noise 
S = zeros(n1,n2,n3);
SparseRatio = 0.2;
Somega = randsample(n1*n2*n3, round(n1*n2*n3*SparseRatio));
minI = min(X0(:));
maxI = max(X0(:));
S(Somega) = minI + (maxI-minI).*rand(length(Somega),1);
X = X0 + 1.5*S;

% Gaussian noise
nstd = 0.2;
X = X + nstd*randn(size(X0)); % noise tensor data

opts.denoising_flag=1; % set the flag whether we use R-TPCA to construct the dictionary
% (1 denotes we use R-TPCA; 0 deonotes we do not use)
    
if opts.denoising_flag  % if we use R-TPCA to construct the dictionary, we set its parameters
    [n1,n2,n3]=size(X);
    opts.lambda = 1/sqrt(max(n1,n2)*n3);
    opts.mu = 1e-4;
    opts.tol = 1e-6;
    opts.rho = 1.2; 
    opts.max_iter = 800;
    opts.DEBUG = 0; %% whether we debug the algorithm
end

% run the dictionary construction algorithm
tic;
[LL,V] = dictionary_learning(X,opts);
time_dictionary = toc;
res_TPCA(s,1) = time_dictionary;

%% test TLRR
max_iter=500;
tic;
%DEBUG = 0; %% do not output the convergence behaviors at each iteration
[Z,tlrr_E,Z_rank,err_va ] = Tensor_LRR(X,LL,max_iter);
Z=tprod(V,Z); %% recover the real representation

% cluster data
[ mean_nmi] = ncut_clustering(Z, label',t );
%fprintf('\n random NMI: %.4f    %.4f   %.4f\n',mean_nmi(1),mean_nmi(2),mean_nmi(3));
time_TLRR = toc;
res_TLRR(s,1) = mean_nmi(1);
res_TLRR(s,2) = mean_nmi(2);
res_TLRR(s,3) = mean_nmi(3);
res_TLRR(s,4) = time_TLRR;

%% test ETLRR
alpha = 0.001;
beta = 0.1;
max_iter_ETL = 500;

tic;
[Z_EL, E] = ETLRR(X, LL, max_iter_ETL, alpha, beta);
Z_EL=tprod(V,Z_EL); %% recover the real representation

% cluster data
t=0.01;
[ mean_nmi_EL] = ncut_clustering(Z_EL, label',t );
%fprintf('\n random NMI: %.4f    %.4f   %.4f\n',mean_nmi_EL(1),mean_nmi_EL(2),mean_nmi_EL(3));
time_ELRR = toc;
res_ETLRR(s,1) = mean_nmi_EL(1);
res_ETLRR(s,2) = mean_nmi_EL(2);
res_ETLRR(s,3) = mean_nmi_EL(3);
res_ETLRR(s,4) = time_ELRR;

%% step 3-4 test TLRR-TF
alpha = 0.001;
beta = 0.1;
init.r = round(2*r) ; 
%init.gamma = 30; % for gamma-norm
%init.beta = 1/sqrt(max(n1,n2)*n3); % for L1-norm
init.beta = alpha; % for L1-norm (same with ELRR)
init.gamma = beta; % for L21 norm (same with ELRR)
init.lambda = 10; % for nuclear norm
init.mu = 1e-4; % lagrange multiplier
init.mu_max = 1e+7;
init.rho = 1.1;
init.tol = 1e-6;
init.max_iter = 500;

tic;
[Z_EQR,E,N] = tensor_LRR_QR3(X,LL,init);
Z_EQR=tprod(V,Z_EQR); %% recover the real representation

% cluster data
t=0.01;
[ mean_nmi_EQR] = ncut_clustering(Z_EQR, label',t );
%fprintf('\n random NMI: %.4f    %.4f   %.4f\n',mean_nmi_EQR(1),mean_nmi_EQR(2),mean_nmi_EQR(3));
time_TLRRTF = toc;
res_TLRRTF(s,1) = mean_nmi_EQR(1);
res_TLRRTF(s,2) = mean_nmi_EQR(2);
res_TLRRTF(s,3) = mean_nmi_EQR(3);
res_TLRRTF(s,4) = time_TLRRTF;

disp(['rep ', num2str(s), ' fin']);

end

%% Results (ACC, NMI, PUR, Time)
% clustering performance
disp(mean(res_TLRR(:,1:3), 1));
disp(mean(res_ETLRR(:,1:3), 1));
disp(mean(res_TLRRTF(:,1:3), 1));

% computing time (dictionary learning + clustering by LRR)
disp(mean(res_TPCA(:,1)));
disp(mean(res_TLRR(:,4)+res_TPCA(:,1)));
disp(mean(res_ETLRR(:,4)+res_TPCA(:,1)));
disp(mean(res_TLRRTF(:,4)+res_TPCA(:,1)));
