clear

addpath(genpath('proximal_operator'));
addpath(genpath('tSVD'));
addpath(genpath('utils'));

%% read data and process data 
% YaleB data: 30 images x 28 people = 840 images
load('./data/extendyaleb.mat');
X=X/255.0; 
%X = X(:,1:600,:); % select first 20 people 
X0 = X;
[n1,n2,n3] = size(X);
%label = label(1:600,:);

rng(110);

%% construct the dictionary  
opts.denoising_flag=1; % set the flag whether we use R-TPCA to construct the dictionary 
% (1 denotes we use R-TPCA; 0 deonotes we do not use)

if opts.denoising_flag% if we use R-TPCA to construct the dictionary, we set its parameters
    [n1,n2,n3]=size(X);
    opts.lambda = 1/sqrt(max(n1,n2)*n3);
    opts.mu = 1e-4;
    opts.tol = 1e-5;
    opts.rho = 1.1;
    opts.max_iter = 500;
    opts.DEBUG = 0; %% whether we debug the algorithm
end    

% run the dictionary construction algorithm
tic;
[LL,V] = dictionary_learning(X,opts);
time_dictionary = toc;

%% test R-TLRR
rng(110);
max_iter=500;
tic;
%DEBUG = 0; %% do not output the convergence behaviors at each iteration
[Z,tlrr_E,Z_rank,err_va ] = Tensor_LRR(X,LL,max_iter);
Z=tprod(V,Z); %% recover the real representation

% cluster data
t=0.01;
[ mean_nmi] = ncut_clustering(Z, label',t );
fprintf('\n random NMI: %.4f    %.4f   %.4f\n',mean_nmi(1),mean_nmi(2),mean_nmi(3));
time_TLRR = toc;

%% test ETLRR
rng(110);
alpha = 0.001;
beta = 10;

max_iter_ETL = 500;

tic;
[Z_EL, E, L_rank, err] = ETLRR(X, LL, max_iter_ETL, alpha, beta);
Z_EL=tprod(V,Z_EL); %% recover the real representation

% cluster data
t=0.01;
[ mean_nmi_EL] = ncut_clustering(Z_EL, label',t );
fprintf('\n random NMI: %.4f    %.4f   %.4f\n',mean_nmi_EL(1),mean_nmi_EL(2),mean_nmi_EL(3));
time_ELRR = toc;

%% test TLRR-TF
rng(110);
alpha = 0.001;
beta = 10;
init.r = 3;
init.beta = alpha; % for L1-norm (same with ELRR)
init.gamma = beta; % for L21 norm (same with ELRR)
init.lambda = 10; % for nuclear norm
init.mu = 1e-4; % lagrange multiplier
init.mu_max = 1e+7;
init.rho = 1.1;
init.tol = 1e-6;
init.max_iter = 500;

tic;
[Z_TLRRTF,E] = tensor_LRR_QR3(X,LL,init);
Z_TLRRTF=tprod(V,Z_TLRRTF); %% recover the real representation

% cluster data
t=0.01;
[ mean_nmi_TLRRTF] = ncut_clustering(Z_TLRRTF, label',t );
fprintf('\n random NMI: %.4f    %.4f   %.4f\n',mean_nmi_TLRRTF(1),mean_nmi_TLRRTF(2),mean_nmi_TLRRTF(3));
time_TLRRTF = toc;

%% Computing time (clustering by tensor LRR)
disp(time_TLRR);
disp(time_ELRR);
disp(time_TLRRTF);
