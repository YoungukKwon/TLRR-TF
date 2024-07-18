clear

addpath proxi_operator
addpath utils


% Use tri-factorization Z = L*D*R
% Use tensor nuclear norm (D) + L1-norm (E)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load image 
tmp = double(imread("./image_kodak/kodim01.png"));
X0 = tmp./max(tmp(:));
maxI = max(abs(X0(:)));
[n1,n2,n3] = size(X0);

%nsim = 1;

%res_TRPCA = zeros(nsim, 3);  % RSE, PSNR, Time
%res_TLRR = zeros(nsim, 3);
%res_ETLRR = zeros(nsim, 3);
%res_ETLRRQR = zeros(nsim, 3);

% add impulse noise
SparseRatio = 0.2;
X = X0;
Omega = find(rand(n1*n2*n3,1)<SparseRatio);
X(Omega) = randi([0,255],length(Omega),1)./255.0;

% dictionary learning using TRPCA
tic
[n1,n2,n3]=size(X);
opts.lambda = 1/sqrt(max(n1,n2)*n3);
opts.mu = 1e-4;
opts.tol = 1e-6;
opts.rho = 1.1;
opts.max_iter = 500;
opts.DEBUG = 0;
[L, E, rank_rtpca, ~, ~, ~] = trpca_tnn(X, opts.lambda, opts);
time_trpca = toc;
L_TPCA = max(L, 0);
L_TPCA = min(L_TPCA, maxI);
[psnr_TPCA, rse_TPCA] = PSNR2(X0, L_TPCA, maxI);
%res_TRPCA(s,1) = rse_TPCA;
%res_TRPCA(s,2) = psnr_TPCA;
%res_TRPCA(s,3) = time_trpca;

% Approximate dictionary
% Use same dictionary for TLRR, TLRR-QR, ELRR, and ELRR-QR
tho=50;
[L_hat,trank,U,V,S ] = prox_low_rank(L,tho);
LL = tprod(U,S);
size_LL = size(LL);
n4 = size_LL(2);

%% TLRR
max_iter = 500;
tic;
[Z, tlrr_E, Z_rank, err_va] = Tensor_LRR(X, LL, max_iter);
time_TLRR = toc;
L_TLRR = tprod(LL, Z);
L_TLRR = max(L_TLRR, 0);
L_TLRR = min(L_TLRR, maxI);
[psnr_TLRR, rse_TLRR] = PSNR2(X0, L_TLRR, maxI);
%res_TLRR(s,1) = rse_TLRR;
%res_TLRR(s,2) = psnr_TLRR;
%res_TLRR(s,3) = time_TLRR;

%% ETLRR
alpha = 0.1;
beta = 1000;
max_iter_ETL = 500;
tic;
[Z_EL, E] = ETLRR(X, LL, max_iter_ETL, alpha, beta);
time_ETLRR = toc;
L_ELRR = tprod(LL,Z_EL);
L_ELRR = max(L_ELRR, 0);
L_ELRR = min(L_ELRR, maxI);
[psnr_ETLRR, rse_ETLRR] = PSNR2(X0, L_ELRR, maxI);
%res_ETLRR(s,1) = rse_ELRR;
%res_ETLRR(s,2) = psnr_ELRR;
%res_ETLRR(s,3) = time_ELRR;

%% TLRR-TF
alpha = 0.1;
beta = 1000;
init2.r = 45; % ~ 4*ceil((n4+n2)/100)
init2.beta = alpha; % for L1-norm (same with ELRR)
init2.gamma = beta; % for L21 norm (same with ELRR)
init2.lambda = 0.01; % for nuclear norm
init2.mu = 1e-4; % lagrange multiplier
init2.mu_max = 1e+7;
init2.rho = 1.2;
init2.tol = 1e-6;
init2.max_iter = 500;

tic;
[Z_ELQR, E, N] = tensor_LRR_QR3(X,LL,init2);
time_TLRRTF = toc;
L_ELRRQR = tprod(LL,Z_ELQR);
L_ELRRQR = max(L_ELRRQR, 0);
L_ELRRQR = min(L_ELRRQR, maxI);
[psnr_TLRRTF, rse_TLRRTF] = PSNR2(X0, L_ELRRQR, maxI);
%res_ETLRRQR(s,1) = rse_ELRRQR;
%res_ETLRRQR(s,2) = psnr_ELRRQR;
%res_ETLRRQR(s,3) = time_ELRRQR;
disp("ETLRRQR fin");
disp(['rep ', num2str(s), ' fin']);

%% Result (PSNR, RSE, Time)

% TRPCA
disp([psnr_TPCA, rse_TPCA, time_trpca])

% TLRR
disp([psnr_TLRR, rse_TLRR, time_TLRR])

% ETLRR
disp([psnr_ETLRR, rse_ETLRR, time_ETLRR])

% TLRR-TF
disp([psnr_TLRRTF, rse_TLRRTF, time_TLRRTF])

% plot comparison

%hs = self_subplot(2,3, [0.05, 0.05], [0.05, 0.05], [0.01, 0.01]);
%axes(hs(1));
%imshow(uint8(X0.*255));
%title('Original','fontname','Times New Roman');
%axes(hs(2));
%imshow(uint8(X.*255));
%title('Noisy','fontname','Times New Roman');
%axes(hs(3));
%imshow(uint8(L_TPCA.*255));
%title('TRPCA','fontname','Times New Roman');
%axes(hs(4));
%imshow(uint8(L_TLRR.*255));
%title('TLRR','fontname','Times New Roman');
%axes(hs(5));
%imshow(uint8(L_ELRR.*255));
%title('ELRR','fontname','Times New Roman');
%axes(hs(6));
%imshow(uint8(L_ELRRQR.*255));
%title('ELRR-QR','fontname','Times New Roman');

%kodak1_tile = {X0, X, L_TPCA, L_TLRR, L_ELRR, L_ELRRQR};
%save("kodak1_tile.mat", "kodak1_tile");