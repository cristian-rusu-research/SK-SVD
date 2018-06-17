function [A gamma time errors] = SKSVD(Y, k0, m, H, R, maxError)
%SKSVD Stagewise K-SVD algorithm - A proof of concept implementation of the
%Stagewise K-SVD algorithm.
%
% [A gamma time errors] = SKSVD(Y, k0, m, H, R, maxError)
%
% Input:
%   Y - dataset (n x m)
%   k0 - target sparsity: 1 <= k0 <= m
%   m - target dimension of the dictionary m <= N
%   H, R - parameters of SK-SVD (H >= 2, R >= 1)
%   maxError - target maximum error ( norm(Y-A*gamma, 'fro') )
%
% Output:
%   A - dictionary of dimension m
%   gamma - representation matrix
%   time - running time
%   errors - Frobenius norm at each iteration
%
% Reference: C. Rusu and B. Dumitrescu, Stagewise K-SVD to design
%    efficient dictionaries for sparse representations, 
%    IEEE Signal Processing Letters, 19(10):631-634, 2012.

% start timer
tic;

%% check the input
k0 = max(1, round(k0)); k0 = min(m, k0);
m = max(k0, round(m)); m = min(size(Y, 2), m);
H = max(2, round(H)); H = min(m, H);
R = max(1, round(R)); R = min(m, R);

if (isempty(Y) || sum(isnan(Y(:))))
    error('Problems with the input dataset!');
end

% check if OMP exists
if (exist('omp', 'file')~=2)
    error('No OMP!');
end

%% internal parameter of SK-SVD

% percetanges for worst constructed data items
PERCENT_BAD_ITEMS_START = 0.00;
PERCENT_BAD_ITEMS_END = 0.05;

% number of K-SVD iterations for regular and reduced runs
K1 = 15;
K2 = 15;

% tolerance of SVD computation
OPTIONS.tol = 1e-5;

% get dimensions
[~, N] = size(Y);
k = 0;

%% if dimension of the dictionary is target sparsity train using only K-SVD
if (k0 == m)
    [A, ~, ~] = svds(Y, k0);
    [A gamma errors] = myksvd(Y, k0, k0, 60, A);
    time = toc;
    return;
end

%% initialization of dictionary
[A, ~, ~] = svds(Y, k0);
[A gamma err] = myksvd(Y, k0, k0, K1, A);

%% main loop of SK-SVD
errors = err;
while(1)    
    k = k + 1;
    
    % order the atoms according to energy
    sumonrows = mean(gamma.^2, 2);
    [~, indices] = sort(sumonrows,1,'descend');
    A = A(:, indices);
    gamma = gamma(indices, :);
    
    % remove less used atom
    A = A(:, 1:end-1);
    
    % add atoms procedure
    % indices of worst reconstructed items
    indices = getWorstReconstructedItems(A,Y,gamma(1:end-1, :),round(N*PERCENT_BAD_ITEMS_START), round(N*PERCENT_BAD_ITEMS_END));
    
    % construct matrix for new atoms
    matrix = Y(:, indices) - A*gamma(1:end-1, indices);
    
    % alternative, add new H atom 
    % [U S] = svds(matrix, howMuchToAddPerIteration, 'L', OPTIONS);
    % [U S] = svds(Y(:, indices) - A(:, 1:round(end*0.99))*gamma(1:round(end*0.99), indices), howMuchToAddPerIteration);
    U = [];
    % add iteratively H atoms
    untilnow = [];
    for i=1:H
        if (~isempty(untilnow))
            [Ue S V] = svds(matrix - untilnow, 1, 'L', OPTIONS);
            untilnow = untilnow + Ue*S*V';
        else
            [Ue S V] = svds(matrix, 1, 'L', OPTIONS);
            untilnow = Ue*S*V';
        end
        
        U = [U Ue];
    end
    
    % check if H atoms were added
% % %     [~, mu] = size(U);
% % %     if (mu ~= H)
% % % %         U = [U rand(nu, howMuchToAddPerIteration-mu)];
% % %         U = [U Y(:, indices(1:H-mu))];
% % %         U = bsxfun(@rdivide, U, sqrt(sum(U.^2)));
% % %     end

    % regular of reduced K-SVD iterations
    if (mod(k, R) == 0)
       A = [A U]; dictSize = size(A, 2);
       [A gamma err] = myksvd(Y, dictSize, k0, K1, A);
    else
        U = myksvdSeparate(Y(:, indices), size(U,2), k0, K2, U, A);
        A = [A U]; dictSize = size(A, 2);
        gamma = omp(A'*Y, A'*A, k0);
        err = norm(Y-A*gamma, 'fro');
    end
    
    % update error vector
    errors = [errors err];
    
    % check stopping conditions
    if (dictSize >= m)
        break;
    end
    if (err < maxError)
        break;
    end
end

% alternative, extra final step
% [A gamma error] = myksvdSimpleStopWhenNoProgress(Y, dictSize, k0, 50, A);

% final outputs
sumonrows = mean(gamma.^2, 2);
[~, indices] = sort(sumonrows,1,'descend');
A = A(:, indices);
A = A(:, 1:m);
gamma = omp(A'*Y, A'*A, k0);
err = norm(Y-A*gamma, 'fro');

errors = [errors err];

% return time
time = toc;
