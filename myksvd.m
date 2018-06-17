function [A gamma error] = myksvd(Y, dictSize, k0, maxIter, A)
% An implementation of the K-SVD algorithm

% Reference: M. Aharon, M. Elad and A. Bruckstein,
% K-SVD: An algorithm for designing overcomplete dictionaries
% for sparse representation, IEEE Trans. Sig. Proc., 
% vol. 54, no. 11, pp. 4311-4322, 2006.

[~, N] = size(Y);
A = A(:, 1:dictSize);
A = bsxfun(@rdivide, A, sqrt(sum(A.^2)));

gamma = omp(A'*Y, A'*A, k0);
Js = zeros(1, dictSize);

k = 0;

while(1)    
    k = k + 1;
    
    p = randperm(dictSize);
%     p = 1:dictSize;
    
    for j = 1:dictSize
        j0 = p(j);
        
        J = find(gamma(j0,:));
        Js(j0) = length(J);
        
        if (length(J)<1)
            x = Y(:, randsample(N, 1));
            x = x/norm(x);
            A(:, j0) = x;
            continue;
        end

        support = setdiff(1:dictSize, j0);
        ERj0 = Y(:, J) - A(:, support)*gamma(support, J);
        
        [U S V] = svds(ERj0, 1);
        A(:, j0) = U;
        gamma(j0, J) = S*V';
    end
    
    gamma = omp(A'*Y, A'*A, k0);
    
    if (k == maxIter)
        break;
    end
end

error = norm(Y-A*gamma, 'fro');
