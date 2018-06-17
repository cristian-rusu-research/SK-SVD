function [A gamma error] = myksvdSeparate(Y, dictSize, k0, maxIter, A, Aext)
% An implementation of partial (or separate) K-SVD

% Reference: C. Rusu and B. Dumitrescu, Stagewise K-SVD
% to design efficient dictionaries for sparse representations,
% Signal Processing Letters, vol. 19, no. 10, pp. 631-634, 2012.

[~, N] = size(Y);
A = A(:, 1:dictSize);
bigDictionarySize = size(Aext, 2);

gamma = omp([A Aext]'*Y, [A Aext]'*[A Aext], k0);
Js = zeros(1, dictSize);

k = 0;

while(1)    
    k = k + 1;
    
    p = 1:dictSize;
    
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
        supportGamma = setdiff([support dictSize+1:dictSize+bigDictionarySize], j0);
        ERj0 = Y(:, J) - [A(:, support) Aext]*gamma(supportGamma, J);
        
        [U S V] = svds(ERj0, 1);
        A(:, j0) = U;
        gamma(j0, J) = S*V';
    end
    
    gamma = omp([A Aext]'*Y, [A Aext]'*[A Aext], k0);
        
    if (k == maxIter)
        break;
    end
end

error = norm(Y-[A Aext]*gamma, 'fro');
