function indices = getWorstReconstructedItems(A, Y, gamma, howmanystart, howmanyend)
% Assuming a dictionary model, this function returns the indices of the
% worst reconstructed data items

thenorms = sqrt(sum(Y.^2));
for i=1:length(thenorms)
    if (thenorms(i) == 0)
        thenorms(i) = 1;
    end
end
error = sum((Y-A*gamma).^2)./thenorms;
[values, indices] = sort(error,2,'descend');
if (howmanystart == 0)
    howmanystart = 1;
end
indices = indices(howmanystart:howmanyend);
