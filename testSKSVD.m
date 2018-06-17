%% TEST SK-SVD

%% clean-up
close all;
clear;
clc;

% the code used omptoolbox, if you have it already comment this line!!!
addpath([pwd '/omptoolbox']);

%% read input data
images = {'peppers.bmp'};
Y = readImages(images);

% normalize
Y = Y./255;
% Y = bsxfun(@rdivide, Y, sqrt(sum(Y.^2)));
[n N] = size(Y);
disp('Done reading!');


%% SK-SVD algorithm
% paramerters of SK-SVD
H = 3;
R = 3;
% target sparsity
k0 = 4;
% target dimension of the dictionary
m = 128;
% target representation error
maxError = 0;

% call of SK-SVD
[A gamma time errors] = SKSVD(Y, k0, m, H, R, maxError);

% reconstruction
yhat = A*gamma;

RMSE = sqrt( mean( (Y(:) - yhat(:)).^2 ) );
