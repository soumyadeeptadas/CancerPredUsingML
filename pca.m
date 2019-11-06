function [U, S] = pca(X)

%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S


[m, n] = size(X);

U = zeros(n);
S = zeros(n);

% We first compute the covariance matrix. Then, we use the "svd" function to
% compute the eigenvectors and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).

covariance = (X' * X) / m;
[U,S,V] = svd(covariance);



end