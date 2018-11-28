function pca_coeff = pca_components(features)

% You will code pca_components() from scratch in this part. Follow the 
% lecture slides discussed in class. pca_components() will
% take hog features from all positive samples (feature dimension = N1) and
% return a 'pca_coeff' matrix of size N1xN2 (N2 < N1) where N2 is the new 
% dimensionality of features. Each column of pca_coeff contains coefficients 
% for one principal component.

% The choice of N2 can be arbitrary. However a better approach for the
% choice of N2 is to choose it such that the cumualative variance ratio 
% is greater than a threshold (e.g. 0.9). Mathematically it can be written as
% sum(eigen_vales(1:N2))/sum(eigen_values(:)) > 0.9 where eigen_values are
% ordered in descending manner.


%INPUT:
% . features: a MxN1 matrix where M is the number of samples and N1 is the
%       size of feature
%
%OUTPUT:
% . pca_coeff: a N1xN2 matrix, where N1 is the size of original features
%       and N2 is the size of new features. Each i-th column of pca_coeff has
%       coefficiens for i-thprincipal axis 

% threshold value to used in PCA, biggest M eigenvectors will be selected
% corresponding the biggest M eigenvector with sum bigger than threshold *
% sum of all eigenvalues
threshold = 0.9;

% substract mean from features 
features = features - mean(features);

% matlab's cov normalize with N-1, but I normalized with N according to
% slides, to use covariance matrix normalized with N-1 uncomment this line
% covariance_mat = features'*features./(length(features)-1);

% for comparison
% matlab_cov = cov(features);

% calculate covariance matrix
covariance_mat = features'*features./length(features);
% find eigen values and corresponding eigen vectors
[eig_vectors, eig_values_matrix] = eig(covariance_mat);
% sort eigen values in descending order, save original index of the value 
% for future referencing corresponding eigen vector
eig_values = diag(eig_values_matrix);
[eig_values,sorted_indices] = sort(eig_values, 'descend');
% calculate cumulative sum to check for threshold condition
cumsums = cumsum(eig_values);
% calculate how many eigen vectors to take
index = 1;
while cumsums(index) < threshold * cumsums(end)
    index = index + 1;
end
% initialize pca_coefficient matrix with zeros
pca_coeff = zeros(length(eig_values), index);
% copy eigenvectors to pca_coefficients
for i = 1:index
    pca_coeff(: ,i) = eig_vectors(:, sorted_indices(i));
end

end