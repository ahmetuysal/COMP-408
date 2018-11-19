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


%delete this place holse
pca_coeff = rand(size(features,2),100);
end