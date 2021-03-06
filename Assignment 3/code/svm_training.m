function svmClassifier = svm_training(features_pos, features_neg)
% This function will train your SVM classifier using the positive
% examples (features_pos) and negative examples (features_neg).

% Use label +1 for postive examples and label -1 for negative examples

% INPUT:
% . features_pos: a N1 by D matrix where N1 is the number of faces and D
%   is the hog feature dimensionality
% . features_neg: a N2 by D matrix where N2 is the number of non-faces and D
%   is the hog feature dimensionality

% OUTPUT:
% svmClassifier: A struct with two fields, 'weights' and 'bias' that are
%       the parameters of a linear classifier

% Use vl_svmtrain on your training features to get a linear classifier
% specified by 'w' and 'b'
% [w b] = vl_svmtrain(X, Y, lambda) 
% http://www.vlfeat.org/sandbox/matlab/vl_svmtrain.html
% 'lambda' is an important parameter, try many values. Small values seem to
% work best e.g. 0.00001, but you can try other values

% combine the features in one matrix to give to vl_svmtrain
X = [ features_pos ; features_neg ];
% create the label vector for indicating positive or negative feature 
Y = [ones(1, length(features_pos)) ,  -ones(1, length(features_neg))];
lambda = 0.00001;
% Function wants an D by N matrix (D: feature dimensions, N: feature count)
% so input the transpose of X
[w, b] = vl_svmtrain(X', Y, lambda);

svmClassifier = struct('weights',w,'bias',b);
end