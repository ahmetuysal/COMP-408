function [accuracy,recall,tn_rate,precision] = classifier_performance(svmClassifier,features_pos,features_neg)

% This function will return some metrics generally computed to determine the
% performance of a binary classifier. To clasify a sample first determine 
% the confidence value (also referred to as score): 
% confidence = features*w +b, 
% where features are the hog features, w is the linear classifier 
% weights and b is the classifier bias. 
% Predict a face if confidence>=0 and non-face if confidence<0

% Below you can see the definitions of true-positive, true-negative,
% false-positive and false-negative in a confusion matrix

%                  |    Predicted Non-Face |  Predicted Face
% _________________|_______________________|__________________
%  Actual Non-Face |    TRUE NEGATIVE      |   FALSE POSITIVE
% -----------------|-----------------------|------------------
%  Actual Face     |    FALSE NEGATIVE     |   TRUE POSITIVE
% -----------------|-----------------------|------------------

% You should calculate the following:
%   Accuracy: Overall, how often is the classifier correct?
%       accuracy = (TP+TN)/total = (TP+TN)/(TP+TN+FP+FN)
%   Recall: When it's actually a face, how often does it predict face?
%       recall = TP/actual_faces = TP/(TP+FN)
%   True Negative Rate: When it's actually non-face, how often does it predict non-face?
%       tn_rate = TN/actual_nonfaces = TN/(TN+FP)
%   Precision: When it predicts face, how often is it correct?
%       precision = TP/predicted_yes = TP/(TP+FP)


% remove this placeholder
accuracy=0; recall=0; tn_rate=0; precision=0;

%===================================================================

%        YOUR CODE GOES HERE


%==================================================================

end