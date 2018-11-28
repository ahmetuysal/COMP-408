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

% initialize counters
true_positive = 0;
false_negative = 0;
false_positive = 0;
true_negative = 0;

% extract weights and bias
w = svmClassifier.weights;
b = svmClassifier.bias;

% check for positive images
for i = 1:length(features_pos)
   if features_pos(i, :) * w + b >= 0
       % face is recognized as face: increase true positive
       true_positive = true_positive + 1;
   else
       % face is not recognized: increase false negative
       false_negative = false_negative +1;
   end
end

% check for negative images
for i = 1:length(features_neg)
   if features_neg(i, :) * w + b >= 0
       % non-face recognized as face: increase false positive
       false_positive = false_positive + 1;
   else
       % non-face is not recognized: increase true negative
       true_negative = true_negative +1;
   end
end

% calculate values from counters
accuracy = (true_positive + true_negative) /...
    (true_positive + true_negative + false_positive + false_negative);
recall = true_positive / (true_positive + false_negative);
tn_rate = true_negative / (true_negative + false_positive);
precision = true_positive / (true_positive + false_positive);

end