% Sliding window face detection with linear SVM. 

% set up paths to VLFeat functions. 
% See http://www.vlfeat.org/matlab/matlab.html for VLFeat Matlab documentation
% This should work on 32 and 64 bit versions of Windows, MacOS, and Linux

close all
clear
% I have imported vlfeat using startup.m file
% run('vlfeat/toolbox/vl_setup')

[~,~,~] = mkdir('visualizations');

data_path = '../data/'; 
train_path_pos = fullfile(data_path, 'caltech_faces/Caltech_CropFaces'); %Positive training examples. 36x36 head crops
train_path_neg = fullfile(data_path, 'train_non_face_scenes'); %We can mine random or hard negatives from here
test_data_path = fullfile(data_path,'test_scenes/test_jpg'); %CMU+MIT test scenes
test_label_path = fullfile(data_path,'test_scenes/ground_truth_bboxes.txt'); %the ground truth face locations in the test set

% The faces are 36x36 pixels. So we will take the hog template size as
% 36 and hog cell size as 6. You can test for different cell sizes as bonus.
% Just make sure hog_template_size is divisible by hog_cell_size

hog_template_size = 36;
hog_cell_size = 6;
% TODO try with 3, 4, 9, 12, 18

%% Step 1. Load positive training crops and random negative examples
% You need to code the function get_training_features(). It will take
% the folder paths of face images (positive examples) and non-face images
% (negative examples), and the parameters for HoG.

% get_training_features() returns two matrices: 
% 'features_pos' is a N1xD matrix where N1 is the number of positive samples and D
%  is the HoG feature dimensionality. 
% 'features_neg' is a N2 by D matrix where N2 is the number of non-faces 
% and D is the hog feature dimensionality. N2 will depend on number of
% non-face images that you mine. You can start with 10,000 but can increase
% it for improved results. This however will slow down your training.

[features_pos, features_neg] = get_training_features(train_path_pos, ...
    train_path_neg,hog_template_size,hog_cell_size);

    
%% Step 2. Train Classifier

% You need to code the svm_training function. This will train an SVM 
% classifier from the negative and positive examples and return a 
% struct with two fields: weights and bias. Weight will be a column vector
% of same size as hog feature and bias will be a scalar value.

svmClassifier = svm_training(features_pos, features_neg);
% Visualize the learned detector using weights of the trained classifier
% This would be a good thing to include in
% your writeup!
n_hog_cells = hog_template_size/hog_cell_size;
imhog = vl_hog('render', single(reshape(svmClassifier.weights, [n_hog_cells n_hog_cells 31])), 'verbose') ;
figure(3); imagesc(imhog) ; colormap gray; set(3, 'Color', [.988, .988, .988])




%% step 3. Check the performance of learned classifier
% You need to complete the function classifier_performance().
% This will return some metrics generally computed to determine the
% performance of a binary classifier. In your report mention the values 
% you get and briefly explain what they represent

[accuracy,recall,tn_rate,precision] = ...
    classifier_performance(svmClassifier,features_pos,features_neg);

disp(['accuracy = ',num2str(accuracy),', recall = ',num2str(recall),...
     ', tn_rate = ',num2str(tn_rate),', precision = ',num2str(precision)]);



%% Step 4. Run detector on test set.
% YOU CODE 'run_detector'. Make sure the outputs are properly structured!
% They will be interpreted in Step 5 to evaluate and visualize your
% results. See run_detector.m for more details.


% run_detector will have (at least) two parameters which can heavily
% influence performance -- how much to rescale each step of your multiscale
% detector, and the threshold for a detection. If your recall rate is low
% and your detector still has high precision at its highest recall point,
% you can improve your average precision by reducing the threshold for a
% positive detection.

[bboxes, confidences, image_ids] = ...
    run_detector(test_data_path, svmClassifier, hog_template_size,hog_cell_size);

disp(size(confidences))

%% Step 5. Evaluate and Visualize detections
% These functions require ground truth annotations, and thus can only be
% run on the CMU+MIT face test set. Use visualize_detections_by_image_no_gt
% for testing on extra images (it is commented out below).

% Don't modify anything in 'evaluate_detections'!
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections(bboxes, confidences, image_ids, test_label_path);

visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_data_path, test_label_path)
% visualize_detections_by_image_no_gt(bboxes, confidences, image_ids, test_data_path)

% visualize_detections_by_confidence(bboxes, confidences, image_ids, test_data_path, test_label_path);

% performance to aim for
% random (stater code) 0.001 AP
% single scale ~ 0.2 to 0.4 AP
% multiscale, 6 pixel step ~ 0.83 AP
% multiscale, 4 pixel step ~ 0.89 AP
% multiscale, 3 pixel step ~ 0.92 AP



% %% step 6. (optional) Implement PCA to reduce feature dimensionality and then train SVM.  
% % The feature size of hog can be computed as (hog_template_size/hog_cell_size)^2*31
% % This may be a long vector and may take a lot of time to train the classifer.
% % PCA allows you to reduce the dimensionality of feature by computing
% % principal axes that represent maximum variance. It may also improve the 
% % classifier performance.
% 
% % You will code the function pca_components from scratch in this part. Follow the 
% % lecture slides discussed in class. pca_components will
% % take hog features of positive samples (feature dimension = N1) and
% % return a 'coeff' matrix of size N1xN2 (where N2 < N1).
% % Each column of coeff contains coefficients for one principal component.
% 
% % In your code you should choose N2 optimally. Refer to your lectures which
% % explains N2 should be chosen such that the cumulative variance ratio > 0.9
% 
% % YOU SHOULD CODE THE FUNCTION pca_components()
% % which computes coefficients for top N2 principal dimensions
% pca_coeff = pca_components(features_pos);
% 
% % This step is done for you. New features are computed from the
% % from the principal coeffients
% features_pos = features_pos*pca_coeff;
% features_neg = features_neg*pca_coeff;
% 
% % You have already done this part. SVM is retrained with the new features
% svmClassifier = svm_training(features_pos, features_neg);
% 
% 
% %You have already done this part. Performance measures.
% [accuracy,recall,tn_rate,precision] = ...
%     classifier_performance(svmClassifier,features_pos,features_neg);
% 
% disp(['accuracy = ',num2str(accuracy),', recall = ',num2str(recall),...
%      ', tn_rate = ',num2str(tn_rate),', precision = ',num2str(precision)]);
% 
%  % YOU SHOULD CODE THE FUNCTION run_detector_pca(). This is exactly like 
%  % the run_detector you coded above except for two differences.
%  %  . it has an extra input argument pca_coeff which is the N1xN2 matrix
%  %  computed above. Each column represents the coefficients of a principal
%  %  axis
%  % . the features determined for the test images should be of length N2
%  % i.e. each hog feature of length N1 will be projected onto N2 dimensions.
%  [bboxes, confidences, image_ids] = ...
%     run_detector_pca(test_data_path, svmClassifier, hog_template_size,hog_cell_size, pca_coeff);
% 
% % Don't modify anything in 'evaluate_detections'!
% [gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
%     evaluate_detections(bboxes, confidences, image_ids, test_label_path);
% 
% % visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_data_path, test_label_path)
% % visualize_detections_by_image_no_gt(bboxes, confidences, image_ids, test_data_path)
% 
% % visualize_detections_by_confidence(bboxes, confidences, image_ids, test_data_path, test_label_path);
% 
