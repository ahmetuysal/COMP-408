function [features_pos, features_neg] = get_training_features...
    (train_path_pos, train_path_neg,hog_template_size,hog_cell_size)

%This function returns the hog features of all positive examples and for
%negative examples

% INPUT:
%   . train_path_pos: a string which is path to the directory containing
%        36x36 images of faces (Positive Examples)
%   . train_path_neg: a string which is path to directory containing images
%       which have no faces in them. (Negative Examples)
%   . hog_template_size: size of hog template. In this function it will be 
%       equal to the image size (i.e. 36) 
%   . hog_cell_size: the number of pixels in each HoG cell. Image size 
%       should be evenly divisible by hog_cell_size.

%     Smaller HoG cell sizes tend to work better, but they make things
%     slower because the feature dimensionality increases and more
%     importantly the step size of the classifier decreases at test time.

% OUTPUT
% . features_pos: a N1 by D matrix where N1 is the number of faces and D
%       is the hog feature dimensionality, which would be
%       (hog_template_size / hog_cell_size)^2 * 31
%       if you're using the default vl_hog parameters
% . features_neg: a N2 by D matrix where N2 is the number of non-faces and D
%       is the hog feature dimensionality

% Useful functions
% vl_hog()
% rgb2gray()

%% Step 1: Determine features for positive images (face images)
% This part should create hog features for all positive training examples 
% (faces) from 36x36 images in 'train_path_pos'. 

% Each face should be converted into a hog grid according to 
% 'hog_cell_size'. For example a hog_cell_size of 6 means there are 6x6 
% pixels in one cell. The hog grid will be of size 6x6 for images of size
% 36x36. A hog vector of length 31 will be computed for each cell.

% For improved performance, try mirroring or warping the positive 
% training examples.
image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files);


% placeholder to be deleted
features_pos = rand(num_images, (hog_template_size / hog_cell_size)^2 * 31);

%===================================================================

%        YOUR CODE GOES HERE


%==================================================================
    



%% Step 2: Mine Negative Samples (non-face) and determine hog features

% This part should return hog features from negative training examples 
% (non-faces). Higher number of negative samples will improve results
% however start with 10000 negative samples for debugging

% Images should be converted to grayscale, because the positive training 
% data is only available in grayscale. 

% The set of non-face images available in the dataset vary in size.
% (unlike face images which were all 36x36). You need to mine negative samples
% by randomly selecting patches of size hog_template_size. This ensures the feature
% length of negative samples matches with positive samples. you might try 
% to sample some number from each image, but some images might be too small  
% to find enough. For best performance, you should sample random negative
% examples at multiple scales.

image_files = dir( fullfile( train_path_neg, '*.jpg' ));
num_images = length(image_files);
num_samples = 10000;

% placeholder to be deleted
features_neg = rand(num_samples, (hog_template_size / hog_cell_size)^2 * 31);

%===================================================================

%        YOUR CODE GOES HERE


%==================================================================



end

