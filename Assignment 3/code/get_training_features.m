function [features_pos, features_neg] = get_training_features...
    (train_path_pos, train_path_neg,hog_template_size,hog_cell_size)

% M = load('features_pos.mat');
% features_pos = M.features_pos;
% N = load('features_neg.mat');
% features_neg = N.features_neg;
% return 
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

first_img = imread(strcat(image_files(1).folder, '\', image_files(1).name));
face_images = zeros([size(first_img), num_images], 'like', first_img); 

% store the images to use in warping
for i = 1:num_images
   face_images(:, :, i) = imread(strcat(image_files(i).folder, '\',...
       image_files(i).name)); 
end


% % Part for warping
% create affine2d objects used for warping
% for reflecting
tform_reflection = affine2d([ -1 0 0; 0 1 0; 0 0 1]);
tform_shear = affine2d([1 .5 0; 0 1 0; 0 0 1]);
tform_shear2 = affine2d([1 .25 0; .25 1 0; 0 0 1]);
tform_shear3 = affine2d([1 -.5 0; 0 1 0; 0 0 1]);
tform_shear4 = affine2d([1 -.25 0; -.25 1 0; 0 0 1]);
tform_shear_reflect = affine2d([-1 .5 0; 0 1 0; 0 0 1]);
tform_shear_reflect2 = affine2d([-1 -.3 0; .25 1 0; 0 0 1]);

tform = affine2d([ 0.5*cos(pi/4) sin(pi/4)     0;
                  -sin(pi/4)     0.5*cos(pi/4) 0;
                   0             0             1]);
               
transforms = [tform_reflection, tform_shear, tform_shear2, tform_shear3,...
    tform_shear4, tform_shear_reflect, tform_shear_reflect2, tform];

% %% Test area for warping
% 
% for i = 1:length(transforms)
%     transform = transforms(i);
%     warper = images.geotrans.Warper(transform, size(first_img));
%     warped_image = warp(warper, face_images(:,:,1));
%     warped_image = imresize(warped_image, [36, 36]);
%     imshow(warped_image)
% end

% initialize features_pos with zeros
features_pos = zeros(num_images * (1 + length(transforms))...
    , (hog_template_size / hog_cell_size)^2 * 31);

% put the HoG for original face images 
for i = 1:num_images
    % calculate the HoG for face image
    hog = vl_hog(im2single(face_images(:, :, i)), hog_cell_size);
    % add the result to features_pos
    features_pos(i, :) = hog(:);
end

% iterate over all transforms and add HoG for warped faces
for i = 1:length(transforms)
    transform = transforms(i);
    warper = images.geotrans.Warper(transform, size(first_img));
    for j = 1:num_images
        % calculate the warped image
        warped_image = warp(warper, face_images(:, :, j));
        % resize the warped image to 36x36
        warped_image = imresize(warped_image, [36, 36]);
        % calculate the HoG for the warped face image
        hog = vl_hog(im2single(warped_image), hog_cell_size);
        % add the result to features_pos
        features_pos(i*num_images + j, :) = hog(:);
    end
end

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
num_samples = 100000;

% counter for stopping at num_samples
sample_count = 0;
% different scales used for extracting, can be changed
scales = [.25, .5, 1, 1.5, 2];
%scales = [.1, .2, .3, .4, .5, .75, 1, 1.25, 1.5, 1.75, 2];
% initialize 
features_neg = zeros(num_samples, (hog_template_size / hog_cell_size)^2 * 31);
    
while sample_count < num_samples
    % randomly select and image to sample windows from given dataset
    rand_img_index = random('unid', num_images);
    rand_img = imread(strcat(image_files(rand_img_index).folder,...
        '\', image_files(rand_img_index).name));
    % convert selected image to grayscale if it's rgb
    if size(rand_img, 3) == 3
        rand_img = rgb2gray(rand_img);
    end
    
    % take windows for different scales of randomly selected image
    for scale = scales
        % scale the image
        rand_scaled_img = imresize(rand_img, scale);
        % take the dimensions of the image
        [w, h] = size(rand_scaled_img);
        
        % if dimensions are smaller than cell size we can't get any sample 
        if w < hog_template_size || h < hog_template_size
            continue
        end

        % determine how many samples will be taken based square root of
        % the rate of area of image to area of sample window 
        num_sample_from_img = floor(sqrt(w * h / hog_template_size^2));

        for i = 1:num_sample_from_img
            % randomly select window location 
            window_x = random('unid', w + 1 - hog_template_size);
            window_y = random('unid', h + 1 - hog_template_size);
            % increase the sample count
            sample_count = sample_count + 1;
            % calculate HoG for selected window
            hog = vl_hog(im2single(rand_scaled_img(...
                window_x:window_x+hog_template_size-1,...
                window_y:window_y+hog_template_size-1)), hog_cell_size);
            % add result to features_neg
            features_neg(i, :) = hog(:);
            % Stop if we react the wanted amount
            if sample_count >= num_samples
                break
            end
        end
        % Stop if we react the wanted amount
        if sample_count >= num_samples
            break
        end
    end
end

 save('features_pos.mat', 'features_pos', '-v7.3')
 save('features_neg.mat', 'features_neg', '-v7.3')

end

