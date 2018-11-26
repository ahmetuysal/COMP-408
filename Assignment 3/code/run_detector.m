function [bboxes, confidences, image_ids] = ....
    run_detector(test_data_path, svmClassifier, hog_template_size,hog_cell_size)

% This function returns detections on all of the images in 'test_data_path'.

% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.

% INPUT:
% . test_data_path: a string which is path to the directory that contains
%       images which may or may not have faces in them. This function
%       should work for the MIT+CMU test set but also for any other images
%       (e.g. class photos)
% . svmClassifier: A struct with two fields, 'Weights' and 'Bias' that are
%       the parameters of a linear classifier
% . hog_template_size: size of hog template.(default 36)
% . hog_cell_size: the number of pixels in each HoG cell (default 6). Image size
%       should be evenly divisible by hog_cell_size.

% OUTPUT:
% . bboxes: a Nx4 matrix. N is the number of detections. bboxes(i,:) is
%       [x_min, y_min, x_max, y_max] for detection i.
%       Remember 'y' is dimension 1 in Matlab!
% . confidences: a Nx1 matrix. confidences(i) is the real valued confidence of
%       detection i.
% . image_ids: a Nx1 cell array. image_ids{i} is the image file name
%       for detection i. (not the full path, just 'albert.jpg')


test_scenes = dir( fullfile( test_data_path, '*.jpg' ));
num_images = length(test_scenes);

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

for i = 1:num_images
    
    image_name = test_scenes(i).name;
    fprintf('Detecting faces in %s\n', image_name);
    img = imread( fullfile( test_data_path, test_scenes(i).name ));
    
    % The PlaceHolder() function is here to help you understand how the
    % outputs (bboxes,confidences,image_ids) should be structured. This
    % function generates random bounding boxes for a test image. It will
    % even do non-maximum suppression on the random bounding boxes to give
    % you an example of how to call the function. This part of the code needs
    % to be commented out when you write your own function.
    
%     [cur_confidences,cur_bboxes] =...
%         PlaceHolder(img, svmClassifier, hog_template_size,hog_cell_size);
    
    
    % You will be coding the Detector() function. It has a similar
    % function definition as that of PlaceHolder().
    
    %  Complete the function Detector() and uncomment the line below
    [cur_confidences,cur_bboxes] =...
       Detector(img, svmClassifier, hog_template_size,hog_cell_size);
    
    cur_image_ids = cell(0,1);
    cur_image_ids(1:size(cur_bboxes,1)) = {test_scenes(i).name}; 
    
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
end
end



function [cur_confidences,cur_bboxes] = ...
    PlaceHolder( img,svmClassifier, hog_template_size,hog_cell_size)


% Creating 15 random detections per image
cur_x_min = rand(15,1) * size(img,2);
cur_y_min = rand(15,1) * size(img,1);
cur_bboxes = [cur_x_min, cur_y_min, cur_x_min + rand(15,1) * 50, cur_y_min + rand(15,1) * 50];
cur_confidences = rand(15,1) * 4 - 2; %confidences in the range [-2 2]


%non_max_supr_bbox can actually get somewhat slow with thousands of
%initial detections. You could pre-filter the detections by confidence,
%e.g. a detection with confidence -1.1 will probably never be
%meaningful. You probably _don't_ want to threshold at 0.0, though. You
%can get higher recall with a lower threshold. You don't need to modify
%anything in non_max_supr_bbox, but you can.

[cur_bboxes, cur_confidences] = ...
    nonMaximum_Suppression(cur_bboxes, cur_confidences,size(img));
end


function [bboxes, confidences] = ...
    nonMaximum_Suppression(bboxes, confidences,img_size)

[is_maximum] = non_max_supr_bbox(bboxes, confidences, img_size);
confidences = confidences(is_maximum,:);
bboxes      = bboxes(is_maximum,:);
end


function [cur_confidences,cur_bboxes] = ...
    Detector(img, svmClassifier, hog_template_size,hog_cell_size)


% Your code should first convert each test image to grayscale if
% it is a coloured image.

% Then the HoG features need to be determined for the entire image with
% a single call to vl_hog for each scale.

% Then step over the HoG cells, taking groups of cells that are the
% same size as your learned template, and classify them. Do this for all sclaes.

% If the classification is above some confidence, keep the detection
% and then pass all the detections for an image to non-maximum
% suppression. 

% For your initial debugging, you can operate only at a
% single scale. A word of advise. Don't save all the HoGs. Save 
% the bounding boxes and confidences of only detected faces.
% This will speed up your detections.

cur_bboxes = zeros(0,4);
cur_confidences = zeros(0,1);

% convert image to grayscale if it's rgb
if size(img, 3) == 3
    img = rgb2gray(img);
end

% different scales used for extracting, can be changed

% calculate the size of patch based on template and cell sizes
size_patch = hog_template_size / hog_cell_size;

% extract weights and bias
w = svmClassifier.weights;
b = svmClassifier.bias;

scale = 1.0;
% iterate over all scales
for i = 1:30
    scale = scale * 0.9;
    % scale the image
    scaled_img = imresize(img, scale);
   
    for pad_x = 1:hog_cell_size-1
        for pad_y = 1:hog_cell_size-1
            % calculate the hog for whole scaled image
            hog = vl_hog(im2single(scaled_img(pad_x:end, pad_y:end)),...
                hog_cell_size);
            % get dimensions of hog matrix
            [size_y, size_x, ~] = size(hog);

            % iterate over all patches
            for x = 1:(size_x - size_patch + 1)
                for y = 1:(size_y - size_patch + 1)
                    % get hog of the patch from hog of the image
                    hog_of_patch = hog(y:y+size_patch-1, x:x+size_patch-1, :);
                    % vectorize the hog for multiplication
                    hog_row_vector = hog_of_patch(:)';
                    % check for face detection      
                    confidence = hog_row_vector * w + b;
                    if confidence >= 2
                        cur_bboxes = [pad_x + (x-1)*hog_cell_size/scale,...
                            pad_y+(y-1)*hog_cell_size/scale,...
                            pad_x-1+(hog_template_size+(x-1)*hog_cell_size)/scale,...
                            pad_y-1+(hog_template_size+(y-1)*hog_cell_size)/scale...
                            ; cur_bboxes];
                        cur_confidences = [ confidence ; cur_confidences];
                    end

                end
            end
        end    
    end
end
    

[cur_bboxes, cur_confidences] = ...
    nonMaximum_Suppression(cur_bboxes, cur_confidences,size(img));

end


