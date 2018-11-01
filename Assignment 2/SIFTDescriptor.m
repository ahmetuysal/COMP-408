function descriptors = SIFTDescriptor(pyramid, keyPt, keyPtScale)

% SIFTDescriptor builds SIFT descriptors from image at detected key points'
% location with detected key points' scale
%
% INPUT:
%   pyramid: Image pyramid. pyramid{i} is a rescaled version of the
%            original image, in grayscale double format
%
%   keyPt: N * 2 matrix, each row is a key point pixel location in
%   pyramid{round(scale)}. So pyramid{round(scale)}(y,x) is the center of the keypoint
%
%   scale: N * 1 matrix, each entry holds the index in the Gaussian
%   pyramid for the keypoint. Earlier code interpolates things, so this is a
%   double, but we'll just round it to an integer.
%
% OUTPUT:
%   descriptors: N * 128 matrix, each row is a feature descriptor
%
%================================================================================================


%% Initializations

% Number of keypoints that were found by the DoG blob detector
N = size(keyPt, 1);

% For each keypoint we will extract a region of size
% patch_size x patch_size centered at the keypoint.
patch_size = 16;

% The patch extracted around each keypoint will be divided into a grid
% of grid_size x grid_size.
grid_size = 4;

% Each histogram covers an area "pixelsPerCell" wide and tall
pixelsPerCell= patch_size / grid_size;

% The number of orientation bins per cell
num_bins = 8;

% Initialize descriptors to zero
descriptors = zeros(N, grid_size*grid_size*num_bins);


%====================================================================

% STEP 1: DETERMINE GRADIENT ANGLES AND MAGNITUDES OF ALL PYRAMIDS

%====================================================================

%   To determine the gradient angles and magnitude, use the function
%   ComputeGradient(). This is partially coded so you must complete
%   it first.

%   ComputeGradient() will take the pyramid images as input
%   and output the gradient magnitudes and angles in two cell arrays
%   of size equal to number of pyramid images.


grad_mag = cell(length(pyramid),1);
grad_theta = cell(length(pyramid),1);

[grad_mag,grad_theta] = ComputeGradient(pyramid);




% Iterate over all keypoints
for i = 1 : N
    
    % ==============================================================
    
    %  STEP2: NORMALIZE GRADIENT DIRECTION IN A PATCH
    
    %===============================================================
    
    % Extract a patch of magnitudes and directions (16x16) around the
    % keypoint using the function Extract_Patch(). This has been done
    % for you.
    
    % Use the function Normalize_Orientation() to normalize the
    % gradient directions relative to the dominant gradient direction.
    % This function is partially coded. You are required to complete it.
    
    scale = round(keyPtScale(i));
    magnitudes = grad_mag{scale};
    thetas = grad_theta{scale};
    
    [patch_mag,patch_theta] = Extract_Patch(keyPt(i,:),patch_size,magnitudes,thetas);
    
    if( isempty(patch_mag))
        continue;
    end
    
    patch_theta = Normalize_Orientation(patch_mag, patch_theta);
    
    
    
    % ==========================================================
    
    %  STEP3: EXTRACT SIFT DESCRIPTOR
    
    %===========================================================
    
    
    % The patch we have extracted should be subdivided into
    % grid_size x grid_size cells, each of which is size
    % pixelsPerCell x pixelsPerCell.
    
    % Compute a gradient histogram for each cell, and concatenate
    % the histograms into a single feature vector of length 128.
    
    % Please traverse the patch row by row, starting in the top left,
    % in order to match the given solution. E.g. if you use two
    % nested 'for' loops, the loop over x should  be the inner loop.
    
    % Initializing the feature vector to size 0. Hint: you can
    % concatenate the histogram descriptors to it like this:
    % feature = [feature, histogram]
    
    % Complete the code for the function ComputeSIFTDescriptor()
    
    % Weight the gradient magnitudes using a gaussian function
    patch_mag = patch_mag .* fspecial('gaussian', patch_size, patch_size / 2);
    
    feature = ComputeSIFTDescriptor...
        (patch_mag,patch_theta,grid_size,pixelsPerCell,num_bins);
    
    % Add the feature vector we just computed to our matrix of SIFT descriptors.
    descriptors(i, :) = feature;
end

% Normalize the descriptors.
descriptors = NormalizeDescriptors(descriptors);
end


function [grad_mag,grad_theta] = ComputeGradient(pyramid)

%   Input:
%   pyramid: all the pyramid images in a cell array
%
%   Output:
%   grad_mag, grad_mag:
%       Two cell arrays of the same shape where grad_mag{i} and
%       grad_theta{i} give the magnitude and direction of the i-th
%       pyramid image. Gradient_angles ranges from 0 to 2*pi.


% For all the pyramid images do the following:
% Step 1:
%   Use the matlab function filter2 with filter [-1 0 1] to fill in img_dx
%   (gradient in x-direction), and the filter [-1;0;1] to fill in img_dy
%   (gradient in y-direction).

% Step 2:
%   Determine the angle and magnitude using the img_dx and img_dy
%   variables. The atan2 function will be helpful for calculating angle

% Step 3:
%   Compute edge orientation (angle of the gradient - 90°) for each
%   surviving pixel

grad_theta = cell(length(pyramid),1);
grad_mag = cell(length(pyramid),1);

%Do this for all pyramid images
for scale = 1:length(pyramid)
    
    currentImage = pyramid{scale};
    grad_mag{scale} = zeros(size(currentImage));
    grad_theta{scale} = zeros(size(currentImage));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                         %
    %                                YOUR CODE HERE
    %
    img_dx = filter2([-1 0 1], currentImage);
    img_dy = filter2([-1; 0; 1], currentImage);
    
    
    grad_mag{scale} = sqrt(img_dx.*img_dx + img_dy.*img_dy);
    grad_theta{scale} = atan2(img_dy, img_dx) - deg2rad(90);
    %                                                                         %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % atan2 gives angles from -pi to pi. To make the histogram code
    % easier, we'll change that to 0 to 2*pi.
    grad_theta{scale} = mod(grad_theta{scale}, 2*pi);
end
end


function norm_angles = Normalize_Orientation(gradient_magnitudes, gradient_angles)
% Computes the dominant gradient direction for the region around a keypoint
% given the gradient magnitudes and gradient angles of the pixels in the
% region surrounding the keypoint.

% INPUT
% gradient_magnitudes, gradient_angles:
%   Two arrays of the same shape where gradient_magnitudes(i) and
%   gradient_angles(i) give the magnitude and direction of the gradient for
%   the ith point.

% OUTPUT
% norm_angles:
%   the gradient angles for the same patch, but normalized with respect
%   to the dominant direction




% Step 1:
% Compute a gradient histogram using the weighted gradient magnitudes
% with the function ComputeGradientHistogram(). In David Lowe's paper
% he suggests using 36 bins for this histogram.

% Step 2:
% Find the maximum value of the gradient histogram, and set "direction"
% to the angle corresponding to the maximum. just use the lower-bound
% angle of the max histogram bin. (E.g. return 0 radians if it's bin 1.)

%Step3:
% Normalize the original gradient directions by subtracting the
% dominant direction

num_bins = 36;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                                                             %
%                                YOUR CODE HERE:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[histogram, angles] = ComputeGradientHistogram(num_bins,...
    gradient_magnitudes, gradient_angles);

[maximum, index] = max(histogram);

dominant_angle = angles(index);

norm_angles = gradient_angles - dominant_angle;

% This line will re-map norm_theta into the range 0 to 2*pi
norm_angles = mod(norm_angles, 2*pi);

end


function [histogram, angles] = ComputeGradientHistogram(num_bins, gradient_magnitudes, gradient_angles)
% Compute a gradient histogram using gradient magnitudes and directions.
% Each point is assigned to one of num_bins depending on its gradient
% direction; the gradient magnitude of that point is added to its bin.
%
% INPUT
% num_bins: The number of bins to which points should be assigned.
% gradient_magnitudes, gradient angles:
%       Two arrays of the same shape where gradient_magnitudes(i) and
%       gradient_angles(i) give the magnitude and direction of the gradient
%       for the ith point. gradient_angles ranges from 0 to 2*pi
%
% OUTPUT
% histogram: A 1 x num_bins array containing the gradient histogram. Entry 1 is
%       the sum of entries in gradient_magnitudes whose corresponding
%       gradient_angles lie between 0 and angle_step. Similarly, entry 2 is for
%       angles between angle_step and 2*angle_step. Angle_step is calculated as
%       2*pi/num_bins.

% angles: A 1 x num_bins array which holds the histogram bin lower bounds.
%       In other words, histogram(i) contains the sum of the
%       gradient magnitudes of all points whose gradient directions fall
%       in the range [angles(i), angles(i + 1))

angle_step = 2 * pi / num_bins;
angles = 0 : angle_step : (2*pi-angle_step);
histogram = zeros(1, num_bins);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                                YOUR CODE HERE:                               %
%
%
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[size_x, size_y] = size(gradient_angles);

for gradient_index_x = 1:size_x
    for gradient_index_y = 1:size_y
        grad_angle = gradient_angles(gradient_index_x, gradient_index_y);
        for angle_index = length(angles):-1:1
            if(grad_angle >= angles(angle_index))
                histogram(angle_index) = histogram(angle_index)...
                    + gradient_magnitudes(gradient_index_x, gradient_index_y);
                break
            end
        end
    end
end
end

function descriptor = ComputeSIFTDescriptor...
    (patch_mag,patch_theta,grid_size,pixelsPerCell,num_bins)

% INPUT:
%   patch_mag, patch_theta:
%       Two arrays of the same shape where patch_mag(i) and
%       patch_theta(i) give the magnitude and direction of the gradient
%       for the ith point. gradient_angles ranges from 0 to 2*pi

% OUTPUT:
%   descriptor: A 128 length descriptor calculated from 8-bin histograms
%   of the 16 cells in the grid

% initialize descriptor to size 0
descriptor = [];

% I first used this code but I changed it when I saw autograder gave error
% but, visual representation was accurate. So, I changed the order and 
% autograder said it is accepted. 

% for y = 1 : grid_size
%     cell_y_start = 1 + (y-1)*pixelsPerCell;
%     cell_y_end = y * pixelsPerCell;
%     for x = 1:grid_size
%         cell_x_start = 1 + (x-1)*pixelsPerCell;
%         cell_x_end = x * pixelsPerCell;
%         mag_cell = patch_mag(cell_x_start:cell_x_end,...
%             cell_y_start:cell_y_end);
%         theta_cell = patch_theta(cell_x_start:cell_x_end,...
%             cell_y_start:cell_y_end);
%         cell_gradient_histogram = ComputeGradientHistogram(num_bins, ...
%             mag_cell, theta_cell);
%         descriptor = [descriptor, cell_gradient_histogram];
%     end
% end

for x = 1 : grid_size
    cell_x_start = 1 + (x-1)*pixelsPerCell;
    cell_x_end = x * pixelsPerCell;
    for y = 1:grid_size
        cell_y_start = 1 + (y-1)*pixelsPerCell;
        cell_y_end = y * pixelsPerCell;
        mag_cell = patch_mag(cell_x_start:cell_x_end,...
            cell_y_start:cell_y_end);
        theta_cell = patch_theta(cell_x_start:cell_x_end,...
            cell_y_start:cell_y_end);
        cell_gradient_histogram = ComputeGradientHistogram(num_bins, ...
            mag_cell, theta_cell);
        descriptor = [descriptor, cell_gradient_histogram];
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                             YOUR CODE HERE:                                  %
%                                                                              %
%         Compute the gradient histograms and concatenate them in the          %
%  feature variable to form a size 1x128 SIFT descriptor for this keypoint.    %
%                                                                              %
%            HINT: Use the ComputeGradientHistogram function below.            %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end


function [patch_mag,patch_theta] = Extract_Patch(keyPt,patch_size,magnitudes,thetas)
% Find the window of pixels that contributes to the descriptor for the
% current keypoint.
xAtScale = keyPt(1);%center of the DoG keypoint in the pyramid{2} image
yAtScale = keyPt(2);
x_lo = round(xAtScale - patch_size / 2);
x_hi = x_lo+patch_size-1;
y_lo = round(yAtScale - patch_size / 2);
y_hi = y_lo+patch_size-1;

% These are the gradient magnitude and angle images from the
% correct scale level. You computed these above.

patch_mag = [];
patch_theta = [];
try
    % Extract the patch from that window around the keypoint
    patch_mag = magnitudes(y_lo:y_hi,x_lo:x_hi);
    patch_theta = thetas(y_lo:y_hi,x_lo:x_hi);
catch err
    % If any keypoint is too close to the boundary of the image
    % then we just skip it.
    
end
end


function descriptors = NormalizeDescriptors(descriptors)
% Normalizes SIFT descriptors so they're unit vectors. You don't need to
% edit this function.
%
% INPUT
% descriptors: N x 128 matrix where each row is a SIFT descriptor.
%
% OUTPUT
% descriptors: N x 128 matrix containing a normalized version of the input.

% normalize all descriptors so they become unit vectors
lengths = sqrt(sum(descriptors.^2, 2));
nonZeroIndices = find(lengths);
lengths(lengths == 0) = 1;
descriptors = descriptors ./ repmat(lengths, [1 size(descriptors,2)]);

% suppress large entries
descriptors(descriptors > 0.2) = 0.2;

% finally, renormalize to unit length
lengths = sqrt(sum(descriptors.^2, 2));
lengths(lengths == 0) = 1;
descriptors = descriptors ./ repmat(lengths, [1 size(descriptors,2)]);

end