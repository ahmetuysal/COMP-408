function H_best = RANSACFit(p1, p2, match, seedSampleSize, maxInlierError, goodFitThresh )

%RANSACFit Use RANSAC to find a robust affine transformation
% Input:
%   p1: N1 * 2 matrix, each row is a point
%   p2: N2 * 2 matrix, each row is a point
%   match: M * 2 matrix, each row represents a match [index of p1, index of p2]
%   seedSampleSize: The number of randomly-chosen seed points that we'll use to fit
%   maxInlierError: A match not in the seed set is considered an inlier if
%                   its error is less than maxInlierError. Error is
%                   measured as sum of Euclidean distance between transformed
%                   point1 and point2. You need to implement the
%                   ComputeCost function.
%
%   goodFitThresh: The threshold for deciding whether or not a model is
%                  good; for a model to be good, at least goodFitThresh
%                  non-seed points must be declared inliers.

% Output:
%   H: a robust estimation of affine transformation from p1 to p2
%


%% Initializations

N = size(match, 1);
if N<3
    error('not enough matches to produce a transformation matrix')
end

if ~exist('seedSetSize', 'var'),
    seedSampleSize = ceil(0.2 * N);
end

if ~exist('maxInlierError', 'var'),
    maxInlierError = 30;
end

if ~exist('goodFitThresh', 'var'),
    goodFitThresh = floor(0.7 * N);
end


%================================================================
%           PART A: NUMBER OF ITERATIONS REQUIRED
%================================================================

% Determine the number of iterations required to ensure a good fit
% and save it in the variable maxIter;
% Refer to lecture slides for the method proposed by David Lowe.

% Assume that the fraction of inliers equlas 0.7.
% Using the seedSampleSize defined above, determine the number of
% iterations required which ensures the probability that all samples
% fail is below 10e-03.

%probability that all samples fail
p_fail = 0.001;

%fraction of inliers
w = 0.7;

%seed sample size
n = seedSampleSize;

%number of iterations
maxIter=[];

% p_fail = (1 - w^n)^k => k = ln(p_fail) / ln(1-w^n)
% We can safely use log since we know p_fail and 1-w^n are positive

maxIter = ceil(log(p_fail)/log(1 - w^n));

%================================================================
%           PART B:RANSAC ITERATIONS
%================================================================

H_best = eye(3);
min_error = inf;

for i = 1 : maxIter
    
    % Randomly select a seed group
    idx = randperm(size(match, 1));
    seed_group = match(idx(1:seedSampleSize), :);
    
    % Select remaining as the non-seed group
    non_seed_group = match(idx(seedSampleSize+1:end), :);
    
    % ===================================================
    % Step 1:
    % ===================================================
    % Use seed_group to compute the transformation matrix. Use the     %
    % function ComputeAffineMatrix below. It is partially coded. You need
    % to complete it.
    
    H = ComputeAffineMatrix( p1(seed_group(:, 1), :), p2(seed_group(:, 2), :) );
    
    % ===================================================
    % Step 2:
    % ===================================================
    % Use non_seed_group to compute error from eucledian distance i.e.
    % ||p1'-H*p||. Use the ComputeError function below. It is partially
    % coded. You need to complete it.
    
    err = ComputeError(H, p1(non_seed_group(:, 1), :), p2(non_seed_group(:, 2),:));
    
    %=======================================================
    % Step 3:
    %=======================================================
    %  . Select the points as inliers which have error less than
    %    maxInlierError. Save them in variable 'inliers'
    
    inliers = [];
   
    for index = 1:length(non_seed_group)
       if (err(index) < maxInlierError)
           inliers = [inliers; non_seed_group(index, :)];
       end
    end
    
    
        
    number_of_inliers = size(inliers,1) + size(seed_group,1);
    if( number_of_inliers > goodFitThresh )
        
        all_inliers = [seed_group ; inliers];
        H = ComputeAffineMatrix( p1(all_inliers(:, 1), :), p2(all_inliers(:, 2), :) );
        err = ComputeError(H, p1(all_inliers(:, 1), :), p2(all_inliers(:, 2),:));
        % I used sum of squares of errors to evaluate H
        sum_err_squares = err' * err;
        
        if(sum_err_squares < min_error)
            min_error = sum_err_squares;
            H_best = H;
        end
    end
end

if sum(sum((H_best - eye(3)).^2)) == 0,
    disp('No RANSAC fit was found.')
end
end


function H = ComputeAffineMatrix( Pt1, Pt2 )
%ComputeAffineMatrix
%   Computes the transformation matrix that transforms a point from
%   coordinate frame 1 to coordinate frame 2
%Input:
%   Pt1: N * 2 matrix, each row is a point in image 1
%       (N must be at least 3) ----> %Ahmet: Why not 4?
%   Pt2: N * 2 matrix, each row is the point in image 2 that
%       matches the same point in image 1 (N should be more than 3)
%Output:
%   H: 3 * 3 affine transformation matrix,
%       such that H*pt1(i,:) = pt2(i,:)

N = size(Pt1,1);
if size(Pt1, 1) ~= size(Pt2, 1),
    error('Dimensions unmatched.');
elseif N<3
    error('At least 3 points are required.');
end

% Convert the input points to homogeneous coordintes.
P1 = [Pt1';ones(1,N)];
P2 = [Pt2';ones(1,N)];

% Now, we must solve for the unknown H that satisfies H*P1=P2
% But MATLAB needs a system in the form Ax=b, and A\b solves for x.
% In other words, the unknown matrix must be on the right.
% But we can use the properties of matrix transpose to get something
% in that form. Just take the transpose of both sides of our equation
% above, to yield P1'*H'=P2'. Then MATLAB can solve for H', and we can
% transpose the result to produce H.

% P1'*H' = P2' -> H' = P1'\P2'
H_transpose = P1'\P2';
H = H_transpose';

% Sometimes numerical issues cause least-squares to produce a bottom
% row which is not exactly [0 0 1], which confuses some of the later
% code. So we'll ensure the bottom row is exactly [0 0 1].
H(3,:) = [0 0 1];
end


function dists = ComputeError(H, pt1, pt2)
% Compute the error using transformation matrix H to
% transform the point in pt1 to its matching point in pt2.
%
% Input:
%   H: 3 x 3 transformation matrix where H * [x; y; 1] transforms the point
%      (x, y) from the coordinate system of pt1 to the coordinate system of
%      pt2.
%   pt1: N1 x 2 matrix where each ROW is a data point [x_i, y_i]
%   pt2: N2 x 2 matrix where each ROW is a data point [x_i, y_i]
%   match: M x 2 matrix, each row represents a match [index of pt1, index of pt2]
%
% Output:
%    dists: An M x 1 vector where dists(i) is the error of fitting the i-th
%           match to the given transformation matrix.
%           Error is measured as the Euclidean distance between (transformed pt1)
%           and pt2 in homogeneous coordinates.


dists = zeros(size(pt1,1),1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                                YOUR CODE HERE.                               %
%           Convert the points to a usable format, perform the                 %
%           transformation on pt1 points, and find their distance to their     %
%           MATCHING pt2 points.                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hint: If you have an array of indices, MATLAB can directly use it to
% index into another array. For example, pt1(match(:, 1),:) returns a
% matrix whose first row is pt1(match(1,1),:), second row is
% pt1(match(2,1),:), etc. (You may use 'for' loops if this is too
% confusing, but understanding it will make your code simple and fast.)

% Convert the input points to homogeneous coordintes.
pt1_homogeneous = [pt1' ; ones(1, length(pt1))];
pt2_homogeneous = [pt2' ; ones(1, length(pt2))];

pt1_transformed = (H * pt1_homogeneous);

distances = pt1_transformed - pt2_homogeneous;

distances_x = distances(1, :);
distances_y = distances(2, :);
% distance_z is always all zero


% transpose the vector to satify documentation (return M x 1 matrix). 
dists = sqrt(distances_x .* distances_x + distances_y .* distances_y)';


if size(dists,1) ~= size(pt1,1) || size(dists,2) ~= 1
    error('wrong format');
end
end





