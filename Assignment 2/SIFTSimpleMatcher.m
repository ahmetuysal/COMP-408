function match = SIFTSimpleMatcher(descriptor1, descriptor2, thresh)
% SIFTSimpleMatcher 
%   Match one set of SIFT descriptors (descriptor1) to another set of
%   descriptors (decriptor2). Each descriptor from descriptor1 can at
%   most be matched to one member of descriptor2, but descriptors from
%   descriptor2 can be matched more than once.
%   
%   Matches are determined as follows:
%   For each descriptor vector in descriptor1, find the Euclidean distance
%   between it and each descriptor vector in descriptor2. If the smallest
%   distance is less than thresh*(the next smallest distance), we say that
%   the two vectors are a match, and we add the row [d1 index, d2 index] to
%   the "match" array.
%   
% INPUT:
%   descriptor1: N1 * 128 matrix, each row is a SIFT descriptor.
%   descriptor2: N2 * 128 matrix, each row is a SIFT descriptor.
%   thresh: a given threshold of ratio. Typically 0.7
%
% OUTPUT:
%   Match: N * 2 matrix, each row is a match.
%          For example, Match(k, :) = [i, j] means i-th descriptor in
%          descriptor1 is matched to j-th descriptor in descriptor2.
    if ~exist('thresh', 'var'),
        thresh = 0.7;
    end

    match = [];
    
    for i = 1:length(descriptor1)
        sift_desc_1 = descriptor1(i, :);
        %initilialize distances with infinity
        smallest = inf;
        second_smallest = inf;
        %initialize smallest index with zero
        smallest_index = 0;
        % this implementation assumes there are at least two descriptors
        % in second array, smallest variables are wrong until first two
        % iterations
        for j = 1: length(descriptor2)
            sift_desc_2 = descriptor2(j, :);
            distances = sift_desc_1 - sift_desc_2;
            % dot product of row vector of distances with its transpose is
            % equivalent with sum of squares of distances
            euclidian_dist = sqrt(distances * distances');
            
            if (euclidian_dist < second_smallest)
                second_smallest = euclidian_dist;
            end
            
            if (euclidian_dist < smallest)
                second_smallest = smallest;
                smallest = euclidian_dist;
                smallest_index = j;
            end
        end
        
        if (smallest < second_smallest * thresh)
            match = [match; [i, smallest_index]];
        end
    end
end
