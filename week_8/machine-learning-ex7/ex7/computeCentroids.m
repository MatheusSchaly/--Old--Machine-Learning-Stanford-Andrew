function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%


counter = zeros(K, 1);

for i = 1:m
  # centroids at index (idx(i), :) keep stacking the examples X(i, :)
  centroids(idx(i), :) = centroids(idx(i), :) + X(i, :);
  # counter counts how many examples belongs to class (idx(i)) and
  # keep stacking them up
  counter(idx(i), :) = counter(idx(i), :) + 1;
end

# divides the stacked X from each class with the number of examples
# from each classes, returning the mean distance of each centroid
# related to its classes' examples
centroids = centroids ./ counter;


% =============================================================


end

