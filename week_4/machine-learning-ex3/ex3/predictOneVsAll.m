function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

# X * all_theta'           # multiply each training example to all parameters (transposed) generating the weighted (parameterized) output
                           # each row of all_theta' are the parameters for a specific class. all_theta is just a set of ten logistic regressions
                           # for each training example, you multiply the first all_theta row to get the probability of the sample being
                           # from the first class. You multiply the second row of all_theta to get the probability of the sample being from
                           # the second class and so on.
# sigmoid(X * all_theta')' # apply sigmoid to the weighted output and transpose it,
                           # generating a matrix K x m where K are the classes predicted probability and m are the training features 

# max() returns the max value of every row, which is, in this case, the classes predicted probability.
# that is irrelevant, so we use _ to designet its irrelevance. But max() also returns the position of the highest value in each row,
# which is what we want, the highest position is the same as the most probable class that a specific training example belongs.
# That is what p is returning, a vector of the predicted classes, e.g. p = [1, 2, 2, 4, 3, 1...]
[_ p] = max(sigmoid(X * all_theta')');


% =========================================================================


end
