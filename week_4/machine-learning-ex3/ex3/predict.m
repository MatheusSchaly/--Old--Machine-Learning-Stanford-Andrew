function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

# Add ones to the X data matrix (bias unit)
X = [ones(m, 1) X];

z2 = X * Theta1'; # multiply each training example (rows of X) by every class weights (columns of Theta1')
                  # generating a matrix where each row is a example and each column is the class pre probability
                  # i.e. X[1,0] = pre probability of example 1 belonging to class 0.
a2 = sigmoid(z2); # apply sigmoid activation function to generate results between 0 and 1 (real probability)
                  # a1 will be the input for the second layer (a2 is the new X)         

# Repeat the process
a2 = [ones(m, 1) a2]; 
z3 = a2 * Theta2';
a3 = sigmoid(z3);

# Generate p output vector, for a more detailed explaination check predictOneVsAll.m
[_, p] = max(a3, [], 2);

% =========================================================================


end
