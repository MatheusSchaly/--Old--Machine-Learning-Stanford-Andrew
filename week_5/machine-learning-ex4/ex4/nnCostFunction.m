function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

# Part 1:

# For a better exaplaination of forward propagation check predict.m from week 4 exercise
X = [ones(m, 1) X];
z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

# a3 (h0(x)) are the predicted values, rows are examples and columns are classes. Each unit 
# of this matrix is the probability from 0 to 1 of a example m belonging to a class k


% The vector y passed into the function is a vector of labels
% containing values from 1..K. This will map the vector into a 
% binary vector of 1's and 0's to be used with the neural network
% cost function.

# yv = eye(num_labels)(y,:)
# yv = yv' # yv is equal to matrixY, this code was found in the resources section of week 5

matrixY = []; # create the base matrix that will contain the y vectors
for i = 1:m
  yIndex = y(i); # pick the ground truth of the ith example
  newColumn = zeros(num_labels, 1); # create a new vector of zeros
  newColumn(yIndex) = 1; # places the ground truth (1) in the correct y index 
  matrixY = [matrixY newColumn]; # concatenate the newColumn to the matrix of y vectors
end

# Vectorized implementation of feedforward cost function (I took a few hours to do this single line):
# By multiplying matrixY' .* log(a3) we get a matrix m x k which the rows are the cost of
# each predicted example and columns are the classes. But as the cost function is given only 
# by the difference between the ground truth index and our predicted probability for that same index
# we end up with a matrix where each row has only one value that is not zero. Because there is only
# one ground truth index, i.e. our example can only belong to one class. After multiplying, we have
# to sum every row, so that we get rid of those irrelevant zeros. Then we take the difference,
# take the mean (1/m), which result in a vector m x 1 that cointains the cost for each
# training example. Finally, we sum all those erros and get the final cost of our neural network

# Regarding the regularization term, we have to sum up all values from our first theta, square it,
# and sum with our second theta (or how many thetas there are). Remember to remove
# the bais weight (:, 2:end). Them we just multiply the result by the regularization parameter
# (lambda) divided by (2 * m). Now we have a vectorized implementation of a regularized 
# feedforward cost function

J = sum((1 / m) * (sum(-matrixY' .* log(a3), 2) - (sum((1 - matrixY)' .* log(1 - a3), 2)))) + ((lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end).^2)) + (sum(sum(Theta2(:, 2:end).^2)))));



# Part 2:

# error3 (a.k.a. errorL where L = last layer) is the induced error (lowercase delta) from the predicted values 
# (a3) and the actual values (matrixY). The errorL rows represent the classes (k). The columns 
# are each example (m), and the unit is the error between what we predicted and the ground truth

error3 = a3' - matrixY;

# Now we get the induced error of the second layer, note that the first layer doesn't have an 
# error because it is the input layer. We must remove the indiced error from the bias node (2:end, :).

error2 = (Theta2(:, 2:end)' * error3) .* sigmoidGradient(z2)';

# Not sure about the following steps:
# Are these Theta_grad matrices of errors with respect to the training set? Are these the accumulated nn gradients?
Theta1_grad = error2 * X;
Theta2_grad = error3 * a2;

# Are these our regularized gradients?
Theta1_grad = Theta1_grad./m + ((lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)]);
Theta2_grad = Theta2_grad./m + ((lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)]);



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
