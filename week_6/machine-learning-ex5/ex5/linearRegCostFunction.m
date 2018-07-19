function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

# Hurried explaination for J:
# Every training example in X will be multiplied by its parameter (X*theta) giving the predicted result
# then we take the difference of it and the ground truth (y) and square it. Then we sum this error
# and take the average (m). Then we add the regularization term to it.

J = ((1 / (2*m)) * (sum((X*theta-y) .^ 2))) + ((lambda/(2*m)) * sum(theta(2:end, 1) .^ 2));


# Regularization term is not applied at theta0

grad = ((1 / m) * X' * (X*theta-y)) + [0; ((lambda / m) * theta(2:end, :))];


% =========================================================================

grad = grad(:);

end
