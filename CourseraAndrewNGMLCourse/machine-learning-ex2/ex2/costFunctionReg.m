function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

temptheta = theta(2:length(theta));

val1 = y' * log(sigmoid(X*theta));
val2 = (1-y') * log(1 - sigmoid(X*theta));

J = -(val1+val2)/m;
regvalJ = sum(temptheta.^2) * lambda / (2 * m);
J = J + regvalJ;

grad = (X' * (sigmoid(X*theta)-y))/m;
regvalgrad = (temptheta * lambda)/m;
grad = grad + [0;regvalgrad];


% =============================================================

end
