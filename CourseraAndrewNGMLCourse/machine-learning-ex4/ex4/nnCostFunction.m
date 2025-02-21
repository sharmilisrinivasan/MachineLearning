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

% Convert y into vector of (m * K)
vecY = zeros(m,num_labels);
for record=1:m
	index = y(record,1);
	vecY(record,index) = 1;
endfor

% Initialize Delta Accumulators
DA1 = zeros(size(Theta1)); %25*401
DA2 = zeros(size(Theta2)); %10*26

for record=1:m

	%Forward Propagation
	a1 = X(record,:); % 1*400
	a1 = [1 a1]; %1*401
	z2 = a1 * Theta1'; %1*25
	a2 = [1 sigmoid(z2)]; %1*26
	z3 = a2 * Theta2'; %1*10
	a3 = sigmoid(z3); %1*10

	% Calculation of J 
	workableY = vecY(record,:); %1*10 (K)
	val1 =  -1 * workableY * log(a3'); %1*1
	val2 = -1 * (1-workableY) * log(1 - a3'); %1*1
	J = J + (val1 + val2) ;

	% Calculation of delta(d)
	d3 = a3 - workableY; %1*10
	d3 = d3'; %10*1
	term1 = Theta2' * d3; %26*1
	term2 = sigmoidGradient(z2'); %25*1
	d2 = term1 .* [1;term2]; %26*1

	% Updating Delta Accumulators
	DA2 = DA2 + (d3 * a2); %10*26
	intTerm = (d2 * a1); %26*401
	DA1 = DA1 + intTerm(2:end,:); %25*401

endfor

J = J/m;
Theta1_grad = DA1 / m; %25*401
Theta2_grad = DA2 / m; %10*26

% Adding regularization lambda

% To J
JRegTerm1 = Theta1(:,2:end).^2; %25*400
JRegTerm2 = Theta2(:,2:end).^2; %10*25
sumTerm = sum(JRegTerm1(:)) + sum (JRegTerm2(:));

J = J + ((lambda * sumTerm) / (2*m) );

% To Theta
int_reg1 = (lambda * Theta1(:,2:end))/m; %25*400
reg1 = [zeros(size(Theta1,1),1) int_reg1]; %25*401
int_reg2 = (lambda * Theta2(:,2:end))/m; %10*25
reg2 = [zeros(size(Theta2,1),1) int_reg2]; %10*26

Theta1_grad = (Theta1_grad + reg1);
Theta2_grad = (Theta2_grad + reg2);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
