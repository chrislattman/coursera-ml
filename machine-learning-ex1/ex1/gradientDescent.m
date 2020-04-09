function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    %J_1 and J_2 are the "partial derivative" values for gradient descent
    J_1 = 0;
    J_2 = 0;
    for i = 1:m
        J_1 = J_1 + ((theta(1) * X(i, 1) + theta(2) * X(i, 2)) - y(i)) * X(i, 1);
        J_2 = J_2 + ((theta(1) * X(i, 1) + theta(2) * X(i, 2)) - y(i)) * X(i, 2);
    end

    temp1 = theta(1) - alpha * (1 / m) * J_1;
    temp2 = theta(2) - alpha * (1 / m) * J_2;
    theta(1) = temp1;
    theta(2) = temp2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
