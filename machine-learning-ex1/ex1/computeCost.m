function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

for i = 1:m
    % J = J + (h_theta(x_i) - y_i)^2 where
    % h_theta(x_i) = theta_1 * X(i,1) + theta_2 * X(i,2)
    % in linear regression, X(i,1) is equal to 1 for all i
    J = J + ((theta(1) * X(i, 1) + theta(2) * X(i, 2)) - y(i))^2;
end

J = J * 1 / (2 * m);

% =========================================================================

end
