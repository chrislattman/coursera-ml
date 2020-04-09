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

features = size(X, 2);
for i = 1:m
    thetaX = 0;
    for j = 1:features
        thetaX = thetaX + theta(j) * X(i, j);
    end
    
    J = J - (y(i) * log(sigmoid(thetaX)) + (1 - y(i)) * ...
        log(1 - sigmoid(thetaX)));
    
    for k = 1:features
        grad(k) = grad(k) + (sigmoid(thetaX) - y(i)) * X(i, k);
    end
end

for n = 2:features
    J = J + lambda / 2 * theta(n)^2;
    grad(n) = grad(n) + lambda * theta(n);
end

J = J / m;
grad = grad / m;

% =============================================================

end
