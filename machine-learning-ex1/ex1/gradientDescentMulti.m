function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
features = size(X, 2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    J = zeros(features, 1);
    for i = 1:m
        inner = 0;
        for j = 1:features
            inner = inner + theta(j) * X(i, j);
        end
        
        for k = 1:features
            J(k) = J(k) + (inner - y(i)) * X(i, k);
        end
    end
    
    temp = zeros(features, 1);
    for n = 1:features
        temp(n) = theta(n) - alpha * (1 / m) * J(n);
    end
    
    for p = 1:features
        theta(p) = temp(p);
    end

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
