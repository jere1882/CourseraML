function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    new_theta = theta    
    for theta_idx = 1:length(theta)
        sum = 0
    
        for i = 1:m
            h_i = theta' * ( X(i,:)');
            diff = h_i - y(i);
            sum = sum + diff*X(i,theta_idx)* alpha/m
        end
        new_theta(theta_idx) = theta(theta_idx) - sum;
    end
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    theta = new_theta

end
