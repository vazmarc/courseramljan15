function [theta, theta_history, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_history = zeros(num_iters, 3);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
	hypothesis = [];
	hypothesis = X * theta;
	error = hypothesis .- y;
    newTheta = theta .- (alpha/m).*(X' * error);	

	%for theta_iter = 1:3
	%	if(!((((newTheta(theta_iter) - theta(theta_iter))/theta_iter)< 0.5) && (((theta(theta_iter) - newTheta(theta_iter))/theta_iter)< 0.5)))
	%		convergence = false;
	%	end
	%end
	%if(convergence)
	%	break;
	%end
	theta = newTheta;
    % ============================================================
	
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
%	theta_history(iter,:) = newTheta';
end
%for iter = 1: m
%	fprintf(' Y = %.0f, hypothesis = %.0f error = %.0f\n', [y(iter) hypothesis(iter) error(iter)]');
%end
	mean(error)
end
