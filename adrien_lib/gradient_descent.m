function [theta, Y, X] = gradient_descent(X, Y)
	training_examples_count = size(X, 1);

	X_T = X';
	features_count = size(X, 2);
	theta = zeros(features_count, length(Y));
	LEARNING_RATE = 0.9;

    for i = 1:training_examples_count
	
        for j = 1:length(Y) 
% 			theta_current_row = zeros(1, size(theta, 1));
			for  k = 1:features_count
				theta(k, j) = theta(k, j) + LEARNING_RATE * (Y(j) - X(i, :) * theta(:, j)) * X(i, j); 
			end
% 			theta(j, :) = theta_current_row;
		end
	end
end