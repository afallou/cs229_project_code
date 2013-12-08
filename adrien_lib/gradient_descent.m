function [theta] = gradient_descent(X, Y)
	training_examples_count = size(X, 1);

	X_T = X';
	features_count = size(theta, 1);
	theta = zeros(features_count, length(Y));
	LEARNING_RATE = 0.9;

	for i = 1:training_examples_count
		parfor j = 1:length(Y) 
			theta_current_row = zeros(1, size(theta, 1));
			for  k = 1:features_count
				theta_current_row(k) = theta_current_row(k) + LEARNING_RATE * (Y(j) - theta(:, j) * X(i, :)) * X(i, j); 
			end
			theta(j, :) = theta_current_row;
		end
	end
end