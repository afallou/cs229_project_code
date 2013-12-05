function [training_matrix, training_examples_matrices_array] = oximetry_linear_regression(training_examples_filenames_vector, feature_vector_size)
	training_examples_matrices_array = input_videos(training_examples_filenames_vector);

	training_matrix = zeros(0,0);
	for i = 1:length(training_examples_matrices_array)
		features_matrix = feature_vectors(training_examples_matrices_array{i}, feature_vector_size);
		training_matrix = cat(1, training_matrix, features_matrix);
	end

	% ... do it for the oximeter output, and do the regression
end