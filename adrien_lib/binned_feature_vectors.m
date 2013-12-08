function [binarized_vector] = binned_feature_vectors(frequency_domain_input, threshold_value)
	BINS_COUNT = 25;
	vector_length = length(frequency_domain_input);
	positive_frequencies_input = frequency_domain_input(1:ceil(vector_length / 2));
	binned_vector = zeros(BINS_COUNT, 1);

	% Number of points in each bin
	bins_size = floor(ceil(vector_length / 2) / BINS_COUNT);

	for i = 0:BINS_COUNT - 1;
		binned_vector(i + 1) = sum(positive_frequencies_input(i * bins_size + 1:(i+1) * bins_size)) / bins_size;
	end

	binarized_vector = binned_vector .* (binned_vector > threshold_value);