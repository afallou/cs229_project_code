function [binarized_vector] = binned_feature_vectors(frequency_domain_input, threshold_value, bins_count)
	vector_length = length(frequency_domain_input);
 	% positive_frequencies_input = frequency_domain_input(1:ceil(vector_length / 2));
	binned_vector = zeros(bins_count, 1);

	% Number of points in each bin
	bins_size = floor(vector_length / (bins_count));
    % fft_magnitude = abs(positive_frequencies_input);
    fft_magnitude = abs(frequency_domain_input);

	for i = 0:bins_count - 1;
		binned_vector(i + 1) = sum(fft_magnitude(i * bins_size + 1:(i+1) * bins_size) .* frequency_domain_input(i * bins_size + 1:(i+1) * bins_size)) ...
                                                     / sum(fft_magnitude(i * bins_size + 1:(i+1) * bins_size));
	end

	binarized_vector = binned_vector .* (binned_vector > threshold_value);