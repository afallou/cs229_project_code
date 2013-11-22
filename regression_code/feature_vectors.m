function [features_matrix] = feature_vectors(video_matrix, feature_vector_size)
	% Compute the feature vector for each pixel in a video, from its matrix representation
	%
	% video_matrix: 4-dimensional matrix representation of the video
	% features_matrix: the 2-D matrix with one training example (pixel) per row

	% time is the 4th dimension of each video matrix
	video_matrix_fft =  fft(video_matrix, [], 4);

	% Taking the Fourier transform components with biggest module, these will be our features (for each color)
	[~, max_indices] = maxk(abs(video_matrix_fft), feature_vector_size, 4);

	% From the result of maxk, we have to get the actual elements
	% TODO(adrien): avoid using loop
	pixel_maximum_elements = zeros(size(video_matrix, 1), size(video_matrix, 2), 3, feature_vector_size);
	for k = 1:3
		for i = 1:size(video_matrix, 1)
			for j = 1:size(video_matrix, 2)
				pixel_maximum_elements(i, j, k, :) = video_matrix_fft(i, j, k, squeeze(max_indices(i, j, k, :)));
			end
		end
	end
	
	% If you have a doubt about the result, check out:
	% http://www.mathworks.com/help/matlab/math/multidimensional-arrays.html
	% for details about linear indices for multidimensional arrays in matlab
	features_matrix_multidimensional = reshape(pixel_maximum_elements, [], 3, feature_vector_size);

	% Because 'color' is not the last dimension we can't use reshape here
	features_red = squeeze(features_matrix_multidimensional(:, 1, :));
	features_green = squeeze(features_matrix_multidimensional(:, 2, :));
	features_blue = squeeze(features_matrix_multidimensional(:, 3, :));

	features_matrix = cat(2, features_red, features_green, features_blue);
end