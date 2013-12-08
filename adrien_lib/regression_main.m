% [training_matrix, training_examples_matrices_array] = oximetry_linear_regression({'/Users/adrien/Dropbox/Stanford/Massive big data/data/adrien_face_2.mp4'}, 5);

% [video_matrix, resampling_rate, pulseox_resampled] = process_images('/Users/adrien/Dropbox/Stanford/Massive big data/cs229_project_code/sock_and_tupperware/nullimages/');
% load('vid_bmp.mat');

full_video_matrix = video_matrix;
video_matrix = video_matrix(100:260, 140:260, :, :);

COLOR_CHANNELS_COUNT = 3;
THRESHOLD_VIDEO = 0;
THRESHOLD_PULSEOX = 0;

video_fft = fft(video_matrix, [], 3);
video_width = size(video_fft, 2);
video_height = size(video_fft, 1);
video_length = size(video_fft, 3);
total_pixel_count = video_height * video_width;
video_fft = reshape(video_fft, total_pixel_count, video_length, COLOR_CHANNELS_COUNT);

first_binned_vector = binned_feature_vectors(video_fft(1, :, 1), THRESHOLD_VIDEO);
binarized_video = zeros(total_pixel_count, size(first_binned_vector), COLOR_CHANNELS_COUNT);

for color_channel_index = 1:3
	parfor i = 1:total_pixel_count
		binarized_video(i, :, color_channel_index) = binned_feature_vectors(abs(video_fft(i, :, color_channel_index)), THRESHOLD_VIDEO);
	end
end

feature_vectors = zeros(total_pixel_count, 3 * size(binarized_video, 2));
parfor i = 1:total_pixel_count
	feature_vectors(i, :) = [binarized_video(i, :, 1), binarized_video(i, :, 2), binarized_video(i, :, 3)];
end

binned_pulseox = binned_feature_vectors(fft(pulseox_resampled), THRESHOLD_PULSEOX);