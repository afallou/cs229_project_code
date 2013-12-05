function [video_matrix, resampling_rate] = process_images(dir_path)

	COLOR_CHANNELS_COUNT = 3;
	% Process a series of bmp images to make a movie out of them
	% dir path: path to directory containing all images forming one 'movie'

	dir_struct = dir(dir_path);

	% First two elements are '.' and '..', we remove them
	dir_struct(1) = [];
	dir_struct(1) = [];

	image_count = length(dir_struct);
	first_image = imread([dir_path, dir_struct(1).name]);

	images_height = size(first_image, 1);
	images_width = size(first_image, 2);

	video_matrix = zeros(images_height, images_width, image_count, COLOR_CHANNELS_COUNT);
	file_date = zeros(image_count, 1);

	% Filling the video 4D-matrix
	parfor image_index = 1:image_count
		current_image = imread([dir_path, dir_struct(image_index).name]);
		for i = 1:3
			video_matrix(:, :, image_index, i) = current_image(:, :, i);
		end

		image_name = dir_struct(image_index).name;
		splitted_name = regexp(image_name, '_', 'split');
		file_date(image_index) = str2num(splitted_name{2});
	end

	% Resampling
	file_date = file_date - file_date(1);
	% 1000 constant because dates are in ms
	resampling_rate = 1000 * image_count / file_date(end);
	% We'll get the same number of images, but with a constant sampling rate
	resampling_vector = linspace(0, file_date(end), image_count);

	% Interpolate for each pixel and color channel
	for color_channel = 1:3	
		for i = 1:images_height		
			parfor j = 1:images_width
				video_matrix(i, j, :, color_channel) = interp1(file_date, squeeze(video_matrix(i, j, :, color_channel)), resampling_vector);
			end
		end
	end
end