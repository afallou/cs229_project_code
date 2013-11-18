[video_matrices_array] = function input_videos(filenames_vector)
	% Get 4-D matrices from the (training or test) input videos
	%
	% filenames_vector: vector of the filename strings for all training videos
	% video_matrices_array: cell array containing the matrix-representation of each video. 
	% Each matrix has size height x width x nb_color_channels x nb_frames

	% Videos may have different height and width >> using a cell array
	video_matrices_array = cell(length(filenames_vector), 1)

	% read the videos in
	for i = 1:length(filenames_vector)
		video_reader = VideoReader(filenames_vector(i))
		video_matrix = read(video_reader)
		video_matrices_array[i] = video_matrix
	end
end