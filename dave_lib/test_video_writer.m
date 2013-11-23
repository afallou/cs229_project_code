% makes a video of white noise except for one channel (red), which has a known
% signal embedded
% dderiso@stanford.edu 2013

% -----------------------------------------------------------------------------
% setup

addpath('dave_lib')

% -----------------------------------------------------------------------------
% params

n_frames = 500;
vid_size = 10;
empty_frame = ones(vid_size, vid_size);
frame_rate = 30;

% -----------------------------------------------------------------------------
% make a heart beat = cosine wave of known frequency and sampling rate

x = hz_wave(frame_rate, 10, n_frames); % 10Hz wave
x = scale(x, 0, 255);
x = round(x);
% plot(x);

% -----------------------------------------------------------------------------
% make white noise for each frame
% for each pixel
%   set value over time to a white noise vector of length n_frames

wn = wgn(1, n_frames, 30); % white noise
wn = scale(wn, 0, 255);
round(wn);

noise_mat_r = ones(vid_size, vid_size, n_frames);
noise_mat_g = noise_mat_r;
noise_mat_b = noise_mat_r;
for i=1:vid_size
  for j=1:vid_size
    noise_mat_r(i,j,:) = round(scale(wgn(1, n_frames, 10), 0, 255))';
    noise_mat_g(i,j,:) = round(scale(wgn(1, n_frames, 30), 0, 255))';
    noise_mat_b(i,j,:) = round(scale(wgn(1, n_frames, 30), 0, 255))';
  end;
end;

% -----------------------------------------------------------------------------
% save video out

vid_out = VideoWriter('test.mp4');
vid_out.FrameRate = 30;
open(vid_out);

for i=1:500
  frame = a;
  frame(:,:,1) = noise_mat_r(:,:,i) + empty_frame*x(i);
  frame(:,:,2) = noise_mat_g(:,:,i);
  frame(:,:,3) = noise_mat_b(:,:,i);
  writeVideo(vid_out, im2uint8(rgb2ntsc(frame)));
end

close(vid_out);

% -----------------------------------------------------------------------------
% end

% -----------------------------------------------------------------------------
% verify 1

% the waveform + the noise has an FFT as expected

noisy_heart = reshape(noise_mat_r(1,1,:), n_frames, 1);
pwelch(noisy_heart + x(0:n_frames)');

% -----------------------------------------------------------------------------
% verify 2

% open video
vid_test = VideoReader('test.mp4.avi');
n_frames = vid_test.NumberOfFrames;
vid_h = vid_test.Height;
vid_w = vid_test.Width;

y = [];
mov(1:n_frames) = struct('cdata', zeros(vid_h, vid_w, 3, 'uint8'), 'colormap', []);
for k = 1 : n_frames
    mov(k).cdata = read(vid_test, k);
    y(k) = mov(k).cdata(1,1,1); % you can try any other pixel, ex (5,5,1)
end
plot(y);
pwelch(y);

% -----------------------------------------------------------------------------
