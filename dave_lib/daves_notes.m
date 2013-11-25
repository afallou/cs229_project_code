
#videoReader
vid = your video (mp4 works)


# bandpass

samplingRate





% -----------------------------------------------------------------------------
input: GDOWN_STACK: gaussuan blur + downsample
vidFile
startIndex = 1
endIndex = len-10 = number of frames - 10
level = 4 = band of the gaussian blur

temp = width x height x color (=3)

output: GDOWN_STACK: stack of one band of Gaussian pyramid of each frame 
1 time axis
2 y axis of the video
3 x axis of the video
4 color channel

2. bandpass
input = gaussian blurred data
dim = 1
wl = fl # min freq
wh = fh # max freq
samplingRate = samplingRate

3. height, width, nChannels


% -----------------------------------------------------------------------------

% example pulseox data, sampled at 125 Hz, units are scaled to range from 0 to 1023
http://bsp.pdx.edu/Data/TBI_PLETH.zip

% extract the first 500KB to a text file
head -c 500000 TBI_PLETH.txt >> pulseox.txt

% read data
po_raw = textread('pulseox.txt');
ps = po_raw(4000:5999);
plot(ps)
[cA,cD] = dwt(ps, 'haar');

[S,F,T,P] = spectrogram(ps,256,250,256,125);
surf(T,F,10*log10(P),'edgecolor','none'); axis tight; 
view(0,90);
xlabel('Time (Seconds)'); ylabel('Hz');

% -----------------------------------------------------------------------------
% create frequency

sampling_rate = 30; % hz
x=randn(1000,1);
fx=fft(x); % frequency domain .
N=length(fx);
p=0.4/2; % desired bandwidth  percentage, /2 because fft is TWO sided !!!
P=round(p*N);
fx(P+1:end)=0;
y=ifft(fx);

(0:floor(N/2))/(N/sampling_rate)



pwelch(x), axes=axis;title(' PSD AWGN');

pwelch(y), axis(axes), title(' Coloredn');


figure, plot(x), hold on, plot(real(y),'r'), hold off


sigma = 5;
size = 30;
x = linspace(-size / 2, size / 2, size);
gaussFilter = exp(-x .^ 2 / (2 * sigma ^ 2));
gaussFilter = gaussFilter / sum (gaussFilter); % normalize

y = rand(500,1);
yfilt = filter (gaussFilter,1, y);

yfilt = conv (y, gaussFilter, 'same');

function [f] = w_fq(fq, fs)
  % fq = 100;
  % fs = 1000;
  t = 0:1/fs:5-1/fs;
  x = cos(2*pi*fq*t) + randn(size(t));
  [pxx,f] = pwelch(x,500,300,500,fs);
  plot(f,10*log10(pxx))
  xlabel('Hz'); ylabel('dB');

% -----------------------------------------------------------------------------

addpath('dave_lib')


x = create_waveform_of_known_fq(10, 500);
[pxx,f] = pwelch(x,500,300,500,fs);
plot(f,10*log10(pxx))

vidOut = VideoWriter('test.mp4');
vidOut.FrameRate = 30;
open(vidOut);
temp = struct('cdata', zeros(400, 400, 3, 'uint8'), 'colormap', []); 

temp.cdata = read(vid, startIndex);
[rgbframe,~] = frame2im(temp);
rgbframe = im2double(rgbframe);
frame = rgb2ntsc(rgbframe);






% -----------------------------------------------------------------------------

addpath('dave_lib')

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



video_matrix = read(vid_test);
video_matrices_array{i} = video_matrix


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





