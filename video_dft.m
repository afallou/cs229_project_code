% Import video
vid = VideoReader('ResultsSIGGRAPH20121\face-ideal-from-0.83333-to-1-alpha-50-level-4-chromAtn-1.avi');

% Creates 4D matrix of (height,width,RGB,frame number);
video_matrix = read(vid);

fs = 30; %Sample freq

% Choose just the central pixel in the red (1) channel.
center_pixel = video_matrix(240,300, 1, :); 
% Had to change this vector into a one-dim vector
center_pixel = center_pixel(:);

% DFT using properties from the MathWorks website
m = length(center_pixel);
n = pow2(nextpow2(m));

f = (0:n-1)*(fs/n);
y = fft(center_pixel,n);

power = y .* conj(y)/n;
plot(f, power)
hold on
xlabel('Frequency (Hz)')
ylabel('Power')
title('{\bf Periodogram}')
%plot(abs(y))
