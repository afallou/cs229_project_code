
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

