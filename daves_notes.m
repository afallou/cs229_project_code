
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


