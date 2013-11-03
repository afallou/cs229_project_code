%------------------------------------------------------------
% preamble

cd('/Users/dderiso/Math/cs229/project/cs229_project_code/EVM_Matlab')
addpath('matlabPyrTools')

%------------------------------------------------------------
% input/output

input_video = '8_sec_no_movement_mp4.mp4';
% input_video = '8_sec_no_movement_mov.mov';

data_dir = '../videos/dave';
results_dir = '../videos/output';
input_video_path = fullfile(data_dir, input_video);

%------------------------------------------------------------
% magical params

pulse_estimate = 23*3;  % count the number of pules you have for 20 sec then * 3
window_size = 3;        % expands frequency range by 2x: lower_lim - x, upper_lim + x Hz
                        % 2: nothing happens, 2.5: nothing happens, 2.8: does nothing, 3: works great, 4: very pronounced
                        
alpha = 50;
level = 4;
fl = (pulse_estimate-window_size)/60; % low freq
fh = (pulse_estimate+window_size)/60; % high freq
samplingRate = 30;
chromAttenuation = 1;

%------------------------------------------------------------
% main function

amplify_spatial_Gdown_temporal_ideal(input_video_path, results_dir, alpha, level, fl, fh, samplingRate, chromAttenuation);

%------------------------------------------------------------
% experiments


%------------------------------------------------------------
% breaking this apart

% 1 construct ugly output name
[~,vidName] = fileparts(vidFile);
outName = fullfile(outDir,[vidName '-ideal-from-' num2str(fl) ...
                       '-to-' num2str(fh) ...
                       '-alpha-' num2str(alpha) ...
                       '-level-' num2str(level) ...
                       '-chromAtn-' num2str(chromAttenuation) '.avi']);

% 2 get video info
vid = VideoReader(vidFile);
vidHeight = vid.Height;
vidWidth = vid.Width;
nChannels = 3;
fr = vid.FrameRate;
len = vid.NumberOfFrames;

% 3 make an empty output
vidOut = VideoWriter(outName);
vidOut.FrameRate = fr;
open(vidOut);
temp = struct('cdata', zeros(vidHeight, vidWidth, nChannels, 'uint8'), 'colormap', []); 

% 4 build some mysterious parameters
startIndex = 1;
endIndex = len-10;

% 5 compute Gaussian blur stack (aka 'spatial filtering')
% Gdown_stack = build_GDown_stack(vidFile, startIndex, endIndex, level);

% this stuff is redundant ----
% Read video
% vid = VideoReader(vidFile);
% Extract video info
% vidHeight = vid.Height;
% vidWidth = vid.Width;
% nChannels = 3;
% temp = struct('cdata', zeros(vidHeight, vidWidth, nChannels, 'uint8'), 'colormap', []);
% ----

% 6 extract first frame of input
temp.cdata = read(vid, startIndex);

% 7 ???
[rgbframe, ~] = frame2im(temp);

% 8 ??? convert from RGB to NTSC?
rgbframe = im2double(rgbframe);
frame = rgb2ntsc(rgbframe);

% 9 compute gaussian blur with respect to color
%blurred = blurDnClr(frame,level);

% decrypt shitty naming conventions
im = frame; % frame of video, really big matrix
nlevs = level; % set as 4
% filt = unspecified optional parameter
filt = 'binom5'; % later set to this by default, 
% corresoponds to 
% if strcmp(name(1:min(5,size(name,2))), 'binom')
  % kernel = sqrt(2) * binomialFilter(str2num(name(6:size(name,2))));


if (exist('nlevs') ~= 1) 
  nlevs = 1;
end

% stopped here ...

tmp = blurDn(im(:,:,1), nlevs, filt);
out = zeros(size(tmp,1), size(tmp,2), size(im,3));
out(:,:,1) = tmp;
for clr = 2:size(im,3)
  out(:,:,clr) = blurDn(im(:,:,clr), nlevs, filt);
end


% create pyr stack
GDown_stack = zeros(endIndex - startIndex +1, size(blurred,1),size(blurred,2),size(blurred,3));
GDown_stack(1,:,:,:) = blurred;

k = 1;
for i=startIndex+1:endIndex
        k = k+1;
        temp.cdata = read(vid, i);
        [rgbframe,~] = frame2im(temp);

        rgbframe = im2double(rgbframe);
        frame = rgb2ntsc(rgbframe);

        blurred = blurDnClr(frame,level);
        GDown_stack(k,:,:,:) = blurred;

end


%------------------------------------------------------------