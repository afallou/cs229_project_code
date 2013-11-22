addpath('dave_lib')
x = create_waveform_of_known_fq(10, 500);
x = x + abs(floor(min(x))); % no negative numbers

vidOut = VideoWriter('test.mp4');
vidOut.FrameRate = 30;
open(vidOut);

a = ones(400, 400, 3, 'uint8');
for i=1:500
  frame = a;
  frame(:,:,1) = x(i);
  frame(:,:,2) = 0;
  frame(:,:,3) = 0;
  writeVideo(vidOut,im2uint8(ntsc2rgb(frame)));
end

close(vidOut);
