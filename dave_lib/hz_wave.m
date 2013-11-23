function [x] = hz_wave(sample_hz, oscillation_hz, duration)
  % t = 0:desired_output_size; %0:1/desired_output_size : 5-1/desired_output_size;
  %   x = cos(2*pi*desired_fq*t); % + randn(size(t)); % add a little random noise
  % 
  t = 0:(1/sample_hz):(duration/sample_hz);
  x = cos(t*2*pi*oscillation_hz);
  % x(x>1) = 1;
  % x(x<-1) = -1;
  % mx = max_val/2 + x*(max_val/2);
  % x = round(mx);