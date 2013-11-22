function [x] = create_waveform_of_known_fq(desired_fq, desired_output_size)
  t = 0:1/desired_output_size : 5-1/desired_output_size;
  x = cos(2*pi*desired_fq*t) + randn(size(t)); % add a little random noise