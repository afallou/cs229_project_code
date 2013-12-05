require('pracma')
require('fftw')

# params
sample_fq = 1000                                              # sampling frequency
sample_interval = 1/sample_fq                                 # sample time
sample_duration = 1000                                        # sample duration (length of signal)

# data
sample_time_vector = (0:sample_duration-1) * sample_interval  # sample time vector
hz_50 = 0.7*sin(2*pi*50*sample_time_vector)                   # 50 Hz sinusoid
hz_120 = sin(2*pi*120*sample_time_vector)                     # 120 Hz sinusoid
gaussian_noise = 2*randn(size(sample_time_vector))            # noise
sample_data = hz_50 + hz_120                                  # sample data
#sample_data = sample_data + gaussian_noise                   # sinusoids plus noise

# fft
next_pow_2 = 2^nextpow2(sample_duration)                      # next power of 2 from length of y
data_fft = fft(sample_data, next_pow_2)/sample_duration       # fft with fq and phase
frequency_labels = sample_fq/2*linspace(0,1,next_pow_2/2+1)   # frequency labels (ex. for plot)
frequency_magnitude = 2*abs(data_fft[1:(next_pow_2/2+1)])     # frequency magnitude

# plot signal
plot(sample_fq*sample_time_vector[1:50],sample_data[1:50], xlab='time (milliseconds)', type='l')
title('Signal Corrupted with Zero-Mean Random Noise')

# Plot single-sided amplitude spectrum.
plot(frequency_labels, frequency_magnitude, type='l', xlab='Frequency (Hz)', ylab='|Y(f)|') 
title('Single-Sided Amplitude Spectrum of y(t)')

# peaks
peaks = which(scale(frequency_magnitude) > 2) # magic threshold for n*std.dev > mean
points(frequency_labels[peaks], frequency_magnitude[peaks], col='red')


