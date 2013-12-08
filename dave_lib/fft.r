# pipeline

require('pracma')
require('fftw')

resample = function(data_in, sampling_fq=-1, time_col='time', signal_col='signal'){
	# purpose: interpolate data into a new timeseries with regular intervals
	# input: data_in = data frame with time and signal columns
	#        time_col (optional) = name of the time column
	#        signal_col (optional) = name of the signal column
	#        fq (optional) = desired sampling rate
	# output: interpolated data frame with time and signal columns
	
	# input
	data_in = sample_data
	time_v = data_in[,time_col]
	data_v = data_in[,signal_col]
	
	# time sequence (if desired frequency is specified or not)
	if(sampling_fq==-1){
		time_new = seq(min(time_v), max(time_v), length.out = length(time_v))
	} else {
		time_new = seq(min(time_v), max(time_v), by = sampling_fq)
	}
	
	# remove time offset so it starts a 0
	# time_new = time_new - min(time_new)
	
	# interpolate
	data_interpolated = approx( x = time_v, y = data_v, xout = time_new )
	out = data.frame(time=data_interpolated$x, signal=data_interpolated$y)
	
	# estimate duration of input and output signal via median interval
	t_l = length(time_v)
	t_l_n = length(time_new)
	sample_fq_raw = (max(time_v) - min(time_v))/ median(time_v[2:t_l] - time_v[1:(t_l-1)])
	sample_fq_resampled = (max(time_new) - min(time_new))/ median(time_new[2:t_l_n] - time_new[1:(t_l_n-1)])
	attr(out, 'sampling_fq_raw') = sample_fq_raw
	attr(out, 'sampling_fq') = sample_fq_resampled
	
	return(out)
}

simple_fft = function(signal_in, sampling_fq=-1){
	# purpose: perform a one-sided fft and return phase and fq
	# input: signal_in = signal vector
	#        sampling_fq = sampling rate of signal
	#        if signal_in is a data frame with signal and time and sampling_fq attributes, sampling_fq is not used
	# output: frequency with corresponding power and phase
	
	if(sampling_fq == -1){
		sampling_fq = attr(signal_in, 'sampling_fq')
		signal_in = signal_in$signal
	}
	
	# params
	#sample_fq = 1000                                             # sampling frequency
	sample_interval = 1/sampling_fq                               # sample time
	sample_duration = length(signal_in)                           # sample duration (length of signal)
	
	#fft
	next_pow_2 = 2^nextpow2(sample_duration)                      # next power of 2 from length of y
	data_fft = fft(signal_in, next_pow_2)/sample_duration         # fft with fq and phase
	
	# frequency
	frequency_labels = sampling_fq/2*linspace(0,1,next_pow_2/2+1)   # frequency labels (ex. for plot)
	frequency_magnitude = 2*abs(data_fft[1:(next_pow_2/2+1)])     # frequency magnitude
	
	# phase
	phase = (angle(data_fft)/pi)[1:(next_pow_2/2+1)]              # phase in units of pi for each fq
	
	#out
	fq_pw = data.frame(fq=frequency_labels, power=frequency_magnitude, phase=phase)
	attr(fq_pw, 'sampling_fq') = sampling_fq                          # store fq as an attr
	
	return(fq_pw)
}

fft_peaks = function(fft_in, threshold = 1, distance = 10){
	
	# fft_in = dfft
	# threshold = 1
	# distance = 2
	peaks = which(scale(fft_in$power) > threshold) # magic threshold for n*std.dev > mean
	fq_pw = data.frame(fq=fft_in$fq[peaks], power=fft_in$power[peaks], phase=fft_in$phase[peaks])
	
	# remove peaks near each other (specified by distance)
	fq_pw = fq_pw[order(fq_pw$power, decreasing=T),] # sort by power
	
	# if there's more than 1 row in the peaks
	if(nrow(fq_pw) > 1){
		# start with nothing
		peaks_to_remove = c()
		
		# for each row (1 less than the total since the inner loop will reach total)
		for(i in 1:(nrow(fq_pw)-1)){
			
			# for the next row to total
			# since the power is sorted, the next value will be smaller and thus less important of a peak
			for(k in (i+1):nrow(fq_pw)){
				# compute abs distance
				dist_i_k = abs(fq_pw[i,'fq'] - fq_pw[k,'fq'])
				
				# if the distance is smaller than the set distance threshold
				if(dist_i_k < distance){
					#add the row index to the array
					peaks_to_remove = append(peaks_to_remove, k)
				}
			}
		}
		
		# if theres rows to remove
		if(length(peaks_to_remove) > 0){
			# remove rows
			fq_pw = fq_pw[-c(unique(peaks_to_remove)),]
		}
	}
	
	return(fq_pw)
	
	#fq_pw = list() 
	#power_peaks = which(scale(fft_in$power) > threshold) # magic threshold for n*std.dev > mean
	#phase_peaks = which(scale(fft_in$phase) > threshold)
	#fq_pw[[1]] = data.frame(fq=fft_in$fq[power_peaks], power=fft_in$power[power_peaks])
	#fq_pw[[2]] = data.frame(fq=fft_in$fq[phase_peaks], phase=fft_in$phase[phase_peaks])
	#fq_pw[[2]] = data.frame(fq=fft_in$fq[power_peaks], phase=fft_in$phase[power_peaks])
}

plot_resampled_fft_peaks = function(data_in){
	resampled_data = resample(data_in)
	dfft = simple_fft(resampled_data)
	dfft_peaks = fft_peaks(dfft)
	
	print("FFT")
	print(dfft)
	print("PEAKS")
	print(dfft_peaks)
	
	# plot
	par(mfrow=c(4,1), mar=c(2,4,2,2))
	
	# data
	plot(data_in$time, data_in$signal, xlab='time (sec)', ylab='Signal', type='l')
	title('Raw Signal')
	
	# resampled data
	plot(resampled_data$time, resampled_data$signal, xlab='time (sec)', ylab='Signal', type='l')
	title('Resampled Signal')
	
	# fft
	plot(dfft$fq, dfft$power, type='l', xlab='Frequency (Hz)', ylab='Power (dB)') 
	title('Single-Sided Power Spectrum with Peaks')
	points(dfft_peaks$fq, dfft_peaks$power, col='red')
	
	# phase
	plot(dfft$fq, dfft$phase, type='l', xlab='Frequency (Hz)', ylab='Phase (pi)') 
	title('Single-Sided Phase for Each Frequency with Peaks')
	points(dfft_peaks$fq, dfft_peaks$phase, col='red')
}


# example
sample_fq = 1000                                              # sampling frequency
sample_interval = 1/sample_fq                                 # sample time
sample_duration = 1000                                        # sample duration (length of signal)

sample_time_vector = (0:sample_duration-1) * sample_interval  # sample time vector
hz_50 = 0.7*sin(2*pi*50*sample_time_vector)                   # 50 Hz sinusoid
hz_120 = sin(2*pi*120*sample_time_vector)                     # 120 Hz sinusoid
gaussian_noise = 2*randn(size(sample_time_vector))            # noise
sample_data = hz_50 + hz_120
sample_data = data.frame(time=sample_time_vector, signal=sample_data)

plot_resampled_fft_peaks(sample_data)



# resampled_data = resample(sample_data)
# dfft = simple_fft(resampled_data)
# dfft_peaks = fft_peaks(dfft)