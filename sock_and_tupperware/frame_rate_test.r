# ---------------------------------------------------------------------------------------
# global

require('pracma')
require('fftw')
#requre('signal')

#setwd('/Users/dderiso/Math/cs229/project/cs229_project_code/sock_and_tupperware/nullimages')
#setwd('C:/Users/Neeloy/Documents/GitHub/cs229_project_code/sock_and_tupperware/nullimages')

# ---------------------------------------------------------------------------------------
# functions

# import/generate data

generate_sample_data = function(frequency_vector, weight_vector, phase_vector, sample_fq=1000, sample_duration=1000, yoffset=0){
	# example
	#sample_fq = 1000                                              # sampling frequency
	sample_interval = 1/sample_fq                                 # sample time
	#sample_duration = 1000                                        # sample duration (length of signal)

	sample_time_vector = (0:sample_duration) * sample_interval  # sample time vector
	# hz_50 = 0.7*sin(2*pi*50*sample_time_vector)                   # 50 Hz sinusoid
	# hz_120 = sin(2*pi*120*sample_time_vector)                     # 120 Hz sinusoid
	
	sample_phase_vector = (0:sample_duration)*0 + 1         # Create vector of ones
	
	sample_data = sample_time_vector*0 + yoffset            # create baseline for adding phase
	for(i in 1:length(frequency_vector)){
		w = weight_vector[i]
		fq = frequency_vector[i]
		ph = phase_vector[i]
		#sample_data = sample_data + w*sin(2*pi*fq*sample_time_vector)
		sample_data = sample_data + w*sin(2*pi*fq*sample_time_vector + ph*sample_phase_vector)   # Have included phase
	}
	
	#gaussian_noise = 2*randn(size(sample_time_vector))            # noise

	#sample_data = hz_50 + hz_120
	sample_data = data.frame(time=sample_time_vector*1000, signal=sample_data)
	return(sample_data)
}

import_pulseox_data = function(){
	time_v = c()
	sensor_v = c()
	for(f in list.files(pattern=".bmp")){ 
		f_split = strsplit(f, split="_")[[1]]
		frame = as.numeric(f_split[2])
		sensor = as.numeric(f_split[4])
		time_v = append(time_v, frame); 
		sensor_v = append(sensor_v, sensor); 
	}
	pulseox_data = data.frame(time=time_v, signal=sensor_v)
	return(pulseox_data)
}

# preprocess

resample = function(data_in, sampling_fq=-1, time_col='time', signal_col='signal'){
	# purpose: interpolate data into a new timeseries with regular intervals
	# input: data_in = data frame with time and signal columns
	#        time_col (optional) = name of the time column, expects millisecond integers!!
	#        signal_col (optional) = name of the signal column
	#        fq (optional) = desired sampling rate
	# output: interpolated data frame with time and signal columns
	
	# input
	# data_in = pulseox_data
# 	sampling_fq=-1
# 	time_col='time'
# 	signal_col='signal'
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
	sample_fq_raw = 1000/median(time_v[2:t_l] - time_v[1:(t_l-1)])
	sample_fq_resampled = 1000/median(time_new[2:t_l_n] - time_new[1:(t_l_n-1)])
	# sample_fq_raw = (max(time_v) - min(time_v))/ median(time_v[2:t_l] - time_v[1:(t_l-1)])
	# sample_fq_resampled = (max(time_new) - min(time_new))/ median(time_new[2:t_l_n] - time_new[1:(t_l_n-1)])
	
	attr(out, 'sampling_fq_raw') = sample_fq_raw
	attr(out, 'sampling_fq') = sample_fq_resampled
	
	return(out)
}

# fft

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
	frequency_magnitude[1] = 0 # hack for now
	
	# phase
	phase = (angle(data_fft)/pi)[1:(next_pow_2/2+1)]              # phase in units of pi for each fq
	
	#out
	fq_pw = data.frame(fq=frequency_labels, power=frequency_magnitude, phase=phase)
	attr(fq_pw, 'sampling_fq') = sampling_fq                          # store fq as an attr
	
	return(fq_pw)
}

# peaks

fft_peaks = function(fft_in, threshold = 1, distance = 10, filter_type='none'){
	
	# fft_in = dfft
	# threshold = 1
	# distance = 2
	
	if(filter_type=='none'){ fft_in$hp = fft_in$power }
	if(filter_type=='ma'){ fft_in$hp = fft_in$power - filter(fft_in$power,c(rep(1, 3))/3)  } # moving average of 3 samples
	if(filter_type=='lowess'){ fft_in$hp = fft_in$power - lowess(fft_in$power, f=.3)$y }  #lowess(fft_in$power, f=.3)$y 
	#if(filter_type=='bw'){ fft_in$hp = filter( butter(4, 0.1, 'high'),fft_in$power) } # uses butterworth filter from signal package for hp > ".1"fq (not really hz)
	
	peaks = which(scale(fft_in$hp) > threshold) # magic threshold for n*std.dev > mean
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

# plots

plot_resampled_fft_peaks = function(data_in, threshold = 1, distance = 10, filter_type='none'){
	resampled_data = resample(data_in)
	dfft = simple_fft(resampled_data)
	dfft_peaks = fft_peaks(dfft, threshold, distance, filter_type)
	
	# try to reconstruct the data from the input
	reconstructed_data = generate_sample_data(
		dfft_peaks$fq,
		dfft_peaks$power,
		dfft_peaks$phase,
		attr(resampled_data, 'sampling_fq'),
		nrow(resampled_data)-1,
		median(resampled_data$signal)
	)
	
	print("FFT")
	print(dfft)
	print("PEAKS")
	print(dfft_peaks)
	
	# plot
	par(mfrow=c(5,1), mar=c(2,4,2,2))
	
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
	
	# reconstructed data
	plot(resampled_data$time, resampled_data$signal, xlab='time (sec)', ylab='Signal', type='l', col='lightgrey')
	title('Reconstructed Signal')
	lines(resampled_data$time, reconstructed_data$signal)
}

load_plot_libs = function(){
	require(ggplot2)
	require(reshape2)
	require(plyr)
	require(grid)

	load_multicore_libs()
}

# svm

rmse = function(error_in){
	return(sqrt(mean(error_in^2, na.rm=T))) 
}

# other

load_multicore_libs = function() {
	require(foreach)
	require(doMC)
	require(plyr)
	registerDoMC(cores=8) #registers half the number of cores in the system, unless otherwise specified
	cores = getDoParWorkers() #prints out the number of cores sanity check
	print(paste("You're now rocking with ", cores, " cores", sep=""))
	
	# system.time(foreach(i = 1:10000,.combine = "cbind") %do% { sum(rnorm(10000)) }) 		# without parallel
	# system.time(foreach(i = 1:10000,.combine = "cbind") %dopar% { sum(rnorm(10000)) })	# with parallel
}

ps = function(...) paste(..., sep="")

p = function(...) cat(ps(..., '\n'))

save_data = function(data_in, f_name_in){
	f_name = ps(f_name_in, ".csv")
	write.table(data_in, f_name, col.names = T, row.names=F, quote=F, sep=",")
	system(ps("gzip ", f_name))
	p('saving', " ", f_name_in, "...")
}	

# feature extraction

# compute the frequency bins
bin_fft = function(fft_in, bins_per_fq, min_fq, max_fq){
	# create bins
	# bins_per_fq = 4
	# min_fq = 1
	# max_fq = 6
	bins = seq(min_fq,max_fq,by=1/bins_per_fq)

	# combine bins with fq, and split fq vector
	fq_features = data.frame(fft_in, bin=1) # start with bin 1
	for(i in 0:(length(bins)-1)){
		next_fq = which(fft_in$fq > bins[i+1])
		fq_features$bin[next_fq] = i+2
	}
	
	return(fq_features)
}

# get mean of each fq
compute_features = function(data_in){
	dv = data_in[,'power']
	output = data.frame(
		fq=mean(data_in[,'fq']), 
		phase=mean(data_in[,'phase']),
		power=mean(dv)#,
		# sd=sd(dv), 
		# n=length(dv), 
		# se=sd(dv)/sqrt(length(dv)) 
	)
	return(output)
}

# perform on timecourse for pixel i,k includes phase
extract_features = function(i,k){
	pixel_data = data.frame(time=pulseox_data$time, signal=im3d[i,k,])
	resampled_data = resample(pixel_data)
	dfft = simple_fft(resampled_data)
	dfft$bin = fq_features$bin
	# extract and reshape
	features_col = ddply(dfft, 'bin', compute_features)
	feature_row = acast(features_col, .~bin, value.var ='power')
	phase_row = acast(features_col, .~bin, value.var ='phase')
	feature_row = data.frame(i=i, k=k, feature_row, phase_row)
	return(feature_row)
}

# ---------------------------------------------------------------------------------------
# import, preprocess, plot

# test case
sample_data = generate_sample_data(c(50, 120), c(.7, 1), c(1,1))
plot_resampled_fft_peaks(sample_data, 1, 5, 'none')

pulseox_data = import_pulseox_data()
plot_resampled_fft_peaks(pulseox_data, 1, .1, 'lowess')

plot_resampled_fft_peaks(pulseox_data, .1, .05, 'lowess')
plot_resampled_fft_peaks(pulseox_data, .01, .05, 'lowess')
plot_resampled_fft_peaks(pulseox_data, .01, .05, 'none')

# ---------------------------------------------------------------------------------------
# image libraries

# install.packages('bmp', dependencies=T)
# install.packages('pixmap', dependencies=T)
require('bmp')
require('pixmap')

# ---------------------------------------------------------------------------------------
# plot 25 images in a tile

# load image file names
image_files = list.files(pattern=".bmp")

# get size of image
im1 = read.bmp(image_files[1], Verbose = FALSE)

# manually locate neeloy's face
w = c(.7, .3) * size(im1)[1]
h = c(.22, .4) * size(im1)[2]
w = w[1]:w[2]
h = h[1]:h[2]

# plot them all
par(mfrow=c(5,5), mar=c(0,0,0,0))
for(i in 1:25){
	im = read.bmp(image_files[i], Verbose = FALSE)
	image(t(as.matrix(im[w,h,1])), xaxt='n', ann=FALSE, yaxt='n')
}

# ---------------------------------------------------------------------------------------
# import image matricies

# import images as matricies, extract r channel, extract face, store in list
im_list = list()
for(i in 1:length(image_files)){
	im = read.bmp(image_files[i], Verbose = FALSE)
	im_mat = as.matrix(im[w,h,1])
	im_list[[i]] = im_mat
}
save_data(im_list, "im")

# import images as matricies, extract r channel, extract face, store in 3D matrix
im3d = array(1, dim=c(length(w), length(h), length(image_files)))
for(i in 1:length(image_files)){
	im = read.bmp(image_files[i], Verbose = FALSE)
	im3d[,,i] = as.matrix(im[w,h,1])
}
save_data(im3d, "im3d")

# ---------------------------------------------------------------------------------------
# plot libraries

load_plot_libs()

# ---------------------------------------------------------------------------------------
# explore the image data

# look at single pixel
plot(im3d[1,,], type="l")

# look at a single image in the list
image(t(im3d[,,1]))

# look at all pixels in a row
im_col = t(im3d[1,,])
nrow(im_col) == length(image_files) # check that the rows down a column correspond to time

# plot each of the pixels in a column independently
mt = melt(im_col)
colnames(mt) = c("time", "pixel", "value")
ggplot(mt, aes(time, value, group=pixel, color=pixel)) + geom_line()

# plot the mean
plot(rowMeans(im_col), type='l')

# try running a pixel through the pipeline
pixel_data = data.frame(time=pulseox_data$time, signal=im3d[10,10,])
plot_resampled_fft_peaks(pixel_data, 1, .1, 'lowess')

# compare to the pulseox data
dev.new()
plot_resampled_fft_peaks(pulseox_data, 1, .1, 'lowess')

# ---------------------------------------------------------------------------------------
# create training labels

pulseox_resampled = resample(pulseox_data)
pulseox_fft = simple_fft(pulseox_resampled)
pulseox_fft_bin = bin_fft(pulseox_fft, 4, 1, 6)
pulseox_features = ddply(pulseox_fft_bin, 'bin', compute_features)
pulseox_features_row_power = acast(pulseox_features, .~bin, value.var ='power')
colnames(pulseox_features_row_power) = paste('fq', colnames(pulseox_features_row_power), sep='.')
pulseox_features_row_phase = acast(pulseox_features, .~bin, value.var ='phase')
colnames(pulseox_features_row_phase) = paste('ph', colnames(pulseox_features_row_phase), sep='.')
pulseox_features_row = cbind(pulseox_features_row_power, pulseox_features_row_phase)

# ---------------------------------------------------------------------------------------
# create training features (in parallel)

video_fft_bin = c()
i_max = size(im3d)[1]
k_max = size(im3d)[2]
for(k in 1:k_max){
	print(k)
	feature_row = adply(1:i_max, .margins=1, extract_features, k, .parallel=T)
	video_fft_bin = rbind(video_fft_bin, feature_row)
}
colnames(video_fft_bin) = c('px.i','px.k',colnames(pulseox_features_row))

# save features
save_data(video_fft_bin, "video_fft_bin_phase")

# ---------------------------------------------------------------------------------------

# # plot features
# ggplot(features_col, aes(x=fq, y=power, fill=bin)) + 
# geom_bar(stat="identity", position="dodge") + labs(x="fq", fill ="bin", y="power") #+ 
# #geom_errorbar(aes(ymin = power-se, ymax = power+se), position=position_dodge(.9), width=0.2)

# ---------------------------------------------------------------------------------------
# create training data matrix

# note: we should probably de-mean the data, but we can add a de-mean term in the formula

# training
training_features = video_fft_bin # - c(0,0,colMeans(video_fft_bin)[3:ncol(video_fft_bin)] ) # feature vector (input)
training_labels = pulseox_features_row # ground truth (desired output)
training_data = cbind(f=training_features, l=training_labels)
#save_data(training_data, "training_data")

# ---------------------------------------------------------------------------------------
# load training data

training_data = read.csv('training_data.csv.gz', header = T, stringsAsFactors = F)

# ---------------------------------------------------------------------------------------
# create model formula

# flags
f_pix = F # use pixel location as a feature (without generalizes better)

# formula
pixel_names = c("px.i", "px.k")
fq_names = paste("fq.", 1:21, sep="")
ph_names = paste("ph.", 1:21, sep="")
fq_ph = c(fq_names, ph_names)
label_names = paste("l.", fq_ph, sep="")
feature_names = paste("f.", fq_ph, sep="") # without pixels in the model
if(f_pix) feature_names = c(paste("f.", pixel_names, sep=""), feature_names) # with pixels in the model
label_formula = paste(label_names, collapse=',')
feature_formula = paste(feature_names, collapse='+')

# multivariate regression
# the intercept term captures basically the entire prediction, so we remove it in the formula with (-1)
model_formula = as.formula(paste("cbind(", label_formula, ")", "~", feature_formula, "-1")) 

# univariate regression
# model_formula = as.formula(paste("labels.1", "~", features)) 

# ---------------------------------------------------------------------------------------
# train linear model

# train
lm_model = lm(model_formula, data=training_data)

# predict
weight_matrix = as.matrix(lm_model$coeff)                # extract weight matrix
weight_matrix[is.na(weight_matrix)] = 0                  # set NA to zero
if(f_pix) pred = as.matrix(training_features) %*% weight_matrix # matrix multiply
if(!f_pix) pred = as.matrix(training_features[,3:ncol(training_features)]) %*% weight_matrix # matrix multiply
predict_lm = colMeans(pred)                              # take an average prediction and add the intercept term
rmse_t_lm = rmse(predict_lm - training_labels)           # compute the root mean squared error
# 0.0162 without pixels, 0.0125 with pixels
# 0.00884 with pixels + phase

# ---------------------------------------------------------------------------------------
# reconstruct pulse ox waveform based on linear model fit

fq_vector = pulseox_features$fq
weight_vector = predict_lm[1:21]
phase_vector = predict_lm[22:42]
resampled_pulseox_data = resample(pulseox_data)

reconstructed_data = generate_sample_data(
	fq_vector,
	weight_vector,
	phase_vector,
	attr(resampled_pulseox_data, 'sampling_fq'),
	nrow(resampled_pulseox_data)-1,
	median(resampled_pulseox_data$signal)
)

# plot_resampled_fft_peaks(reconstructed_data, 1, .1, 'lowess')

# plot
plot(resampled_pulseox_data, xlab='time (sec)', ylab='Signal', type='l', col='lightgrey')
title('Actual vs. Predicted Signal')
lines(resampled_pulseox_data$time, reconstructed_data$signal)

# ---------------------------------------------------------------------------------------
# cross validation

# create cross validation set
cv_features = c() # feature vector (input)
cv_labels = c() # ground truth (desired output)
cv_data = data.frame(features=training_features, labels=training_labels)

# cross validate
# breaks, so we have to do it manually
# install.packages('DAAG', dependencies=T)
# require(DAAG)
# lm_cv = cv.lm(df=training_data, form.lm=model_formula, m=3)

# cross validation
# predict_cv_glm = predict.lm(glm_model, cv_features)
# rmse_cv_glm = rmse(predict_cv_glm - cv_labels)

# ---------------------------------------------------------------------------------------
# train SVM model

require('e1071') # SVM package

# this breaks:

# The columns in data passed to svm need to contain only numeral values.
# I simply assigned a number to each category of each feature. However,
# *** there must not be a column where all the numbers are equal*** (there
# mustn't be a feature with always the same value), so don't try to use
# bit-representation suitable for neural networks.
# https://stat.ethz.ch/pipermail/r-help/2007-March/127853.html

# svm model
# model_formula = as.formula(paste("labels.1", "~", features))
# svm_model = svm(model_formula, data=training_data, type="eps-regression")
# predict_svm = predict.lm(svm_model, features)
# rmse_t_svm = rmse(predict_svm - labels)

# cross validation
# predict_cv_svm = predict.lm(svm_model, cv_features)
# rmse_cv_svm = rmse(predict_cv_svm - cv_labels)




