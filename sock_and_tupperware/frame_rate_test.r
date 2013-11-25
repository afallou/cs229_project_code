# import
time_v = c()
sensor_v = c()
for(f in list.files()){ 
	f_split = strsplit(f, split="_")[[1]]
	frame = as.numeric(f_split[2])
	sensor = as.numeric(f_split[4])
	time_v = append(time_v, frame); 
	sensor_v = append(sensor_v, sensor); 
}

# calculate offset
time_v_delta = time_v[2:length(time_v)] - time_v[1:(length(time_v)-1)]

# interpolate
s_good = sensor_v #[250:length(sensor_v)]
t_good = time_v #[250:length(time_v)]
t_new = seq(min(t_good), max(t_good), length.out = length(t_good)) # new timeseries with regular intervals
sample_fq = 1000*length(t_good)/(max(t_good) - min(t_good)) # approx sampling frequency
data_ts = ts(data=s_good, start=0, end=(max(t_good)-min(t_good))/1000, frequency=sample_fq) # interpolates

# old interpolation steps
# t_s_interpolated = approx( x = t_good, y = s_good, xout = t_new )  # the interpolation step
# t_s_interpolated$x = t_s_interpolated$x - min(t_s_interpolated$x)  # remove time offset so it starts a 0

# fft
s = spectrum( data_ts, log="dB", ylim=c(0, 40), xlim=c(0,10), detrend=F, demean=F, taper=0, main="Periodogram", sub=NA)
f_p = data.frame(fq=s$freq, pw=s$spec)
max_power_i = which(f_p$pw == max(f_p$pw))
max_fq = f_p$fq[max_power_i]
max_power = f_p$pw[max_power_i]

# peaks
f_p_peaks = f_p$pw - lowess(f_p$pw, f=.3)$y # substracts the moving average (lowess interpolation) for a cheap lowpass
plot(f_p_peaks, type="l")
peaks = which(scale(f_p_peaks) > .5) # magic threshold for n*std.dev > mean
spectrum( data_ts, log="dB", ylim=c(0, 40), xlim=c(0,10), detrend=F, demean=F, taper=0, main="Periodogram", sub=NA)
points(f_p$fq[peaks],10*log10(f_p$pw[peaks]), col='red')

# plot orig vs fft
par(mfrow=c(2,1))
plot(data_ts, type="l", col = 'red', main='warped + resampled + timeseries')
spectrum( data_ts, log="dB", ylim=c(0, 40), xlim=c(0,10), detrend=F, demean=F, taper=0, main="Periodogram", sub=NA)

# fit a series of sine waves
hz_wave = function(oscillation_hz, duration, sample_hz=2) return (sin(seq(0,duration,by=1/sample_hz)*2*pi*oscillation_hz))
#sin_fit = hz_wave(max_fq, (max(t_good)-min(t_good))/1000, sample_fq)*4 + mean(data_ts)
sin_fit = 0
for(i in 1:length(peaks)){ sin_fit = sin_fit + hz_wave(f_p$fq[peaks[i]], (max(t_good)-min(t_good))/1000, sample_fq) }

# plot processing steps
par(mfrow=c(5,1), mar=c(2,4,2,2))
plot(sensor_v, type='l', main='raw')
plot(time_v, sensor_v, type='l', col = 'blue', main='warped')
plot(data_ts, type="l", col = 'red', main='warped + resampled timeseries')
spectrum( data_ts, log="dB", ylim=c(0, 40), xlim=c(0,10), detrend=F, demean=F, taper=0, main="Periodogram", sub=NA)
points(f_p$fq[peaks],10*log10(f_p$pw[peaks]), col='red')
plot(data_ts, type="l", col = 'black', main='predicted waveform')
lines(seq(0, attr(data_ts,'tsp')[2	], length.out = length(data_ts)), sin_fit*4 + mean(data_ts), col = 'red') # plot fitted sine wave


