import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

filepath='original1.wav' #'0b09edd3_nohash_0.wav'
sample_rate, samples = wavfile.read(filepath)
#plt.plot(samples)
#plt.show()

new=np.array([x for x in samples if ((x<-50) or (x>50))])
print(len(new), len(samples))
wavfile.write('trim2.wav',sample_rate, new)
#freqs, times, spec = signal.spectrogram(new, sample_rate, nperseg=100, detrend=False, window='hann')


def log_specgram(audio, sample_rate, window_size=20,
				 step_size=10, eps=1e-10):
	nperseg = int(round(window_size * sample_rate / 1e3))
	noverlap = int(round(step_size * sample_rate / 1e3))
	freqs, times, spec = signal.spectrogram(audio,fs=sample_rate, window='hann', nperseg=nperseg,
											noverlap=noverlap, detrend=False)
	#plt.pcolormesh(times, freqs, spec, cmap='coolwarm')
	#plt.show()
	return freqs, times, np.log(spec.astype(np.float32) + eps)

freqs, times, spec = log_specgram(samples, sample_rate, window_size=30, step_size=10)
plt.pcolormesh(times, freqs, spec, cmap='coolwarm')
plt.show()



