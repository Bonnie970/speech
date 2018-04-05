import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

filepath='00b01445_nohash_1.wav' #'0b09edd3_nohash_0.wav'
sample_rate, samples = wavfile.read(filepath)
plt.plot(samples)
plt.show()

new=np.array([x for x in samples if ((x<-50) or (x>50))])
print(len(new), len(samples))
wavfile.write('trim2.wav',sample_rate, new)
freqs, times, spec = signal.spectrogram(new, sample_rate, nperseg=100, detrend=False, window='hann')
plt.pcolormesh(times, freqs, spec, cmap='coolwarm')
plt.show()

