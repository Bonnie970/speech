"""
Speech Project
Alex Yin
Bonnie Hu
"""
import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd

# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile

from sklearn.decomposition import PCA
import cv2
from cv2 import resize

# Visualization
import matplotlib.pyplot as plt

def display_analysis(filename,samples,spectrogram,freqs,times):
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(211)
	ax1.set_title('Raw wave of ' + filename)
	ax1.set_ylabel('Amplitude')
	ax1.plot(samples)

	ax2 = fig.add_subplot(212)
	ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
	           extent=[times.min(), times.max(), freqs.min(), freqs.max()],cmap='coolwarm')
	ax2.set_yticks(freqs[::16])
	ax2.set_xticks(times[::16])
	ax2.set_title('Spectrogram of ' + filename)
	ax2.set_ylabel('Freqs in Hz')
	ax2.set_xlabel('Seconds')

	plt.show()

def log_specgram(audio, sample_rate, window_size=20,
				 step_size=10, eps=1e-10):
	nperseg = int(round(window_size * sample_rate / 1e3))
	noverlap = int(round(step_size * sample_rate / 1e3))
	freqs, times, spec = signal.spectrogram(audio,
									fs=sample_rate,
									window='hann',
									nperseg=nperseg,
									noverlap=noverlap,
									detrend=False)
	return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def main():
	datadir = './dataset/train/audio/'
	labels = {'yes':0,'no':1,'up':2,'down':3,'left':4,'right':5,'on':6,'off':7,'go':8,'stop':9}
	train_in = []
	train_out = []
	for word, label in labels.items():
		print(word)
		dirpath = datadir+word
		files = os.listdir(dirpath)	
		print(len(files))
		for fn in files:
			filepath = os.path.join(dirpath,fn)
			sample_rate, samples = wavfile.read(filepath)
			freqs, times, spectrogram = log_specgram(samples, sample_rate, window_size=20, step_size=10)
			# average spectrogram shape is (97.3,161)   vvvvv: x and y are reversed in cv2 
			spectrogram = cv2.resize(spectrogram,dsize=(161,100),interpolation=cv2.INTER_CUBIC)
			train_in.append(spectrogram)
			train_out.append(label)
	train_in = np.array(train_in, dtype=np.float32)
	train_out = np.array(train_out, dtype=np.int32)
	print(train_in.shape)
	print(train_out.shape)
	save_filename = './mini_speech_data.npz'
	np.savez(save_filename,
			 train_in = train_in,
			 train_out = train_out,
			 labels = labels)

if __name__ == '__main__':
	main()