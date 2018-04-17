"""
Speech Project
Alex Yin
Bonnie Hu
"""
import os
from os.path import isdir, join

# Math
import numpy as np
from scipy import signal
from scipy.io import wavfile

import cv2
from cv2 import resize

# Visualization
import matplotlib.pyplot as plt

# voice activity detector
from vad import VoiceActivityDetector


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
	freqs, times, spec = signal.spectrogram(audio,fs=sample_rate, window='hann', nperseg=nperseg,
											noverlap=noverlap, detrend=False)
	return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def one_hot_encode(label,num_class=10):
	l = np.zeros(num_class)
	l[label] = 1
	return l

def main():
	datadir = './dataset/train/audio/'
	labels = {'yes':0,'no':1,'up':2,'down':3,'left':4,'right':5,'on':6,'off':7,'go':8,'stop':9}
	labels_list = np.array(['yes','no','up','down','left','right','on','off','go','stop'])
	train_in = []
	train_out = []
	test_in = []
	test_out = []
	removal_thr = 0
	
	vadcount = 0
	for word, label in labels.items():
		print(word)
		dirpath = datadir+word
		files = os.listdir(dirpath)
		np.random.shuffle(files)
		
		for i,fn in enumerate(files):	
			
			filepath = os.path.join(dirpath,fn)
			sample_rate, samples = wavfile.read(filepath)
			#print(filepath)
			# run voice activity detector 
			v = VoiceActivityDetector(filepath)
			detected_windows = v.detect_speech()
			speechtime = v.convert_windows_to_readible_labels(detected_windows)
			
			silence_free = np.array([])
			if len(speechtime)!=0: 
				for segment in speechtime: 
					n0 = int(segment['speech_begin']*sample_rate)
					n1 = int(segment['speech_end']*sample_rate)
					silence_free = np.append(silence_free, samples[n0: n1])
				#print(len(silence_free))
			# !!! IMPORTANT check the quality of silence removel
			# bad ones will crush the spectrogram step 
			if len(silence_free) >= 3200: 
				vadcount += 1
				freqs, times, spectrogram = log_specgram(silence_free, sample_rate, window_size=10, step_size=5)
			else: 
				freqs, times, spectrogram = log_specgram(samples, sample_rate, window_size=10, step_size=5)

			#plt.pcolormesh(freqs, times, spectrogram, cmap='coolwarm')
			#plt.show()
			# average spectrogram shape is (97.3,161)   vvvvv: x and y are reversed in cv2 
			spectrogram = cv2.resize(spectrogram,dsize=(161,100),interpolation=cv2.INTER_CUBIC)
			spectrogram = spectrogram.reshape(100,161,1)

			if i<len(files)/5:
				test_in.append(spectrogram)
				test_out.append(one_hot_encode(label))
			else:
				train_in.append(spectrogram)
				train_out.append(one_hot_encode(label))
	train_in = np.array(train_in, dtype=np.float32)
	train_out = np.array(train_out, dtype=np.int32)
	test_in = np.array(test_in, dtype=np.float32)
	test_out = np.array(test_out, dtype=np.int32)
	print(train_in.shape)
	print(train_out.shape)
	print(test_in.shape)
	print(test_out.shape)
	print(vadcount, ' audios were processed with VAD.')

	save_filename = './mini_speech_data2.npz'
	np.savez(save_filename,
			 train_in = train_in,
			 train_out = train_out,
			 test_in = test_in,
			 test_out = test_out,
			 labels = labels_list)

if __name__ == '__main__':
	main()