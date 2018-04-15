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
    """log scale spectrogram"""
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,fs=sample_rate,window='hann',nperseg=nperseg,noverlap=noverlap, detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def one_hot_encode(label,num_class=10):
    """convert numerical label to one-hot encoded labels"""
    l = np.zeros(num_class)
    l[label] = 1
    return l

def spectrogram_normalization(spectrogram):
    """normalize values in spectrogram between 0-1"""
    _min = spectrogram.min()
    _max = spectrogram.max()
    return (spectrogram-_min)/(_max-_min)

def main():
    datadir = './dataset/train/audio/'
    labels = {'yes':0,'no':1,'up':2,'down':3,'left':4,\
            'right':5,'on':6,'off':7,'go':8,'stop':9}
            #  'one':10,'two':11,'three':12,'four':13,'five':14}
    labels_list = np.array(['yes','no','up','down','left',\
            'right','on','off','go','stop'])
            #  'one','two','three','four','five'])

    train_in = []
    train_out = []
    test_in = []
    test_out = []
    for word, label in labels.items():
        print(word)
        dirpath = datadir+word
        files = os.listdir(dirpath)
        # randomly shuffle the order of files
        np.random.shuffle(files)
        print(len(files))
        for i,fn in enumerate(files):
            filepath = os.path.join(dirpath,fn)
            sample_rate, samples = wavfile.read(filepath)
            freqs, times, spectrogram = log_specgram(samples, sample_rate, window_size=20, step_size=10)
            # average spectrogram shape is (97.3,161)   vvvvv: x and y are reversed in cv2
            spectrogram = cv2.resize(spectrogram,dsize=(161,100),interpolation=cv2.INTER_CUBIC)
            # reshape to channel last for CNN training
            spectrogram = spectrogram.reshape(100,161,1)
            # normalize values between 0-1 to adjust for different volume etc.
            spectrogram = spectrogram_normalization(spectrogram)
            #  display_analysis(filename=fn,samples=samples,spectrogram=spectrogram,freqs=freqs,times=times)

            if i<len(files)/5:
                # 1/5 for testing
                test_in.append(spectrogram)
                test_out.append(one_hot_encode(label,num_class=len(labels_list)))
            else:
                # 4/5 for training
                train_in.append(spectrogram)
                train_out.append(one_hot_encode(label,num_class=len(labels_list)))

    train_in = np.stack(train_in, axis=0).astype(np.float32)
    train_out = np.stack(train_out, axis=0).astype(np.int32)
    test_in = np.stack(test_in, axis=0).astype(np.float32)
    test_out = np.stack(test_out, axis=0).astype(np.int32)
    print(train_in.shape)
    print(train_out.shape)
    print(test_in.shape)
    print(test_out.shape)

    save_filename = './mini_speech_data.npz'
    np.savez(save_filename,
                 train_in = train_in,
                 train_out = train_out,
                 test_in = test_in,
                 test_out = test_out,
                 labels = labels_list)

if __name__ == '__main__':
    main()
