import os
import sys
import argparse
import json
from vad import VoiceActivityDetector

def save_to_file(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)

if __name__ == "__main__":
    #  parser = argparse.ArgumentParser(description='Analyze input wave-file and save detected speech interval to json file.')
    #  parser.add_argument('inputfile', metavar='INPUTWAVE',
    #                      help='the full path to input wave file')
    #  parser.add_argument('outputfile', metavar='OUTPUTFILE',
    #                      help='the full path to output json file to save detected speech intervals')
    #  args = parser.parse_args()
    dirpath = '/home/zixuan/Desktop/speech/dataset/train/audio/yes'
    file_list = os.listdir(dirpath)
    wav_files = []
    for fn in file_list:
        if fn[-4:] == '.wav':
            wav_files.append(fn)
    #  print(wav_files)
    for fn in wav_files:
        wav_path = os.path.join(dirpath,fn)
        v = VoiceActivityDetector(wav_path)
        v.plot_detected_speech_regions()

    sys.exit()
    raw_detection = v.detect_speech()
    speech_labels = v.convert_windows_to_readible_labels(raw_detection)

    save_to_file(speech_labels, args.outputfile)

