import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wave
import librosa
import math
from pydub import AudioSegment


# Convert MP3 to Wav
# import required modules
import os
from os import path
from pydub import AudioSegment
import shutil
data_origin = './3 to 3/training'

folders = os.listdir(data_origin)
# print("obj: ",obj)
for folder in folders:
    fileNames = os.listdir('{}/{}'.format(data_origin,folder))
    for file in fileNames:
        if file[len(file)-3:] == 'wav':
            try:
                os.makedirs('./Data/{}/'.format(folder))
            except:
                pass
            shutil.copyfile('{}/{}/{}'.format(data_origin,folder, file),
                            './Data/{}/{}'.format(folder, file))
            continue
        elif file[len(file)-3:] == 'mp3':
            sound = AudioSegment.from_file(
                ('{}/{}/{}'.format(data_origin,folder, file)), format='mp3')
            try:
                sound.export(
                    './Data/{}/{}.wav'.format(folder, file.split('.')[0]), format='wav')
            except:
                os.makedirs('./Data/{}/'.format(folder))
                sound.export(
                    './Data/{}/{}.wav'.format(folder, file.split('.')[0]), format='wav')
        elif file[len(file)-3:] == 'm4a':
            sound = AudioSegment.from_file(
                ('{}/{}/{}'.format(data_origin,folder, file)), format='m4a')
            try:
                sound.export(
                    './Data/{}/{}.wav'.format(folder, file.split('.')[0]), format='wav')
            except:
                os.makedirs('./Data/{}/'.format(folder))
                sound.export(
                    './Data/{}/{}.wav'.format(folder, file.split('.')[0]), format='wav')
    print("Folder - {} Done".format(folder))
