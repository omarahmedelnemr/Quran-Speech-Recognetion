import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import random

train_to_test = 0.9

Extened = ''
# DataPath = './{}Data'.format(Extened)
DataPath = './{}PreprossedData'.format(Extened)
SpecrtoPath = './{}Spectro'.format(Extened)
testSpectroPath = './TestSpectro'
foldersArray = os.listdir(DataPath)

for folder in foldersArray:
    print("Working on Folder: {}".format(folder))
    # Get The Folder Contents
    sampleArray = os.listdir('{}/{}'.format(DataPath, folder))

    #Calculate Train To Test Splits
    train_size  = int(len(sampleArray)*train_to_test)
    test_size   = len(sampleArray)-train_size
    train_done  = 0
    test_done   = 0
    num = 1
    for sample in sampleArray:

        signal, sr = librosa.load('{}/{}/{}'.format(DataPath, folder, sample), sr=22050)

        my_hop_len = 512

        # MFCCs
        # mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=512, hop_length=512)
        D = np.abs(librosa.stft(signal))
        spec = librosa.amplitude_to_db(D, ref=np.max)
        librosa.display.specshow(spec, sr=sr, hop_length=my_hop_len)

        # Randomlly Split The Code into Train and Test Sets With A Percentage
        r = random.randint(0,1)
        if (r == 1 and test_done <test_size) or train_done == train_size:
            try:
                os.makedirs("{}/{}/".format(testSpectroPath, folder))
            except:
                pass
            plt.savefig(
                "{}/{}/{} - {}.png".format(testSpectroPath, folder, sample, num))
            test_done+=1
        else:

            try:
                os.makedirs("{}/{}/".format(SpecrtoPath, folder))
            except:
                pass
            plt.savefig("{}/{}/{} - {}.png".format(SpecrtoPath, folder, sample, num))
            train_done+=1
        num += 1
    print("Folder - {} Done with Splits: Train: {}, Test:{}".format(folder,train_size,test_size))
