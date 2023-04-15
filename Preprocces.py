import wave
import math
import os
from pydub import AudioSegment


# Get The Longest Duration For All Sound Files For Pading Operation
origin = './Data/'
# origin = './FINAL test'
distiny = './PreprossedData'
allFiles = os.listdir(origin)
print(allFiles)
maxDuration = 0
for folder in allFiles:
    files = os.listdir('{}/{}/'.format(origin, folder))
    for file in files:
        obj = wave.open("{}/{}/{}".format(origin, folder, file))
        n_sample = obj.getnframes()
        n_sample_rate = obj.getframerate()
        duration = n_sample/n_sample_rate
        if duration > maxDuration:
            maxDuration = duration

        if duration > 3:
            print("{}/{}/{}".format(origin, folder, file))
# Making the Sample in Same Duration
print("max : ", maxDuration)
allFolders = os.listdir(origin)

for folder in allFolders:

    sampleArray = os.listdir('{}/{}/'.format(origin, folder))
    pad_ms = int(math.ceil(maxDuration)*1000)
    for sample in sampleArray:
        audio = AudioSegment.from_wav('{}/{}/{}'.format(origin, folder, sample))
        if pad_ms < len(audio):
            print("Audio with {} was longer that {} second.".format(len(audio), pad_ms))
            continue
        silence = AudioSegment.silent(duration=pad_ms-len(audio)+1)

        padded = audio + silence  # Adding silence after the audio
        try:
            padded.export('{}/{}/{}'.format(distiny, folder, sample), format='wav')
        except:
            os.makedirs('{}/{}'.format(distiny, folder))
            padded.export('{}/{}/{}'.format(distiny, folder, sample), format='wav')

    print("{} Done".format(folder))
print('All Done')


# import wave
# import math
# import os
# from pydub import AudioSegment


# # Get The Longest Duration For All Sound Files For Pading Operation
# # origin = './Data/'
# origin = './FINAL test'
# distiny = './Final Procces'
# allFiles = os.listdir(origin)
# print(allFiles)
# maxDuration = 0
# files = os.listdir('{}/'.format(origin))
# for file in files:
#     obj = wave.open("{}/{}".format(origin, file))
#     n_sample = obj.getnframes()
#     n_sample_rate = obj.getframerate()
#     duration = n_sample/n_sample_rate
#     if duration > maxDuration:
#         maxDuration = duration

#     if duration > 4:
#         print("{}/{}".format(origin, file))
# # Making the Sample in Same Duration
# print("max : ", maxDuration)
# allFolders = os.listdir(origin)


# sampleArray = os.listdir('{}/'.format(origin))
# pad_ms = int(math.ceil(44)*1000)
# for sample in sampleArray:
#     audio = AudioSegment.from_wav(
#         '{}/{}'.format(origin, sample))
#     if pad_ms < len(audio):
#         print
#         print("Audio with {} was longer that {} second.".format(
#             len(audio), pad_ms))
#         continue
#     silence = AudioSegment.silent(duration=pad_ms-len(audio)+1)

#     padded = audio + silence  # Adding silence after the audio
#     try:
#         padded.export(
#             '{}/{}'.format(distiny, sample), format='wav')
#     except:
#         os.makedirs('{}/'.format(distiny))
#         padded.export(
#             '{}/{}'.format(distiny, sample), format='wav')

# # print("{} Done".format(folder))
# print('All Done')
