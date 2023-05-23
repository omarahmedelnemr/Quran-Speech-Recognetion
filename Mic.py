import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import librosa
from matplotlib import pyplot as plt
import numpy as np
import pyaudio
import wave
from pydub import AudioSegment
import tensorflow as tf
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL

from keras.utils import load_img, img_to_array

# Define the size of the audio buffer
CHUNK = 1024

# Initialize PyAudio
p = pyaudio.PyAudio()



# Define the length of the audio sample in seconds
sample_length = 2

# Define the format of the audio data
format = pyaudio.paInt16

# Define the number of channels in the audio data
channels = 1

# Define the sample rate of the audio data
sample_rate = 44100

# Define the name of the output file
output_file = "output.wav"

# Define the number of samples to record
samples_to_record = int(sample_rate / CHUNK * sample_length)

# Initialize a buffer for accumulating audio data
buffer = []


# Load the model 1
model1_name = "Teachable Machine"
model1 = load_model("./Models/teachable nonPre/keras_model.h5", compile=False)
print("model ,", model1)
# Load the labels
class_names = open("./Models/teachable nonPre/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
dataImg = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)





# Clear the console
if os.name == "nt":
    os.system("cls")  # For Windows
else:
    os.system("clear")  # For Unix/Linux/MacOS

# Continuously read audio data from the stream
n = 0

print('Say Somthing ... ')

# Open a stream to listen to the microphone
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100,
                input=True, frames_per_buffer=CHUNK)
while n < 1:
    # Read audio data from the stream
    data = stream.read(CHUNK, exception_on_overflow=False)

    # Add the new audio data to the buffer
    buffer.append(data)
    # If the buffer contains enough data for a new sample
    if len(buffer) >= samples_to_record:
        print("Analysing .. ")
        # Truncate the buffer to the required length
        buffer = buffer[-samples_to_record:]

        # Convert the buffer to a byte string
        data = b"".join(buffer)

        # Create a new WAV file
        with wave.open(output_file, "wb") as wav_file:
            # Set the WAV file parameters
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(p.get_sample_size(format))
            wav_file.setframerate(sample_rate)

            # Write the audio data to the WAV file
            wav_file.writeframes(data)

        # pad_ms = int(5*1000)
        # audio = AudioSegment.from_wav('output.wav')
        # silence = AudioSegment.silent(duration=pad_ms-len(audio)+1)
        # padded = audio + silence  # Adding silence after the audio
        # padded.export('output.wav', format='wav')

        signal, sr = librosa.load(
            'output.wav', sr=22050)

        fft = np.fft.fft(signal)
        magnitude = np.abs(fft)

        my_n_fft = 2048
        my_hop_len = 512

        # MFCCs
        D = np.abs(librosa.stft(signal))
        mfcc = librosa.amplitude_to_db(D, ref=np.max)
        # mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        # # librosa.feature.mfcc

        librosa.display.specshow(mfcc, sr=sr, hop_length=my_hop_len)
        plt.savefig('output.png')
        # Clear the buffer
        buffer = []

        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Open Saved Image
        image = Image.open("output.png").convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        dataImg[0] = normalized_image_array

        # Predicts the model 1
        prediction = model1.predict(dataImg)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Model 1 ({}): ".format(model1_name))
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)

        n += 1
    # Wait for a short time before taking the next audio sample
    p.get_default_output_device_info()
    p.get_default_input_device_info()
    p.get_host_api_count()
    p.get_device_info_by_host_api_device_index(0, 1)
    p.get_device_info_by_index(0)
    p.get_device_count()
p.terminate()
