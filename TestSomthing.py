import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PIL import Image, ImageOps  # Install pillow instead of PIL
from keras.preprocessing import image
from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np
from keras.utils import load_img, img_to_array



print("as")

# Clear the console
if os.name == "nt":
    os.system("cls")  # For Windows
else:
    os.system("clear")  # For Unix/Linux/MacOS

model2 = load_model('./Models/AyahModel')

img_for_Model2 = load_img("./output.png", target_size=(64, 64))
img_for_Model2 = img_to_array(img_for_Model2)

img_for_Model2 = np.expand_dims(img_for_Model2, axis=0)
result = model2.predict(img_for_Model2)
# Clear the console
if os.name == "nt":
    os.system("cls")  # For Windows
else:
    os.system("clear")  # For Unix/Linux/MacOS

print("Result: ", result)
