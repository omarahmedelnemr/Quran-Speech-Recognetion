import tensorflow as tf
import numpy as np
import random
import cv2
import os


codes = {'Right':1,"Wrong":0}
# create a list of file paths to your spectrogram images

# read the images from file and append them to the image list
file_paths = os.listdir('./Spectro')  # insert your list of file paths here
images = []
labels =[]
for file_path in file_paths:
    files = os.listdir("./Spectro/{}".format(file_path))
    for file in files:

        image = cv2.imread("./Spectro/{}/{}".format(file_path,file))
        print("IMG: ",image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB color space
        images.append(image)
        labels.append(file_path)
# set the random seed for reproducibility
random.seed(42)

# define the classes
classes = ['Right', 'Wrong']

# # create a list of spectrogram images and labels
# labels = file_paths  # insert the corresponding labels here (e.g. ['class1', 'class2', ...])

# shuffle the data
data = list(zip(images, labels))
# print(data)
random.shuffle(data)

# split the data into training and test sets
split_index = int(len(data) * 0.8)  # 80% training, 20% test
training_data = data[:split_index]
test_data = data[split_index:]

# convert the data to numpy arrays
x_train = np.array([i[0] for i in training_data])
y_train = np.array([classes.index(i[1]) for i in training_data])
x_test = np.array([i[0] for i in test_data])
y_test = np.array([classes.index(i[1]) for i in test_data])

# normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0


print()
# define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(classes))
])

# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train the model
model.fit(x_train, y_train,batch_size=128, epochs=100, validation_data=(x_test, y_test))

model.save("Models/Edgham2")