from keras.preprocessing import image
from keras import optimizers
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('3 to 3 Spec/training/train/',
                                                 target_size=(64, 64),
                                                 batch_size=16,
                                                 class_mode='binary')


test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory('3 to 3 Spec/training/test/',
                                            target_size=(64, 64),
                                            batch_size=16,
                                            class_mode='binary')

print('--------------------------------------------')
print(len(training_set))
print(len(test_set))
print(len(training_set[0][0]))
print('--------------------------------------------')


cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
        activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer=optimizers.Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x=training_set, validation_data=test_set, epochs=500)

print("Done Training!")

cnn.save('Models/CNN Model 4 nonPre')
print("Model Saved!")


cnn.evaluate(test_set)