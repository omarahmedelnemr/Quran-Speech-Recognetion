from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

image_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)
training_set = image_datagen.flow_from_directory('Spectro/',
                                                 target_size=(64, 64),
                                                 batch_size=16,
                                                 class_mode='binary',
                                                 subset='training')

test_set = image_datagen.flow_from_directory('Spectro/',
                                             target_size=(64, 64),
                                             batch_size=16,
                                             class_mode='binary',
                                             subset='validation')

Validation_datagen = ImageDataGenerator(rescale=1./255)
Validation_set = Validation_datagen.flow_from_directory('TestSpectro/',
                                                    target_size=(64, 64),
                                                    batch_size=16,
                                                    class_mode='binary',
                                                    )

print('--------------------------------------------')
print(len(training_set))
print(len(test_set))
print(training_set)
print(test_set)
print('--------------------------------------------')

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x=training_set, batch_size=16 ,validation_data=test_set, epochs=100)

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x=training_set, batch_size=16 ,validation_data=test_set, epochs=100)

print("Done Training!")

cnn.save('Models/Edgham1')
print("Model Saved!")


cnn.evaluate(Validation_set)

print("Done Training!")

cnn.save('Models/Edgham1')
print("Model Saved!")


cnn.evaluate(Validation_set)
