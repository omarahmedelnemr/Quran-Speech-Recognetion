from keras.preprocessing import image
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('Spectro/',
                                                 target_size=(64, 64),
                                                 batch_size=16,
                                                 class_mode='binary')


test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory('TestSpectro/',
                                            target_size=(64, 64),
                                            batch_size=16,
                                            class_mode='binary')
# tr = pd.DataFrame(training_set)

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
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x=training_set, validation_data=test_set, epochs=100)

print("Done Training!")

cnn.save('Models/AyahModel')
print("Model Saved!")


cnn.evaluate(test_set)


# test_image = image.load_img(
#     'Test/Specto/B/1 B.png', target_size=(64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
# result = cnn.predict(test_image)
# training_set.class_indices
# if result[0][0] == 1:
#     prediction = 'بسم الله الرحمن الرحيم'
# else:
#     prediction = 'قل هو الله احد'


# print(prediction)
