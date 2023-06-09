import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np
labels = ['UnContaminatedMeat', 'ContaminatedMeat']
img_size = 224


def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                # convert BGR to RGB format
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]
                # Reshaping images to preferred size
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


train = get_data('Dataset')
val = get_data('Dataset')
l = []
for i in train:
    if (i[1] == 0):
        l.append("UnContaminatedMeat")
    else:
        l.append("ContaminatedMeat")
sns.set_style('darkgrid')
# sns.countplot(l)
plt.figure(figsize=(5, 5))
plt.imshow(train[1][0])
plt.title(labels[train[0][1]])

plt.figure(figsize=(5, 5))
plt.imshow(train[-1][0])
plt.title(labels[train[-1][1]])
x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=30,
    zoom_range=0.2,  # Randomly zoom image
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images


datagen.fit(x_train)
model = Sequential()
model.add(Conv2D(32, 3, padding="same",
          activation="relu", input_shape=(224, 224, 3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()
opt = Adam(lr=0.000001)
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True), metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=2,
                    validation_data=(x_val, y_val))  # epochs = 500
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(2)  # range(500)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy - 1')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss - 1')
plt.show()
# predictions = model.predict_classes(x_val)
# predictions = predictions.reshape(1, -1)[0]
# print(classification_report(y_val, predictions,
#       target_names=['UnContaminatedMeat (Class 0)', 'ContaminatedMeat (Class 1)']))
model.save('meat_contamination_cnn_model.h5')

# base_model = tf.keras.applications.MobileNetV2(
#     input_shape=(224, 224, 3), include_top=False, weights="imagenet")
# base_model.trainable = False
# base_learning_rate = 0.00001
# model = tf.keras.Sequential([base_model,
#                              tf.keras.layers.GlobalAveragePooling2D(),
#                              tf.keras.layers.Dropout(0.2),
#                              tf.keras.layers.Dense(2, activation="softmax")
#                              ])

# model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# history = model.fit(x_train, y_train, epochs=2,
#                     validation_data=(x_val, y_val))

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs_range = range(2)

# plt.figure(figsize=(15, 15))
# plt.subplot(2, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy - 2')

# plt.subplot(2, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss - 2')
# plt.show()
# predictions = model.predict_classes(x_val)
# predictions = predictions.reshape(1, -1)[0]

# print(classification_report(y_val, predictions,
#       target_names=['Rugby (Class 0)', 'Soccer (Class 1)']))
# model.summary()
# base_model.summary()
# base_model.save('meat_contamination_cnn_baseModel.h5')
