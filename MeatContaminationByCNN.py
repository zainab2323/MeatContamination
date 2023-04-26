# Import the necessary libraries
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# Define the data directories
train_dir = 'path/to/train_data_directory'
validation_dir = 'path/to/validation_data_directory'
test_dir = 'path/to/test_data_directory'

# Define the image dimensions and batch size
img_width, img_height = 100, 100
batch_size = 32

# Define the data generator for image augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,        # rescale pixel values between 0 and 1
    shear_range=0.2,       # randomly apply shearing transformation
    zoom_range=0.2,        # randomly apply zoom transformation
    horizontal_flip=True)  # randomly flip images horizontally

validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# Generate the training, validation and test datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
          input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Train the model
model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

# Evaluate the model on the test dataset
scores = model.evaluate_generator(
    test_generator, steps=test_generator.samples // batch_size)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])
