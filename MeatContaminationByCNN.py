# Import the necessary libraries
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# Set the parameters
batch_size = 32
epochs = 10
num_classes = 3
input_shape = (100, 100, 3)

# Define the data generators for training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'train_data_directory',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')
val_generator = train_datagen.flow_from_directory(
    'train_data_directory',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
          activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'], run_eagerly=True)

# Train the model
# model.fit_generator(train_generator,
#                     steps_per_epoch=train_generator.samples // batch_size,
#                     validation_data=val_generator,
#                     validation_steps=val_generator.samples // batch_size,
#                     epochs=epochs)

model.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size, epochs=10,
          validation_data=val_generator, validation_steps=val_generator.samples // batch_size)


# Save the trained model
model.save('meat_contamination_cnn.h5')
