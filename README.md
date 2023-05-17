# Meat Contamination Using Deep Learning (CNN)

Identification of Meat Contamination:

- Import the necessary libraries

  We start by importing the libraries we need, including Keras, NumPy, and ImageDataGenerator from Keras.
  Set the parameters: We set the batch size, number of epochs, number of classes, and the input shape for our CNN model.

- Define the data generators

  We use ImageDataGenerator to generate batches of augmented data for the training and validation sets. The rescale parameter is used to normalize the pixel values to the range of [0, 1]. The data is split into 80% for training and 20% for validation.

- Define the CNN model

  We define the CNN model using the Sequential API in Keras. The model consists of two Conv2D layers with ReLU activation, followed by two MaxPooling2D layers, a Flatten layer, a Dense layer with ReLU activation, a Dropout layer with a rate of 0.5, and a Dense output layer with Softmax activation.

- Compile the model

  We compile the model by specifying the loss function, optimizer, and metrics to use during training.

- Train the model

  We train the model using the fit_generator method, which takes the data generators as input. We also specify the number of steps per epoch and validation steps based on the batch size and number of samples in the data generators.

- Save the trained model

  Finally, we save the trained model to a file for future use.
