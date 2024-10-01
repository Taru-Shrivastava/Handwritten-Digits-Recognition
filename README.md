# Handwritten-Digits-Recognition
This Python script uses TensorFlow and Keras to build and train a neural network for digit classification using the MNIST dataset. Here's a breakdown:

Data Loading and Preprocessing:

The MNIST dataset (handwritten digits) is loaded and split into training and test sets.
Pixel values are normalized by scaling them between 0 and 1.
Model Architecture:

A Sequential neural network is created.
The first layer flattens the 28x28 image input into a 1D vector.
A Dense hidden layer with 128 neurons and ReLU activation is added.
A Dropout layer helps prevent overfitting by randomly deactivating 20% of the neurons during training.
The output layer has 10 neurons, corresponding to the 10 digit classes (0-9), using softmax activation.
Model Compilation:

The model is compiled with the Adam optimizer, using sparse categorical cross-entropy as the loss function, and accuracy as the evaluation metric.
Training:

The model is trained on the MNIST training dataset for 5 epochs.
Evaluation and Prediction:

The trained model's performance is evaluated on the test set.
Predictions are made on the test data, and the predicted label for a sample image is displayed alongside the image using Matplotlib.
