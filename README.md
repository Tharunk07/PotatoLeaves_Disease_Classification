# PotatoLeaves_Disease_Classification
Deep Learning model to predict the disease in the leaves of potato

# Introduction
This project is a potato leaves disease prediction model using convolutional neural networks (CNNs). The model is trained to classify potato leaves images into four different classes: "Healthy", "Early Blight", "Late Blight".

# Requirements

Python 3.x
TensorFlow 2.x
Keras
OpenCV
Matplotlib
Numpy
# Data
The dataset used for training and testing the model consists of potato leaves images collected from different sources. Each image is labeled with one of the three classes mentioned above. The data is divided into 80% for training and 20% for testing.

# Model
The model is a CNN architecture, which includes several convolutional and pooling layers, followed by fully connected layers. The model is trained using the Adam optimizer and categorical cross-entropy as the loss function.

# Training
The model is trained for a total of 10 epochs with a batch size of 32. The training process takes around 5 minutes on a GPU.

# Evaluation
The trained model is then evaluated on the test data. The model is able to achieve an accuracy of over 85% on the test set.

# Usage
The trained model can be used to classify new potato leaves images. To use the model, you can use the predict() function of the model, which takes an image as an input and returns the class label as an output.

# Conclusion
The potato leaves disease prediction model using CNNs is a powerful tool for identifying different diseases in potato leaves. The model achieved high accuracy on the test set and can be used for further research and development in the field of plant disease diagnosis.
