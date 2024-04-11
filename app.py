import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('my_best_model.hdf5')
def preprocess_image(uploaded_file):
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize the image
    image = cv2.resize(image, (32, 32))

    # Normalize the pixel values
    image = image / 255

    return image

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    preprocessed_image = preprocess_image(uploaded_file)

    # Make a prediction (replace with your model prediction code)
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))

    # Display the prediction result
    maxnum = max(prediction[0])
    print(maxnum)
    if maxnum == prediction[0][0]:
        st.write("The Predicted class is Early Blight")
    elif maxnum == prediction[0][1]:
        st.write("The Predicted class is Healthy")
    else:
        st.write("The Predicted class is Late Blight")
   
    