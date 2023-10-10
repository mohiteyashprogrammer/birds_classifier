import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Set a random seed to ensure different random selections each time
np.random.seed()

# Path to your test data directory
testing_directory = 'C:\\Users\\yash mohite\\OneDrive\\Documents\\birds_classifier\\test'

# Load your trained model
model = tf.keras.models.load_model('C:\\Users\\yash mohite\\OneDrive\\Documents\\birds_classifier\\model.h5')

# List all subdirectories (bird name folders) in the testing directory
bird_name_folders = [folder for folder in os.listdir(testing_directory) if os.path.isdir(os.path.join(testing_directory, folder))]

# Streamlit app
st.title("Bird Image Classifier")

# Upload a new image
uploaded_image = st.file_uploader("Upload a bird image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make predictions on the uploaded image
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction[0])

    # Get the predicted class name directly from the folder name
    predicted_class_name = bird_name_folders[predicted_class_index]

    # Display the prediction
    st.title(f"Predicted Bird Name: {predicted_class_name}")
