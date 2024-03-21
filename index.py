import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Function to load the model
@st.cache_resource
def load_model():
    # Load the model
    model = tf.keras.models.load_model("CASFOD(prewit).h5")
    return model

# Load the model
model = load_model()

st.title("Splicing and Copy-move Detection App")

# When the input changes, the cached model will be used
uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg"])

# Function to preprocess image
def preprocess_image(image):
    # Convert image to array
    image_array = np.asarray(image)
    # Expand dimensions to match input shape of the model
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image_array = preprocess_image(image)

    # Predict the image
    prediction = model.predict(image_array)
    st.write(prediction)

    # Display prediction result
    if prediction[0][1] > prediction[0][0]:
        st.write("The image is detected as a original image.")
    else:
        st.write("The image is detected as a tempred image.")
