import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Function to load the models
@st.cache(allow_output_mutation=True)
def load_models():
    # Load the models
    model1 = tf.keras.models.load_model("CoMoFoD128(prewit).h5")
    model2 = tf.keras.models.load_model("CASFOD128(prewit).h5")
    return model1, model2

# Load the models
model1, model2 = load_models()

# Set page configuration
st.set_page_config(
    page_title="Image Authenticity Detection App",
    layout="wide"
)

# Function to preprocess image
def preprocess_image(image):
    # Resize image to 128x128 pixels
    image_resized = image.resize((128, 128))
    # Convert image to array
    image_array = np.array(image_resized) / 255.0  # Normalize pixel values
    # Expand dimensions to match input shape of the models
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Main content
st.title("Image Authenticity Detection App")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg"])

if uploaded_file is not None:
    # Read image as bytes
    image_bytes = uploaded_file.read()
    # Open image using PIL
    image = Image.open(io.BytesIO(image_bytes))
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image_array = preprocess_image(image)

    # Predict with model 1
    prediction1 = model1.predict(image_array)
    # Predict with model 2
    prediction2 = model2.predict(image_array)

    # Combine predictions
    combined_prediction = (prediction1 + prediction2) / 2

    # Display combined prediction result
    if combined_prediction[0][1] > combined_prediction[0][0]:
        st.write("The image is detected as a tampered image.")
    else:
        st.write("The image is detected as an original image.")
