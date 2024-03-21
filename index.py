import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Function to load models
@st.cache(allow_output_mutation=True)
def load_models():
    model1 = tf.keras.models.load_model("CoMoFoD128(prewit).h5")
    model2 = tf.keras.models.load_model("CASFOD128(prewit).h5")
    return model1, model2

# Load models
model1, model2 = load_models()

st.title("Splicing and Copy-move Detection App")

# When the input changes, the cached models will be used
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg"])
selected_model = st.selectbox("Select model:", ["Model 1", "Model 2"])

# Function to preprocess image
def preprocess_image(image):
    # Resize image to 128x128 pixels
    image_resized = image.resize((128, 128))
    # Convert image to array
    image_array = np.array(image_resized) / 255.0  # Normalize pixel values
    # Expand dimensions to match input shape of the model
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict(image_array, model):
    return model.predict(image_array)

if uploaded_file is not None:
    # Read image as bytes
    image_bytes = uploaded_file.read()
    # Open image using PIL
    image = Image.open(io.BytesIO(image_bytes))
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image_array = preprocess_image(image)

    # Select the model
    if selected_model == "Model 1":
        prediction = predict(image_array, model1)
    else:
        prediction = predict(image_array, model2)

    st.write(prediction)

    # Display prediction result
    if prediction[0][1] > prediction[0][0]:
        st.write("The image is detected as an original image.")
    else:
        st.write("The image is detected as a tampered image.")
