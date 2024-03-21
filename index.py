import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    # Load the model
    model = tf.keras.models.load_model("CASFOD128(prewit).h5")
    return model

# Load the model
model = load_model()

st.title("Aplikasi Deteksi Penyuntingan dan Pemindahan Salinan pada Gambar")

# Ketika input berubah, model yang di-cache akan digunakan
uploaded_file = st.file_uploader("Pilih gambar...", type=["png","jpg"])

# Function to preprocess image
def preprocess_image(image):
    # Resize image to 128x128 pixels
    image_resized = image.resize((128, 128))
    # Convert image to array
    image_array = np.asarray(image_resized) / 255.0  # Normalize pixel values
    # Expand dimensions to match input shape of the model
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang Diunggah.', use_column_width=True)

    # Preprocess the image
    image_array = preprocess_image(image)

    # Predict the image
    prediction = model.predict(image_array)
    st.write(prediction)

    # Display prediction result
    if prediction[0][1] > prediction[0][0]:
        st.write("Gambar terdeteksi sebagai gambar asli.")
    else:
        st.write("Gambar terdeteksi sebagai gambar yang telah diubah.")
