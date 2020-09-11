from keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np

model = load_model('/home/nghianguyen/Documents/streamlit/dog_cat/catvsdog.h5')
#button = st.button('Upload')
#if button:
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
st.set_option('deprecation.showfileUploaderEncoding', False)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = image.resize((150, 150), Image.ANTIALIAS) 
    img = np.array(img)
    #img = img_to_array(img)
    img = img/255.0
    img = img.reshape(1, 150, 150, 3)
    result = model.predict(img)
    st.write("Result")
    if result[0] >= 0.5:
        st.image(image, "Dog", use_column_width=True)
    else:
        st.image(image, "cat", use_column_width=True)
