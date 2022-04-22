import streamlit as st
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("savedmodel.h5")
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
img_width = 128
img_height = 128

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    with open ('data/sample.jpg','wb') as file:
          file.write(img_file_buffer.getbuffer())

    data_dir = './data'
    sample = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(img_height, img_width),
        shuffle=False,
        label_mode=None,
        batch_size=32,
        color_mode='rgb',
        interpolation='nearest'
        )
    y_pred = model.predict(sample)
    answer = classes[np.argmax(y_pred)]

    st.text(answer)