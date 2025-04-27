import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("Color Segmentation with KMenas Clustering!")
st.write("Use the slider to try different numbers of clusters or colors. Then press the button to segment the image.")
st.markdown("""
LinkedIn: linkedin.com/in/hooman-amini-ha3

**Notice:**
- Be patient if the image file is large, as processing may take time.
- Upload your image in JPG, PNG, or JPEG format, and choose the number of colors (clusters) to keep.

""")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
k_num = st.slider("Number of Colors", 2, 10, 8)
segment_button = st.button("CONVERT!")

if uploaded_file is not None and segment_button:
    image = np.array(Image.open(uploaded_file))
    fig = plt.figure()
    X = np.array(image).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k_num, random_state=42).fit(X)
    X_seg = kmeans.cluster_centers_[kmeans.labels_]
    plt.imshow(X_seg.reshape(image.shape).astype(np.uint8))
    st.write(f"{k_num} Colors Kept   | Image resolution: {image.shape[0] , image.shape[1]} ")
    plt.axis('off')
    st.pyplot(fig)
