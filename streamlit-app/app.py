import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.image import resize
from PIL import Image
import tensorflow as tf
import os

st.set_option('deprecation.showfileUploaderEncoding', False)

# ---------- Load the saved model safely ----------
MODEL_PATH = 'Brain_Tumor_cnn.h5'

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    st.error(f"Model file not found at {MODEL_PATH}. Please make sure it is in your repo.")
    st.stop()

# Define the class mapping
class_mapping = {
    0: "Glioma tumor",
    1: "No tumor",
    2: "Pituitary tumor",
    3: "Meningioma tumor"
}

# ---------- Image Processing Functions ----------
def equalize_image(image):
    equalized = cv2.equalizeHist(image)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def filtering(image, method, kernel_size):
    if method == 'Median':
        filtered_image = cv2.medianBlur(image, kernel_size)
    elif method == 'Mean':
        mean_kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        filtered_image = cv2.filter2D(image, -1, mean_kernel)
    elif method == 'Gaussian':
        sigma_x = st.slider("Select the Sigma Value", 0, 20, 1)
        filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x)
    return filtered_image

def edge_detection(image, method, low_threshold=50):
    if method == 'robert':
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])
        gradient_x = cv2.filter2D(image, -1, kernel_x)
        gradient_y = cv2.filter2D(image, -1, kernel_y)
    elif method == 'prewitt':
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        gradient_x = cv2.filter2D(image, -1, kernel_x)
        gradient_y = cv2.filter2D(image, -1, kernel_y)
    elif method == 'sobel':
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    elif method == 'canny':
        edges = cv2.Canny(image, low_threshold, low_threshold * 3)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    else:
        return image

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = np.uint8(255 * (gradient_magnitude / np.max(gradient_magnitude)))
    return gradient_magnitude

def morph_ops(image, kernel, thresh):
    _, gray_image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    closed_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    opened_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
    eroded_image = cv2.erode(gray_image, kernel, iterations=1)
    dilated_image = cv2.dilate(gray_image, kernel, iterations=1)
    return closed_image, opened_image, eroded_image, dilated_image

# ---------- Prediction Function ----------
def predict_class(image):
    img_resized = resize(image, (224, 224)) / 255.0
    img_resized = img_resized.numpy().reshape(1, 224, 224, 3)
    predictions = model.predict(img_resized)
    predicted_label = class_mapping[predictions.argmax(axis=1)[0]]
    return predicted_label

# ---------- Streamlit UI ----------
st.markdown("""
    <style>
    .stApp {
        background-color: white;
        background-attachment: fixed;
        background-size: cover;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Brain Tumor Classification and Analysis")
uploaded_image = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
choice = st.sidebar.selectbox("Choose the Operation", ('Predict Type','Histogram Equalization','Edge Detection','Morphological Operations','Filtering'))

if uploaded_image is not None:
    # Convert uploaded image to RGB and grayscale
    img_pil = Image.open(uploaded_image).convert('RGB')
    img_np = np.array(img_pil)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    if choice == 'Predict Type':
        st.subheader("Predict Type")
        st.image(img_np, caption="Uploaded Image", width=300)
        predicted_class = predict_class(img_np)
        st.subheader("Result")
        st.info(f"The predicted class for the image is: {predicted_class}")

    elif choice == 'Histogram Equalization':
        st.subheader("Histogram Equalization")
        equalized_image = equalize_image(img_gray)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_np, caption="Original Image", width=300)
        with col2:
            st.image(equalized_image, caption="Equalized Image", width=300)

    elif choice == 'Morphological Operations':
        kernel_size = st.number_input('Enter Kernel Size', 1, 10, 3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        thresh = st.slider("Select threshold", 0, 200, 50)
        c_im, o_im, e_im, d_im = morph_ops(img_gray, kernel, thresh)
        col1, col2 = st.columns(2)
        with col1:
            st.image(c_im, caption="Image After Closing", width=300)
            st.image(e_im, caption="Image After Erosion", width=300)
        with col2:
            st.image(o_im, caption="Image After Opening", width=300)
            st.image(d_im, caption="Image After Dilation", width=300)

    elif choice == "Edge Detection":
        method = st.selectbox("Select the Method", ("Canny", "Robert", "Sobel", "Prewitt"))
        edge = edge_detection(img_gray, method.lower())
        st.image(edge, clamp=True, channels='GRAY')

    elif choice == "Filtering":
        method = st.selectbox("Select the Method", ("Mean", "Median", "Gaussian"))
        if method == 'Mean':
            kernel_size = st.slider("Select the Kernel Size", 1, 20, 5)
        else:
            kernel_size = st.slider("Select the Kernel Size", 1, 21, 5, step=2)
        filtered = filtering(img_gray, method, kernel_size)
        st.image(filtered, clamp=True, channels='GRAY')

else:
    # ---------- About Page ----------
    st.header("About")
    st.write("""
    This application is designed to help you analyze brain tumor images. 
    It provides various image processing techniques and predicts the type of tumor present using a CNN.
    """)
    st.subheader("Types of Brain Tumors")
    st.write("- Glioma Tumor\n- No Tumor (Normal Brain)\n- Pituitary Tumor\n- Meningioma Tumor")
    st.header("Supported Operations")
    st.write("""
    - Predict Type: Predicts tumor type using a deep learning model.
    - Edge Detection: Highlights object boundaries in the image.
    - Histogram Equalization: Improves image contrast.
    - Morphological Operations: Processes binary images to enhance features.
    - Filtering: Removes noise using Mean, Median, or Gaussian filters.
    """)
    st.header("Creator Info")
    st.write("- Chris Vinod Kurian\n- Drishtti Narwal\n- Gaurav Prakash\n- Hevardhan Saravanan")
    st.warning("This app is for educational purposes only and is not a substitute for professional medical advice.")
