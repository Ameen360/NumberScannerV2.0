import streamlit as st
import cv2
import numpy as np
from models.plate_recognition_system import PlateRecognitionSystem

system = PlateRecognitionSystem()

st.title("License Plate Recognition System")

uploaded_file = st.file_uploader("Upload a license plate image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

        plate_number, plate_type, confidence = system.process_plate(img)

        st.write("### Recognition Results")
        st.write(f"**Plate Number:** {plate_number}")
        st.write(f"**Plate Type:** {plate_type}")
        st.write(f"**Confidence:** {confidence:.1f}%")

        vis_img, _ = system.visualize_results(img)
        st.image(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB), caption="Recognition Visualization", use_container_width=True)
    else:
        st.error("Error: Could not read the uploaded image.")