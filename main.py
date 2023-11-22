from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import streamlit as st
from PIL import Image
import os

def run_model(image):
    model = YOLO("./data/runs\detect/train5/weights/best.pt")
    im1 = Image.open(image)
    results = model.predict(source=im1, save=True) 
    path = results[0].save_dir
    # save plotted images
    return path

def main():
    st.title("Face_detection_with_YOLOv8")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        save_directory = "uploaded"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        saved_path = os.path.join(save_directory, uploaded_file.name)
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getvalue())
    
        input_image = './uploaded/' + uploaded_file.name

        # Display the original image
        # st.image(input_image, caption="Original Image", use_column_width=True)

        st.text("Running YOLO model...")
        # Run the YOLO model
        path_image = run_model(input_image)
        path_output_imaged = '.\\' + path_image + '\\' + uploaded_file.name
        # path_imaged = "./runs/detect/predict12/00000004.jpg"

        # Display the processed image
        st.image(path_output_imaged, caption="Output Image", use_column_width=True)
if __name__ == "__main__":
    main()


