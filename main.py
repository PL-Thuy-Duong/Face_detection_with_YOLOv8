from ultralytics import YOLO
from PIL import Image
import streamlit as st
import os
import cv2
import numpy as np
import base64
import threading
import time

stop_webcam = False  
webcam_thread = None  




def detect_faces_video(name_file):
    with st.spinner('Running YOLO model on video...'):
        time.sleep(7)
        model = YOLO("./data/runs/detect/train5/weights/best.pt")
        video_path = './uploaded/' + name_file
        cap = cv2.VideoCapture(video_path)
        processed_images = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_webcam:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = model.predict(source=frame_rgb, save=True)
            path = results[0].save_dir

            image_path = os.path.join(path, 'image0.jpg')

            processed_images.append(cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_RGB2BGR))

        cap.release()
        if processed_images:
            out_path = './runs/detect/output_video.mp4'
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (processed_images[0].shape[1], processed_images[0].shape[0]))

            for img in processed_images:
                out.write(img)

            out.release()

            st.success("Video saved successfully!")
            # st.markdown(f'<video src="data:video/mp4;base64,{base64.b64encode(open(out_path, "rb").read()).decode()}" width="640" height="480" controls></video>', unsafe_allow_html=True)
    st.download_button(
            label="Download Video",
            data=open(out_path, "rb").read(),
            key="download_processed_video",
            file_name="results_video.mp4",
            mime="video/mp4",
        )


def webcam_thread_func():
    global stop_webcam

    model = YOLO("./data/runs/detect/train5/weights/best.pt")

    cap = cv2.VideoCapture(0)
    processed_images = []

    while not stop_webcam:
        ret, frame = cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        results = model.predict(source=frame_rgb, save=True)
        path = results[0].save_dir

        image_path = os.path.join(path, 'image0.jpg')

        if os.path.exists(image_path):
            processed_images.append(cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_RGB2BGR))

    cap.release()

    if processed_images:
        out_path = './runs/detect/output_video.mp4'
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10 , (processed_images[0].shape[1], processed_images[0].shape[0]))

        for img in processed_images:
            out.write(img)

        out.release()

        st.success("Video saved successfully!")
        st.markdown(f'<video src="data:video/mp4;base64,{base64.b64encode(open(out_path, "rb").read()).decode()}" width="640" height="480" controls></video>', unsafe_allow_html=True)
        st.download_button(
            label="Download Processed Video",
            data=open(out_path, "rb").read(),
            key="download_processed_video",
            file_name="processed_video.mp4",
            mime="video/mp4",
        )
def run_model(image):
    model = YOLO("./data/runs/detect/train5/weights/best.pt")
    im1 = Image.open(image)
    results = model.predict(source=im1, save=True)
    path = results[0].save_dir
    return path

def detect_faces_webcam():
    global stop_webcam
    global webcam_thread

    st.text("Running YOLO model on webcam...")

    stop_button = st.button("Stop Webcam")

    if stop_button:
        stop_webcam = True
        if webcam_thread is not None: 
            webcam_thread.join()
        st.text("Webcam stopped successfully!")

    if webcam_thread is None or not webcam_thread.is_alive():
        stop_webcam = False
        webcam_thread = threading.Thread(target=webcam_thread_func)
        webcam_thread.start()


def main():

    st.title("FACE DETECTION USING YOLO-V8")

    tab1, tab2, tab3 = st.tabs(["IMAGE", "VIDEO", "WEBCAM"])

    with tab1:

        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:

            save_directory = "uploaded"
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            saved_path = os.path.join(save_directory, uploaded_file.name)

            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            input_image = './uploaded/' + uploaded_file.name
            with st.spinner('Running YOLO model...'):
                time.sleep(7)
                path_image = run_model(input_image)
                path_output_imaged = os.path.join(path_image, uploaded_file.name)

                if os.path.exists(path_output_imaged):
                    st.success('Image is detected successfully!')
                    st.image(path_output_imaged, caption="Output Image", use_column_width=True)
                else:
                    st.write("Processed image not found. Check the paths and file existence.")  
        
    with tab2:
        file_path = './runs/detect/output_video.mp4'
        if(file_path == True):
            os.remove(file_path)
        else:
            pass
        video_file = st.file_uploader("Choose a video file", type=["mp4"])
        if video_file is not None:
            name_file = video_file.name
            video_path = './uploaded/' + name_file

            with open(video_path, 'wb') as f:
                f.write(video_file.getvalue())
        
            
            detect_faces_video(name_file)
            # # st.video(annotated_frame)      

    with tab3:

        # detect_faces_webcam()
        img_file_webcam = st.camera_input("Please take a picture")

        if img_file_webcam is not None:
          
            save_directory = "uploaded"
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            saved_path = os.path.join(save_directory, img_file_webcam.name)

            with open(saved_path, "wb") as f:
                f.write(img_file_webcam.getvalue())

            with st.spinner('Running YOLO model...'):
                time.sleep(7)
                path_image = run_model(saved_path)
                path_output_imaged = os.path.join(path_image, img_file_webcam.name)

                if os.path.exists(path_output_imaged):
                    st.success('Image is detected successfully!')
                    st.image(path_output_imaged, caption="Output Image", use_column_width=True)
                else:
                    st.write("Processed image not found. Check the paths and file existence.")
            
        

def footer():
    st.markdown("""
            <div style="text-align: center; margin-top: 20px; font-size: 12px;">
                <p>© 2023 Team HDT - Python. All rights reserved.</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    footer()

    

