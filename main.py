# Import Lib
import os
import numpy as np
import pandas as pd
import shutil
import cv2
import random
import matplotlib.pyplot as plt
import copy
import wandb
import streamlit as st
from ultralytics import YOLO
import ultralytics
import locale

wandb.login(key='79ecc228ee3ff622c9c004d4985ad6690dafd321')

# Variables
bs=' ' # blank-space
class_id=0 # id for face
newline='\n' # new line character
extension='.txt'

# Creating paths for separate images and labels
curr_path=os.getcwd()
imgtrainpath = os.path.join(curr_path,'images','train')
imgvalpath=os.path.join(curr_path,'images','validation')
imgtestpath=os.path.join(curr_path,'images','test')

labeltrainpath=os.path.join(curr_path,'labels','train')
labelvalpath=os.path.join(curr_path,'labels','validation')
labeltestpath=os.path.join(curr_path,'labels','test')

# Defining data path and labels_path
data_path='HK1/HK1-Nam3/PYTHON/DO_AN/dataset'
labels_path = os.path.join(curr_path, 'face_labels')

# Creating labels path
os.makedirs(labels_path)

# Defining input images and raw annotations path
img_path=os.path.join(data_path, 'images')
raw_annotations_path=os.path.join(data_path, 'faces.csv')

# Creating a list of all images
face_list=os.listdir(img_path)
data_len=len(face_list)
random.shuffle(face_list)

#split train, val, test 
train_split=0.8
val_split=0.1
test_split=0.1
imgtrain_list=face_list[:int(data_len*train_split)]
imgval_list=face_list[int(data_len*train_split):int(data_len*(train_split+val_split))]
imgtest_list=face_list[int(data_len*(train_split+val_split)):]

# Extract basename for file label
def change_extension(file):
    basename=os.path.splitext(file)[0]
    filename=basename+extension
    return filename
labeltrain_list = list(map(change_extension, imgtrain_list))
labelval_list = list(map(change_extension, imgval_list))
labeltest_list = list(map(change_extension, imgtest_list))

# Read file label
raw_annotations=pd.read_csv(raw_annotations_path)

# Transform Format
raw_annotations['x_centre']=0.5*(raw_annotations['x0']+raw_annotations['x1'])
raw_annotations['y_centre']=0.5*(raw_annotations['y0']+raw_annotations['y1'])
raw_annotations['bb_width']=raw_annotations['x1']-raw_annotations['x0']
raw_annotations['bb_height']=raw_annotations['y1']-raw_annotations['y0']
# Size bounding box
raw_annotations['xcentre_scaled']=raw_annotations['x_centre']/raw_annotations['width']
raw_annotations['ycentre_scaled']=raw_annotations['y_centre']/raw_annotations['height']
raw_annotations['width_scaled']=raw_annotations['bb_width']/raw_annotations['width']
raw_annotations['height_scaled']=raw_annotations['bb_height']/raw_annotations['height']

# Mỗi ảnh có thể có nhiều bounding box nên chạy từng ảnh
imgs=raw_annotations.groupby('image_name')

for image in imgs:
    img_df=imgs.get_group(image[0])
    basename=os.path.splitext(image[0])[0]
    txt_file=basename+extension
    filepath=os.path.join(labels_path, txt_file)
    lines=[]
    i=1
    for index,row in img_df.iterrows():
        if i!=len(img_df):
            line=str(class_id)+bs+str(row['xcentre_scaled'])+bs+str(row['ycentre_scaled'])+bs+str(row['width_scaled'])+bs+str(row['height_scaled'])+newline
            lines.append(line)
        else:
            line=str(class_id)+bs+str(row['xcentre_scaled'])+bs+str(row['ycentre_scaled'])+bs+str(row['width_scaled'])+bs+ str(row['height_scaled'])
            lines.append(line)
        i=i+1
    with open(filepath, 'w') as file:
        file.writelines(lines)

random_file=os.path.join(labels_path, os.listdir(labels_path)[4])
with open (random_file, 'r') as f:
    content=f.read()

# function to move files from source to detination
def move_files(data_list, source_path, destination_path):
    i=0
    for file in data_list:
        filepath=os.path.join(source_path, file)
        dest_path=os.path.join(data_path, destination_path)
        if not os.path.isdir(dest_path):
            os.makedirs(dest_path)
        shutil.move(filepath, dest_path)
        i=i+1
    print("Number of files transferred:", i)

# function to resize the images and copy the resized image to destination

def_size=640 # Image size for YOLOv8 

def move_images(data_list, source_path, destination_path):
    i=0
    for file in data_list:
        filepath=os.path.join(source_path, file)
        dest_path=os.path.join(data_path, destination_path)

        if not os.path.isdir(dest_path):
            os.makedirs(dest_path)
        finalimage_path=os.path.join(dest_path, file)
        img_resized=cv2.resize(cv2.imread(filepath), (def_size, def_size))
        cv2.imwrite(finalimage_path, img_resized)
        i=i+1
    print("Number of files transferred:", i)

shutil.rmtree(labels_path) # removing labels path as it is empty

#CREATE CONFIG FILE
ln_1='# Train/val/test sets'+newline
ln_2='train: ' +"'"+imgtrainpath+"'"+newline
ln_3='val: ' +"'" + imgvalpath+"'"+newline
ln_4='test: ' +"'" + imgtestpath+"'"+newline
ln_5=newline
ln_6='# Classes'+newline
ln_7='names:'+newline
ln_8='  0: face'
config_lines=[ln_1, ln_2, ln_3, ln_4, ln_5, ln_6, ln_7, ln_8]

# Creating path for config file
config_path=os.path.join(curr_path, 'config.yaml')

# Writing config file
with open(config_path, 'w') as f:
    f.writelines(config_lines)
#image visualisation
# function to obtain bounding box  coordinates from text label files
def get_bbox_from_label(text_file_path):
    bbox_list=[]
    with open(text_file_path, "r") as file:
        for line in file:
            _,x_centre,y_centre,width,height=line.strip().split(" ")
            x1=(float(x_centre)+(float(width)/2))*def_size
            x0=(float(x_centre)-(float(width)/2))*def_size
            y1=(float(y_centre)+(float(height)/2))*def_size
            y0=(float(y_centre)-(float(height)/2))*def_size

            vertices=np.array([[int(x0), int(y0)], [int(x1), int(y0)],
                               [int(x1),int(y1)], [int(x0),int(y1)]])
            bbox_list.append(vertices)

    return tuple(bbox_list)
# defining red color in RGB to draw bounding box
red=(255,0,0)


#TRAINING
model=YOLO('yolov8s.yaml').load('yolov8s.pt')
# Tắt WandB
wandb.finish()
#trained
# results=model.train(data=config_path, epochs=100, resume=True, iou=0.5, conf=0.001)

# Assuming 'curr_path' is defined earlier in your code
trainingresult_path = os.path.join(curr_path, 'runs', 'detect', 'train5')
results_path = os.path.join(trainingresult_path, 'results.png')

# Check if the file exists before attempting to read it
if os.path.exists(results_path):
    # Load the image using cv2.imread
    results_png = cv2.imread(results_path)

    # Debugging prints
    print(f"Loaded image: {results_png}")
    print(f"Image data type: {results_png.dtype}")

    # Check if the image data is not None and has the correct dtype
    if results_png is not None and results_png.dtype == 'uint8':
        # Explicitly set the data type to uint8 for imshow
        plt.figure(figsize=(30, 30))
        plt.imshow(results_png.astype('uint8'))
        plt.show()
    else:
        print("Error: Unable to load image or incorrect data type.")
else:
    print(f"Error: Image file not found at {results_path}.")

# if __name__ == "__main__":
   
#     st.set_option("deprecation.showfileUploaderEncoding", False)

#     # title area
#     st.markdown("""
#     # Face Recognition Using YOLOv8
#     > Powered by [*ageitgey* face_recognition](https://github.com) python engine
#     """)

    # # displays a file uploader widget and return to BytesIO
    # image_uploaded = st.file_uploader(
    #     label="Select a picture contains faces:", type=['jpg', 'png']
    # )
    # # detect faces in the loaded image
    # max_faces = 0
    # rois = []  # region of interests (arrays of face areas)
    # if image_uploaded is not None:
    #     image_array = byte_to_array(image_uploaded)
    #     face_locations = face_recognition.face_locations(image_array)
    #     for idx, (top, right, bottom, left) in enumerate(face_locations):
    #         # save face region of interest to list
    #         rois.append(image_array[top:bottom, left:right].copy())

    #         # Draw a box around the face and lable it
    #         cv2.rectangle(image_array, (left, top),
    #                       (right, bottom), COLOR_DARK, 2)
    #         cv2.rectangle(
    #             image_array, (left, bottom + 35),
    #             (right, bottom), COLOR_DARK, cv2.FILLED
    #         )
    #         font = cv2.FONT_HERSHEY_DUPLEX
    #         cv2.putText(
    #             image_array, f"#{idx}", (left + 5, bottom + 25),
    #             font, .55, COLOR_WHITE, 1
    #         )

    #     st.image(BGR_to_RGB(image_array), width=720)
    #     max_faces = len(face_locations)

   

    #     # add roi to known database
    #     if st.checkbox('add it to knonwn faces'):
    #         face_name = st.text_input('Name:', '')
    #         face_des = st.text_input('Desciption:', '')
    #         if st.button('add'):
    #             encoding = face_to_compare.tolist()
    #             DB.loc[len(DB)] = [face_name, face_des] + encoding
    #             DB.to_csv(PATH_DATA, index=False)
    # else:
    #     st.write('No human face detected.')

    # import streamlit as st
# import torch
# from PIL import Image
# from pathlib import Path
# import cv2
# import numpy as np
# import subprocess
# from io import BytesIO

# # Đường dẫn tới tập trọng số của mô hình YOLOv8
# model_path = '/HK1/HK1-Nam3/PYTHON/DO_AN/data/runs\detect/train5/weights/best.pt'

# # Ứng dụng Streamlit
# st.title('Ứng Dụng Dự Đoán Đối Tượng với YOLOv8')

# # Thêm các thành phần giao diện người dùng và sử dụng mô hình để dự đoán
# uploaded_file = st.file_uploader("Chọn hình ảnh", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Đọc và xử lý hình ảnh
#     image = Image.open(uploaded_file).convert("RGB")

#     # Chuyển đổi hình ảnh thành mảng NumPy
#     image_np = np.array(image)

#     # Lưu hình ảnh tạm thời
#     temp_image_path = 'temp_image.jpg'
#     cv2.imwrite(temp_image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

#     # Thực hiện dự đoán bằng YOLOv8
#     command = f"!yolo predict model={model_path} source={temp_image_path}"
#     process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     stdout, stderr = process.communicate()

#     # Hiển thị hình ảnh và kết quả dự đoán
#     #st.image(image, caption="Uploaded Image", use_column_width=True)
#     st.write("Kết quả dự đoán:")
#     st.code(stdout.decode())

#     # Hiển thị hình ảnh đã được dự đoán
#     result_image_path = stdout.decode().strip()
#     result_image = Image.open(result_image_path)
#     st.image(result_image, caption="Annotated Image", use_column_width=True)

#     # Xóa hình ảnh tạm thời
#     Path(temp_image_path).unlink()
import torch
from PIL import Image
from pathlib import Path
# from models.experimental import attempt_load
# from utils.general import non_max_suppression, scale_coords

# Đường dẫn tới trọng số tốt nhất của mô hình YOLOv8
weights_path = 'path/to/your/best.pt'

# Nạp mô hình YOLOv8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = attempt_load(weights_path, map_location=device)
stride = int(model.stride.max())  # Lấy stride lớn nhất

# Hàm dự đoán
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # chuyển định dạng và thêm chiều batch
    img_size = img_tensor.shape[-2:]  # kích thước hình ảnh

    # Dự đoán với mô hình
    with torch.no_grad():
        predictions = model(img_tensor.to(device))[0]
        predictions = non_max_suppression(predictions, conf_thres=0.5, iou_thres=0.4)[0]

    # Chuyển đổi tọa độ về kích thước gốc của hình ảnh
    if predictions is not None:
        predictions[:, :4] = scale_coords(img_tensor.shape[-2:], predictions[:, :4], img_size).round()

    return predictions

# Đường dẫn tới hình ảnh đầu vào
image_path = 'path/to/your/input_image.jpg'

# Thực hiện dự đoán
predictions = predict(image_path)

# Hiển thị kết quả dự đoán
if predictions is not None:
    for pred in predictions:
        print(f"- Đối tượng: {int(pred[5])}, Độ chắc chắn: {pred[4]:.2f}, Box: {pred[:4].tolist()}")
else:
    print("Không có đối tượng được dự đoán.")



