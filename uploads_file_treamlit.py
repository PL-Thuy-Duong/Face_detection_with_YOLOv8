from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import numpy as np

model = YOLO("/HK1/HK1-Nam3/PYTHON/DO_AN/data/runs\detect/train5/weights/best.pt")

im1 = Image.open("dataset/images/00000003.jpg")
results = model.predict(source=im1, save=True)  # save plotted images
 
path_final = "/HK1/HK1-Nam3/PYTHON/DO_AN/runs/detect/predict/00000061.jpg"
# Display the PIL image using OpenCV
im1_cv2 = cv2.cvtColor(np.array(path_final), cv2.COLOR_RGB2BGR)
# mỗi lần chạy là mõi lần chạy là một lần tạo predict
cv2.imshow('test', im1_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()
