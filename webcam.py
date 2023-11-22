import cv2
def detect_from_webcam():
    # sử dụng webcam
    cap = cv2.VideoCapture(0)  # 0 là thiết bị webcam mặc định
    # _, img = cap.read()
    # Tạo một bộ lọc khuôn mặt (haarcascade_frontalface_default.xml là một trong những bộ lọc có sẵn)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        _, img = cap.read()

        # Chuyển đổi ảnh sang ảnh xám để giảm chiều sâu màu
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt trong ảnh
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Vẽ hình chữ nhật xung quanh các khuôn mặt phát hiện được
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Hiển thị ảnh với khuôn mặt được phát hiện
        cv2.imshow('Detected Faces', img)
        # Thoát vòng lặp nếu người dùng nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Đóng webcam nếu sử dụng
    cap.release()
    # đóng cửa sổ
    cv2.destroyAllWindows()

import cv2
# import darknet

# Load cấu hình và trọng số của YOLOv3
# net = darknet.load_net_custom("yolov3-face.cfg".encode("utf-8"), "yolov3-face.weights".encode("utf-8"), 0, 1)
# meta = darknet.load_meta("face.data".encode("utf-8"))

# Tạo kết nối với webcam (0 là thiết bị webcam mặc định)
cap = cv2.VideoCapture(0)

while True:
    # Đọc khung hình từ webcam
    ret, frame = cap.read()

    # Resize khung hình theo kích thước mong muốn của YOLOv3
    sized = cv2.resize(frame, (darknet.network_width(net), darknet.network_height(net)))

    # Chuyển đổi từ BGR sang RGB
    darknet_image = darknet.make_image(darknet.network_width(net), darknet.network_height(net), 3)
    frame_rgb = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())

    # Dự đoán đối tượng trong ảnh
    detections = darknet.detect_image(net, meta, darknet_image, thresh=0.25)

    # Vẽ hình chữ nhật xung quanh khuôn mặt được phát hiện
    for detection in detections:
        x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
        x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Hiển thị khung hình với khuôn mặt được phát hiện
    
    cv2.imshow('Detected Faces (YOLOv3)', frame)

    # Thoát vòng lặp nếu người dùng nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()



