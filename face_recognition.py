import os
import cv2
import numpy as np
import pickle
import threading
import queue
import requests
from insightface.app import FaceAnalysis

# URL của server Flask
SERVER_URL = "http://192.168.0.104:5000/update"

# Nạp dữ liệu khuôn mặt từ file
with open("face_db.pkl", "rb") as f:
    face_db = pickle.load(f)

# Khởi tạo InsightFace
face_app = FaceAnalysis(providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Mở camera (0 = camera laptop)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Queue lưu trữ frames từ camera
frame_queue = queue.Queue(maxsize=1)  # Giữ tối đa 1 frame để giảm độ trễ

# 📌 **Luồng đọc camera liên tục**
def camera_reader():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.empty():
            frame_queue.get_nowait()  # Giữ frame mới nhất
        frame_queue.put(frame)

# Chạy thread để đọc camera song song
thread = threading.Thread(target=camera_reader, daemon=True)
thread.start()

while True:
    if frame_queue.empty():
        continue

    frame = frame_queue.get()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB

    # Nhận diện khuôn mặt
    faces = face_app.get(rgb_frame)

    for face in faces:
        bbox = face.bbox.astype(int)
        face_embedding = face.embedding

        best_match = None
        best_score = -1

        # Duyệt qua từng người trong face_db
        for img_name, data in face_db.items():
            db_embedding = data["embeddings"]
            similarity = np.dot(face_embedding, db_embedding) / (np.linalg.norm(face_embedding) * np.linalg.norm(db_embedding))

            if similarity > best_score:
                best_score = similarity
                best_match = img_name

        # Nếu tìm thấy khuôn mặt khớp (ngưỡng 0.5)
        if best_score > 0.5:
            cv2.putText(frame, f"Xac thuc: {best_match}", (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

            # Gửi request đến server Flask
            try:
                response = requests.post(SERVER_URL, json={"user": best_match})
                print(f"✅ Da gui du lieu den server: {response.status_code}")
            except Exception as e:
                print(f"⚠ Loi khi gui request: {e}")

        else:
            cv2.putText(frame, "Khong xac dinh", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

    # Hiển thị video
    cv2.imshow("Face Recognition", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
