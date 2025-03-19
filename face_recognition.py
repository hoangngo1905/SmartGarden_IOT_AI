import os
import cv2
import numpy as np
import pickle
import threading
import queue
import requests
from datetime import datetime
from insightface.app import FaceAnalysis

# URL của server Flask
SERVER_URL = "http://localhost:5000/update"  # Đổi thành địa chỉ IP của server nếu cần

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

# Biến theo dõi trạng thái xác thực
authentication_sent = False
last_auth_time = None
auth_cooldown = 10  # Thời gian chờ giữa các lần gửi xác thực (giây)

# Luồng đọc camera liên tục
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

print("⚡ Hệ thống xác thực khuôn mặt đã sẵn sàng...")
print("📷 Đang kết nối với máy chủ web vườn thông minh...")
print(f"🔗 URL máy chủ: {SERVER_URL}")
print("🔍 Đặt camera đối diện và nhìn thẳng để xác thực!")

while True:
    if frame_queue.empty():
        continue

    frame = frame_queue.get()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB

    # Nhận diện khuôn mặt
    faces = face_app.get(rgb_frame)

    current_time = datetime.now()
    
    # Vẽ thông tin trạng thái
    cv2.putText(frame, "Trang thai: Dang tim kiem khuon mat...", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if authentication_sent and last_auth_time:
        time_diff = (current_time - last_auth_time).total_seconds()
        if time_diff < auth_cooldown:
            remaining = int(auth_cooldown - time_diff)
            cv2.putText(frame, f"Da xac thuc! Gui lai sau {remaining}s", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"Xac thuc: {best_match}", (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Do tin cay: {best_score:.2f}", (bbox[0], bbox[3] + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Kiểm tra nếu chưa gửi xác thực hoặc đã qua thời gian chờ
            if not authentication_sent or (last_auth_time and (current_time - last_auth_time).total_seconds() > auth_cooldown):
                # Gửi request đến server Flask
                try:
                    response = requests.post(SERVER_URL, json={"user": best_match})
                    print(f"✅ Da gui du lieu den server: {response.status_code}")
                    authentication_sent = True
                    last_auth_time = current_time
                except Exception as e:
                    print(f"⚠ Loi khi gui request: {e}")
        else:
            cv2.putText(frame, "Khong xac dinh", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

    # Hiển thị video
    cv2.imshow("He thong xac thuc khuon mat - Vuon Thong Minh", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()