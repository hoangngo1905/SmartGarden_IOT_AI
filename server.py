import os
import cv2
import numpy as np
import pickle
import threading
import queue
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from insightface.app import FaceAnalysis
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)  # Cho phép CORS

# Nạp dữ liệu khuôn mặt từ file
with open("face_db.pkl", "rb") as f:
    face_db = pickle.load(f)

# Khởi tạo InsightFace
face_app = FaceAnalysis(providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Biến lưu trạng thái xác thực
auth_status = {
    "authenticated": False,
    "user": None,
    "last_update": None
}

# Dữ liệu cảm biến
sensor_data = {
    "temperature": "28",
    "soil_moisture": "60%",
    "light": "750",
    "water_level": "75%"
}

# Route chính cho giao diện web
@app.route('/')
def index():
    return render_template('index.html')

# Route API trả về dữ liệu cảm biến
@app.route('/data')
def get_data():
    return jsonify(sensor_data)

# Route API nhận trạng thái xác thực từ ứng dụng nhận diện khuôn mặt
@app.route('/update', methods=['POST'])
def update_auth():
    data = request.json
    if 'user' in data:
        auth_status["authenticated"] = True
        auth_status["user"] = data["user"]
        auth_status["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Invalid data"})

# Route API kiểm tra trạng thái xác thực
@app.route('/auth_status')
def get_auth_status():
    return jsonify(auth_status)

if __name__ == '__main__':
      # Import thư viện datetime
    from datetime import datetime
    
    app.run(host='0.0.0.0', port=5000, debug=True)