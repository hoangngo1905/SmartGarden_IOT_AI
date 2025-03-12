from flask import Flask, render_template, jsonify, request, redirect

app = Flask(__name__)

# Biến toàn cục lưu trạng thái xác thực khuôn mặt
face_authenticated = False

# Dữ liệu cảm biến mặc định
sensor_data = {
    "temperature": "N/A",
    "humidity": "N/A",
    "soil_moisture": "N/A",
    "light": "N/A",
    "water_level": "N/A"
}

@app.route("/")
def home():
    """Trang chính"""
    return render_template("index.html")

@app.route("/data")
def get_data():
    """API trả về dữ liệu cảm biến"""
    return jsonify(sensor_data)

@app.route("/update", methods=["POST"])
def update_data():
    """API nhận dữ liệu từ ESP32 hoặc xác thực khuôn mặt"""
    global sensor_data, face_authenticated
    try:
        data = request.get_json()
        if not data:
            return jsonify({"message": "Dữ liệu JSON không hợp lệ!", "status": "error"}), 400
        
        print("📩 Dữ liệu nhận được:", data)  # Log dữ liệu

        # Nếu nhận dữ liệu cảm biến từ ESP32
        if "temperature" in data:
            for key in sensor_data.keys():
                if key in data:
                    sensor_data[key] = data[key]
            return jsonify({"message": "✅ Dữ liệu cảm biến đã cập nhật!", "status": "success"}), 200
        
        # Nếu nhận kết quả xác thực khuôn mặt
        if "recognized" in data:
            if data["recognized"] == "true":
                face_authenticated = True
                return jsonify({"message": "✅ Khuôn mặt hợp lệ!", "status": "success"}), 200
            else:
                face_authenticated = False
                return jsonify({"message": "❌ Khuôn mặt không hợp lệ!", "status": "error"}), 403
        
        return jsonify({"message": "❌ Dữ liệu không xác định!", "status": "error"}), 400
    except Exception as e:
        return jsonify({"message": "❌ Lỗi cập nhật dữ liệu!", "error": str(e)}), 400

@app.route("/garden")
def garden():
    """Chỉ cho phép vào vườn nếu đã xác thực khuôn mặt"""
    global face_authenticated

    if not face_authenticated:
        return "🚫 Truy cập bị chặn! Bạn chưa xác thực khuôn mặt.", 403

    # Reset trạng thái sau khi vào (để không ai khác dùng lại)
    face_authenticated = False
    return redirect("http://192.168.0.113/")  # Chuyển hướng đến ESP32

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
