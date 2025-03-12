import sounddevice as sd
import numpy as np
import wave
import os
import keyboard
import librosa
import tensorflow as tf

# 🔹 Load mô hình CNN đã train
model_path = "voice_command_model.h5"
if not os.path.exists(model_path):
    print("❌ Không tìm thấy voice_command_model.h5! Hãy huấn luyện mô hình trước.")
    exit()

model = tf.keras.models.load_model(model_path)

# 🔹 Danh sách lệnh - ĐẢM BẢO ĐÚNG THỨ TỰ NHƯ LÚC TRAIN
labels = ["bat_den", "tat_den"]

# 🔹 Tham số xử lý âm thanh (phải khớp với lúc train)
n_mfcc = 64  # 🔹 Đảm bảo dùng đúng số lượng MFCC khi train
max_length = 100  # 🔹 Độ dài tối đa của mỗi mẫu MFCC

# 🔹 Hàm thu âm từ microphone
def record_audio(filename="test.wav", samplerate=16000):
    print("🎤 Đang thu âm... (Nhấn phím '1' để dừng)")

    audio_data = []
    stream = sd.InputStream(samplerate=samplerate, channels=1, dtype=np.int16)
    stream.start()

    while True:
        frame, overflowed = stream.read(1024)
        audio_data.append(frame)

        if keyboard.is_pressed("1"):  # Dừng khi nhấn phím "1"
            print("🛑 Dừng ghi âm!")
            break

    stream.stop()
    stream.close()

    # Gộp dữ liệu lại
    audio_data = np.concatenate(audio_data, axis=0)

    # Lưu file WAV
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    print(f"✅ Ghi âm hoàn tất: {filename}")

# 🔹 Hàm trích xuất đặc trưng MFCC từ file âm thanh
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # 🔹 Sử dụng 64 MFCC

    # 🔹 Đảm bảo dữ liệu có đúng số khung thời gian
    if mfcc.shape[1] < max_length:
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]

    return mfcc.reshape(1, n_mfcc, max_length, 1)  # 🔹 Định dạng đúng với model

# 🛠 Ghi âm & trích xuất đặc trưng
record_audio("test.wav")
features = extract_mfcc("test.wav")

# 🛠 Kiểm tra dữ liệu đầu vào trước khi dự đoán
print(f"📊 Kích thước đầu vào của model: {features.shape}")

# 🛠 Dự đoán bằng mô hình CNN
predictions = model.predict(features)
predicted_label_idx = np.argmax(predictions)
predicted_label = labels[predicted_label_idx]

# 🛠 Hiển thị xác suất dự đoán từng nhãn
for i, label in enumerate(labels):
    print(f"🔮 {label}: {predictions[0][i] * 100:.2f}%")

# 🛠 Hiển thị kết quả cuối cùng
print(f"✅ Kết quả dự đoán: {predicted_label.upper()}")