import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import random

# 🔹 Thông số đặc trưng âm thanh
n_mfcc = 64   # Số lượng MFCC
max_length = 100  # Độ dài tối đa của MFCC
sampling_rate = 16000  # Tần số lấy mẫu

# 🔹 Danh sách nhãn
labels = ["bat_den", "tat_den", "bat_bom", "tat_bom"]
num_classes = len(labels)

# 🔹 Đường dẫn dữ liệu
dataset_path = "datasetrecord/"
if not os.path.exists(dataset_path):
    print("❌ Không tìm thấy thư mục dataset! Hãy thu âm trước.")
    exit()

# 🔹 Hàm trích xuất đặc trưng MFCC
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=sampling_rate)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Xử lý độ dài MFCC
    if mfcc.shape[1] < max_length:
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]

    return mfcc

# 🔹 Đọc dữ liệu và nhãn
X, y = [], []

for label_idx, label in enumerate(labels):
    label_path = os.path.join(dataset_path, label)
    if not os.path.exists(label_path):
        print(f"⚠ Không tìm thấy thư mục {label_path}, bỏ qua...")
        continue

    for file in os.listdir(label_path):
        file_path = os.path.join(label_path, file)
        if file_path.endswith(".wav"):
            features = extract_features(file_path)
            X.append(features)
            y.append(label_idx)

# Chuyển đổi sang numpy array
X = np.array(X).reshape(-1, n_mfcc, max_length, 1)
y = np.array(y)

print(f"📊 Tổng số mẫu: {len(y)}")
for i, label in enumerate(labels):
    print(f"📊 {label}: {np.sum(y == i)} mẫu")

# Chia tập dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(n_mfcc, max_length, 1)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# 🔹 Biên dịch mô hình
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 🔹 Huấn luyện mô hình
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# 🔹 Đánh giá mô hình
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Độ chính xác trên tập test: {test_acc:.2%}")

# 🔹 Lưu mô hình
model.save("voice_command_model.h5")
print("💾 Đã lưu model vào voice_command_model.h5")

# 🔹 Vẽ biểu đồ loss & accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.show()