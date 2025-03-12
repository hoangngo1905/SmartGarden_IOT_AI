import sounddevice as sd
import numpy as np
import wave
import whisper
import os
import librosa
import soundfile as sf
import simpleaudio as sa

# 🔹 Thư mục lưu dữ liệu
DATA_DIR = "datarecord"
os.makedirs(DATA_DIR, exist_ok=True)

# 🔹 Thông số ghi âm
DURATION = 5  # Giây
SAMPLERATE = 16000  # Whisper hoạt động tốt với 16kHz

# 🔹 Tìm số thứ tự file tiếp theo
def get_next_filename():
    files = [f for f in os.listdir(DATA_DIR) if f.startswith("record_") and f.endswith(".wav")]
    numbers = [int(f.split("_")[1].split(".")[0]) for f in files if f.split("_")[1].split(".")[0].isdigit()]
    next_num = max(numbers) + 1 if numbers else 1
    return os.path.join(DATA_DIR, f"record_{next_num}.wav"), next_num

# 🔹 Ghi âm & lưu file WAV
def record_audio():
    filename, record_num = get_next_filename()
    print(f"🎤 Đang thu âm... (Lưu vào {filename})")

    audio_data = sd.rec(int(SAMPLERATE * DURATION), samplerate=SAMPLERATE, channels=1, dtype=np.int16)
    sd.wait()  # Chờ ghi âm xong

    # Lưu file WAV
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLERATE)
        wf.writeframes(audio_data.tobytes())

    print(f"✅ Ghi âm hoàn tất: {filename}")

    # Phát lại âm thanh để kiểm tra
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()

    return filename, record_num

# 🔹 Tiền xử lý âm thanh (lọc nhiễu, chuẩn hóa âm lượng)
def preprocess_audio(filename):
    print("🔄 Đang tiền xử lý âm thanh...")
    y, sr = librosa.load(filename, sr=16000)
    y = librosa.effects.preemphasis(y)  # Tăng độ rõ của âm thanh
    y = librosa.util.normalize(y)  # Chuẩn hóa âm lượng
    sf.write(filename, y, sr)

# 🔹 Nhận diện giọng nói bằng Whisper
def transcribe_audio(filename, record_num):
    print(f"🛠 Đang nhận diện giọng nói từ {filename}...")
    model = whisper.load_model("medium")  # Dùng mô hình lớn hơn
    result = model.transcribe(filename, language="vi")  # Ép nhận diện tiếng Việt
    transcript = result["text"]

    # Lưu văn bản vào file transcripts.txt
    transcript_file = os.path.join(DATA_DIR, "transcripts.txt")
    with open(transcript_file, "a", encoding="utf-8") as f:
        f.write(f"Record {record_num}: {transcript}\n")

    print(f"📝 Nhận diện hoàn tất: {transcript}")
    return transcript

# 🛠 Chạy chương trình
filename, record_num = record_audio()  
preprocess_audio(filename)  # Tiền xử lý âm thanh trước khi nhận diện
transcribe_audio(filename, record_num)
