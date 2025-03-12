import os
import io
import sounddevice as sd
import numpy as np
from google.cloud import speech

# Cấu hình biến môi trường (có thể bỏ nếu đã thiết lập trước)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your-key.json"

def record_audio(duration=5, samplerate=16000):
    print("🎤 Đang ghi âm... Nói gì đó!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    print("✅ Ghi âm hoàn tất!")
    return audio_data

def speech_to_text(audio_data, samplerate=16000):
    client = speech.SpeechClient()
    
    audio = speech.RecognitionAudio(content=audio_data.tobytes())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=samplerate,
        language_code="vi-VN"  # Nhận diện tiếng Việt
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        print("📢 Văn bản nhận diện:", result.alternatives[0].transcript)

# Chạy chương trình
audio_data = record_audio(duration=5)
speech_to_text(audio_data)
