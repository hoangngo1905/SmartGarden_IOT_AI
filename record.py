import speech_recognition as sr
import pyaudio
import wave
import os
import threading
import time
import serial
from datetime import datetime
import serial.tools.list_ports

# Liệt kê các cổng Serial
ports = serial.tools.list_ports.comports()
for port in ports:
    print(f"Tìm thấy cổng: {port.device}")

class VoiceCommandWateringSystem:
    def __init__(self, port="COM4", baud_rate=9600):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False

        # Khởi tạo Serial
        try:
            self.arduino = serial.Serial(port, baud_rate, timeout=1)
            print(f"✅ Đã kết nối với Arduino trên {port}")
            time.sleep(2)  
        except Exception as e:
            print(f"⚠️ Không thể kết nối với Arduino: {e}")
            self.arduino = None

        # Định nghĩa lệnh
        self.commands = {
            "bật bơm": self.turn_pump_on,
            "mở bơm": self.turn_pump_on,
            "tắt bơm": self.turn_pump_off,
            "ngừng bơm": self.turn_pump_off,
            "dừng lại": self.stop_listening
        }

        # Điều chỉnh tiếng ồn
        with self.microphone as source:
            print("🎤 Hiệu chỉnh tiếng ồn... Giữ im lặng!")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("✅ Sẵn sàng nhận lệnh!")

    def turn_pump_on(self):
        print("🟢 BẬT BƠM!")
        if self.arduino:
            self.arduino.write(b'ON\n')
            print("➡️ Đã gửi lệnh BẬT đến Arduino")

    def turn_pump_off(self):
        print("🔴 TẮT BƠM!")
        if self.arduino:
            self.arduino.write(b'OFF\n')
            print("➡️ Đã gửi lệnh TẮT đến Arduino")

    def stop_listening(self):
        print("⏹️ Dừng nghe lệnh.")
        self.is_listening = False

    def process_audio(self, audio_data):
        try:
            text = self.recognizer.recognize_google(audio_data, language="vi-VN")
            print(f"🎙️ Đã nghe thấy: {text}")

            recognized = False
            for command, action in self.commands.items():
                if command in text.lower():
                    action()
                    recognized = True
                    break

            if not recognized:
                print("⚠️ Không hiểu lệnh. Hãy thử lại!")
            return recognized
        except sr.UnknownValueError:
            print("⚠️ Không nhận diện được lời nói.")
        except sr.RequestError as e:
            print(f"⚠️ Lỗi Google Speech Recognition: {e}")
        return False

    def listen_continuously(self):
        self.is_listening = True
        print("🎧 Đang nghe lệnh... (Nói 'dừng lại' để thoát)")
        
        while self.is_listening:
            with self.microphone as source:
                try:
                    print("👂 Đang lắng nghe...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    self.process_audio(audio)
                except sr.WaitTimeoutError:
                    print("⚠️ Không nghe thấy gì, tiếp tục lắng nghe...")
                except Exception as e:
                    print(f"⚠️ Lỗi: {e}")

    def record_audio_file(self, duration=5, filename=None):
        if filename is None:
            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024

        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        
        print(f"🎙️ Ghi âm trong {duration} giây...")
        frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * duration))]

        stream.stop_stream()
        stream.close()
        audio.terminate()

        waveFile = wave.open(filename, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        print(f"💾 File lưu tại: {filename}")
        return filename

    def recognize_from_file(self, filename):
        print(f"🔍 Đang phân tích file {filename}...")
        with sr.AudioFile(filename) as source:
            audio = self.recognizer.record(source)
            return self.process_audio(audio)

    def __del__(self):
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("🔌 Đã đóng kết nối Arduino.")

def main():
    port = input("Nhập cổng COM (ví dụ: COM4): ") or "COM4"
    system = VoiceCommandWateringSystem(port=port)

    print("🔵 Chọn chế độ:")
    print("1️⃣ Nghe trực tiếp liên tục")
    print("2️⃣ Ghi âm và phân tích")

    choice = input("Nhập lựa chọn (1 hoặc 2): ")

    if choice == "1":
        system.listen_continuously()
    elif choice == "2":
        duration = int(input("⏳ Nhập thời gian ghi âm (giây): "))
        filename = system.record_audio_file(duration)
        system.recognize_from_file(filename)
    else:
        print("⚠️ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()
