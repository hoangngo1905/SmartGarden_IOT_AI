#define SOIL_SENSOR A0  // Cảm biến độ ẩm đất
#define RELAY_PIN 4     // Chân điều khiển máy bơm

void setup() {
    Serial.begin(9600);  // UART với ESP32
    pinMode(SOIL_SENSOR, INPUT);
    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, LOW);  // Đảm bảo máy bơm tắt ban đầu
}

void loop() {
    int soilMoisture = analogRead(SOIL_SENSOR);  // Đọc giá trị độ ẩm đất
    Serial.println(soilMoisture);  // Gửi dữ liệu đến ESP32
    Serial.flush();  // Đảm bảo dữ liệu được gửi đi hoàn toàn

    // Debug giá trị độ ẩm đất
    Serial.print("Độ ẩm đất: ");
    Serial.println(soilMoisture);

    // Điều khiển relay dựa vào độ ẩm đất
    if (soilMoisture < 700
    ) {
        digitalWrite(RELAY_PIN, HIGH);  // Bật máy bơm
        Serial.println("Máy bơm: BẬT");
    } else {
        digitalWrite(RELAY_PIN, LOW);   // Tắt máy bơm
        Serial.println("Máy bơm: TẮT");
    }

    delay(5000);  // Đọc dữ liệu mỗi 2 giây
}