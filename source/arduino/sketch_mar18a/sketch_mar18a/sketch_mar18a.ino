#include "HX711.h"

#define DOUT 4  // HX711 Data
#define SCK 5   // HX711 Clock

HX711 scale;

void setup() {
    Serial.begin(115200); // Use a higher baud rate for better resolution
    scale.begin(DOUT, SCK);
    Serial.println("HX711 Reading...");
}

void loop() {
    if (scale.is_ready()) {
        long reading = scale.read();
        Serial.print("Raw reading: ");
        Serial.println(reading);
    } else {
        Serial.println("HX711 not ready...");
    }
    delay(500);
}
