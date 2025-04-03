#include <HX711.h>
#include "HX711.h"

// Pin definitions
#define DOUT 4  // Data pin (DOUT = DT equivalent)
#define SCK 5   // Clock pin

HX711 scale;

void setup() {
    Serial.begin(9600);
    scale.begin(DOUT, SCK);
    scale.set_scale();  // Set scale factor (you need to calibrate this)
    scale.tare();       // Reset scale to zero
}

void loop() {
    Serial.print("Weight: ");
    Serial.println(scale.get_units(), 2); // Read weight value
    delay(500);
}
