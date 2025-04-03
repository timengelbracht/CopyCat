#include "HX711.h"

#define DOUT 4  
#define SCK 5   
#define SCALE_FACTOR 10000  // Replace with your calibration value
#define FORCE_CUTOFF 5.0  // Maximum force allowed (kg)
#define FILTER_SIZE 10  // Number of samples for smoothing

HX711 scale;
float forceReadings[FILTER_SIZE];  // Array to store force values
int filterIndex = 0;

void setup() {
    Serial.begin(115200);
    scale.begin(DOUT, SCK);
    
    Serial.println("Taring... Remove all weight.");
    delay(3000);
    scale.tare();  // Set zero reference
    Serial.println("Tare complete. Ready!");

    // Initialize filter array
    for (int i = 0; i < FILTER_SIZE; i++) {
        forceReadings[i] = 0.0;
    }
}

void loop() {
    float rawForce = scale.get_units() / SCALE_FACTOR;  // Convert raw data to kg

    // Apply cutoff limit
    if (rawForce > FORCE_CUTOFF) {
        rawForce = FORCE_CUTOFF;
    } else if (rawForce < 0) {
        rawForce = 0;  // Ensure no negative values
    }

    // Add new value to filter array
    forceReadings[filterIndex] = rawForce;
    filterIndex = (filterIndex + 1) % FILTER_SIZE;  // Circular buffer

    // Compute moving average
    float filteredForce = 0;
    for (int i = 0; i < FILTER_SIZE; i++) {
        filteredForce += forceReadings[i];
    }
    filteredForce /= FILTER_SIZE;  // Average of last N values

    Serial.print("Filtered Force (kg): ");
    Serial.println(filteredForce, 3);  // Print with 3 decimal places
    
    delay(50);  // Small delay for stability
}
