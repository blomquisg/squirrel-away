#include <wiringPi.h>
#include <stdio.h>
#include <stdlib.h>

//TODO decide on the actual gpio pin number
#define PIR_PIN 4

int main() {
    wiringPiSetup();
    pinMode(PIR_PIN, INPUT);

    while (1) {
        if (digitalRead(PIR_PIN) == HIGH) { // motion detected
            // activate camera image differencing
        }
        delay(500); // give the CPU a little rest
    }
    return 0;
}