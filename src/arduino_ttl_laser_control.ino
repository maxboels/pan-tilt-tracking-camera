#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Initialize the PCA9685 driver at the default I2C address
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// --- Define your PCA9685 Channels ---
#define SERVO_CHANNEL 0
#define LASER_CHANNEL 2 // Your laser is plugged into Channel 2

void setup() {
  Serial.begin(115200);
  Serial.println("PCA9685 Laser Test");

  // Initialize the PCA9685
  pwm.begin();
  
  // Set the PWM frequency to 60Hz, which is typical for servos
  pwm.setPWMFreq(60);

  // Ensure the laser starts in the OFF state
  // Setting the duty cycle to 0%
  pwm.setPWM(LASER_CHANNEL, 0, 0); 
  Serial.println("Laser should be OFF.");

  delay(1000); // Wait a second before starting
}

void loop() {
  // --- Turn the Laser ON ---
  Serial.println("Turning laser ON");
  // Set the duty cycle to 100% (4095 is the max value for the 12-bit controller)
  // This sends a constant HIGH signal to the laser's TTL pin
  pwm.setPWM(LASER_CHANNEL, 0, 4095);
  
  delay(2000); // Keep laser on for 2 seconds

  // --- Turn the Laser OFF ---
  Serial.println("Turning laser OFF");
  // Set the duty cycle back to 0%
  // This sends a constant LOW signal
  pwm.setPWM(LASER_CHANNEL, 0, 0);
  
  delay(2000); // Keep laser off for 2 seconds
}