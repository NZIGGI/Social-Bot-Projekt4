#include <Servo.h>

Servo panServo;  // MG995 horizontal (Pan) auf D5
Servo tiltServo; // ES08MD vertikal (Tilt) auf D6

int panPos = 90;
int tiltPos = 90;

void setup() {
  Serial.begin(9600);
  panServo.attach(5);   // D5
  tiltServo.attach(6);  // D6
  panServo.write(90);
  tiltServo.write(90);
  delay(1000);
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    // Parse "PAN:90,TILT:90"
    int panIndex = command.indexOf("PAN:");
    int commaIndex = command.indexOf(",");
    int tiltIndex = command.indexOf("TILT:");
    
    if (panIndex >= 0 && commaIndex > panIndex) {
      int panEnd = commaIndex;
      String panStr = command.substring(panIndex + 4, panEnd);
      panPos = panStr.toInt();
      panPos = constrain(panPos, 20, 160);
      panServo.write(panPos);
    }
    
    if (tiltIndex >= 0) {
      String tiltStr = command.substring(tiltIndex + 5);
      tiltPos = tiltStr.toInt();
      tiltPos = constrain(tiltPos, 20, 160);
      tiltServo.write(tiltPos);
    }
  }
}
