#include <Servo.h>

Servo panServo;
Servo tiltServo;

const byte PAN_PIN  = 5;
const byte TILT_PIN = 6;

float panPos  = 90.0;
float tiltPos = 90.0;

// Aus deiner Sprungantwort
const float Kpan_px_per_deg  = 16.8;
const float Ktilt_px_per_deg = 17.4;

// Dämpfung: kleiner = ruhiger, größer = schneller
const float alphaPan  = 0.25;
const float alphaTilt = 0.20;

// Falls Achse falsch herum läuft: Vorzeichen ändern
const float PAN_SIGN  = -1.0;
const float TILT_SIGN = -1.0;

// Python sendet normierte Fehler
const float halfWidth  = 640.0;   // 1280 / 2
const float halfHeight = 360.0;   // 720 / 2

String rxLine = "";

void setup() {
  Serial.begin(115200);

  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);

  panServo.write((int)panPos);
  tiltServo.write((int)tiltPos);

  delay(1000);
}

void processLine(String line) {
  int panStart  = line.indexOf("PAN:");
  int tiltStart = line.indexOf("TILT:");

  if (panStart < 0 || tiltStart < 0) return;

  int comma = line.indexOf(",", panStart);

  String panStr  = line.substring(panStart + 4, comma);
  String tiltStr = line.substring(tiltStart + 5);

  float panErrNorm  = panStr.toFloat();
  float tiltErrNorm = tiltStr.toFloat();

  // Normierten Fehler zurück in Pixel
  float panErr_px  = panErrNorm  * halfWidth;
  float tiltErr_px = tiltErrNorm * halfHeight;

  // Kleine Totzone gegen Zittern
  if (abs(panErr_px) < 5.0)  panErr_px = 0.0;
  if (abs(tiltErr_px) < 5.0) tiltErr_px = 0.0;

  // Direkte Positionskorrektur über gemessene Strecke px/deg
  float panDelta_deg  = PAN_SIGN  * alphaPan  * panErr_px  / Kpan_px_per_deg;
  float tiltDelta_deg = TILT_SIGN * alphaTilt * tiltErr_px / Ktilt_px_per_deg;

  panPos  += panDelta_deg;
  tiltPos += tiltDelta_deg;

  panPos  = constrain(panPos, 20, 160);
  tiltPos = constrain(tiltPos, 45, 135);

  panServo.write((int)panPos);
  tiltServo.write((int)tiltPos);

  Serial.print("PAN=");
  Serial.print(panPos);
  Serial.print(", TILT=");
  Serial.println(tiltPos);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n') {
      processLine(rxLine);
      rxLine = "";
    } 
    else if (c != '\r') {
      rxLine += c;
    }
  }
}