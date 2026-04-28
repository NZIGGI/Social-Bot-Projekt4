import cv2
import mediapipe as mp
import serial
import time
import numpy as np

# --- MediaPipe Gesichtserkennung initialisieren ---
mp_face_detection = mp.solutions.face_detection

# --- Serielle Verbindung zum Arduino (COM-Port anpassen, z.B. '/dev/ttyACM0' unter Linux) ---
ser = serial.Serial('COM5', 9600, timeout=1)  # Windows: 'COM3', Linux/Mac: '/dev/ttyUSB0' oder '/dev/ttyACM0'
time.sleep(2)  # Warten bis Arduino bereit

# --- Webcam starten ---
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_face_detection.FaceDetection(
    model_selection=1,  # 0: Nahaufnahme, 1: größere Entfernung
    min_detection_confidence=0.3
) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        pan_angle = 90  # Neutral: Mitte (0-180)
        tilt_angle = 90

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Gesichtzentrum berechnen
                center_x = x + width // 2
                center_y = y + height // 2

                # Pan (horizontal): links/rechts -> Servo D5
                error_x = (center_x - w // 2) / (w // 2)  # Normalisiert -1 bis +1
                pan_angle = int(90 + error_x * 45)  # Max 45° Abweichung, anpassen bei Bedarf
                pan_angle = np.clip(pan_angle, 20, 160)  # Servo-Limits

                # Tilt (vertikal): oben/unten -> Servo D6
                error_y = (center_y - h // 2) / (h // 2)  # Normalisiert -1 bis +1 (y invertiert?)
                tilt_angle = int(90 - error_y * 45)  # Minus für natürliche Bewegung
                tilt_angle = np.clip(tilt_angle, 20, 160)

                # Rechteck und Zentrum zeichnen
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Winkel an Arduino senden: "PAN:90,TILT:90\n"
                command = f"PAN:{pan_angle},TILT:{tilt_angle}\n"
                ser.write(command.encode())

        cv2.imshow("Gesichtserkennung mit Servo-Steuerung", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
ser.close()