import cv2
import mediapipe as mp

# --- MediaPipe Gesichtserkennung initialisieren ---
mp_face_detection = mp.solutions.face_detection

# --- Webcam starten ---
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_face_detection.FaceDetection(
    model_selection=0.5,         # 0: Nahaufnahme, 1: größere Entfernung
    min_detection_confidence=0.3
) as face_detection:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Bildgröße ermitteln
        h, w, _ = frame.shape

        # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Gesicht erkennen
        results = face_detection.process(rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Rechteck um das Gesicht zeichnen
                cv2.rectangle(frame, (x, y), (x + width, y + height),
                              (0, 255, 0), 2)

        # Bild anzeigen
        cv2.imshow("Gesichtserkennung", frame)

        # ESC zum Beenden
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Aufräumen
cap.release()
cv2.destroyAllWindows()