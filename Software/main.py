import cv2
import mediapipe as mp
import time
import pyttsx3
import speech_recognition as sr
from gpt4all import GPT4All

# --- Modellname für 8GB RAM ---
MODEL_NAME = "orca-mini-3b-gguf2-q4_0.gguf"

# --- TTS Funktion ---
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# --- Mikrofon Aufnahme Funktion ---
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Sprich jetzt...")
        audio = r.listen(source, phrase_time_limit=5)
    try:
        text = r.recognize_google(audio, language="de-DE")
        print("Du sagtest:", text)
        return text
    except sr.UnknownValueError:
        print("Konnte dich nicht verstehen.")
        return ""
    except sr.RequestError:
        print("Spracherkennung nicht verfügbar.")
        return ""

# --- LLM laden ---
print(f"Lädt LLM-Modell: {MODEL_NAME} ...")
llm = GPT4All(MODEL_NAME)
print("LLM bereit!")

# --- Gesichtserkennung Setup ---
mp_face_detection = mp.solutions.face_detection
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.3
)

last_interaction_time = 0
interaction_interval = 5  # Sekunden zwischen Antworten

print("Starte Gesichtserkennung + Voice-Chat ...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Voice-Chat nur alle `interaction_interval` Sekunden
        current_time = time.time()
        if current_time - last_interaction_time > interaction_interval:
            prompt = listen()
            if prompt:
                # Chat session für bessere Konversation
                with llm.chat_session():
                    response = llm.generate(prompt)
                speak(response)
                last_interaction_time = current_time

    cv2.imshow("Gesichtserkennung", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC zum Beenden
        break

cap.release()
cv2.destroyAllWindows()