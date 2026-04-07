import cv2
import mediapipe as mp
import time
import pyttsx3
import speech_recognition as sr
import requests

# --- TTS ---
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# --- Mikrofon ---
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Sprich jetzt...")
        audio = r.listen(source, phrase_time_limit=5)
    try:
        text = r.recognize_google(audio, language="de-DE")
        print("Du sagtest:", text)
        return text
    except:
        return ""

# --- Kostenloses Online LLM (HuggingFace) ---
def ask_llm(prompt):
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-base",
            json={"inputs": prompt}
        )
        data = response.json()
        return data[0]["generated_text"]
    except:
        return "Ich konnte gerade nicht antworten."

# --- Gesichtserkennung ---
mp_face_detection = mp.solutions.face_detection
cap = cv2.VideoCapture(1)

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.3
)

last_interaction_time = 0
interaction_interval = 6  # etwas höher wegen Online-Latenz

print("System gestartet (kostenloser Online Voice-Chat)...")

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

            cv2.rectangle(frame, (x, y), (x + width, y + height),
                          (0, 255, 0), 2)

        # Voice-Chat nur alle paar Sekunden
        if time.time() - last_interaction_time > interaction_interval:
            prompt = listen()

            if prompt:
                response = ask_llm(prompt)
                print("LLM:", response)
                speak(response)

                last_interaction_time = time.time()

    cv2.imshow("Gesichtserkennung", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()