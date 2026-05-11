import cv2
import serial
import time
import numpy as np

SERIAL_PORT = "COM3"
BAUDRATE = 9600
CAM_INDEX = 1

FRAME_W = 640
FRAME_H = 480

DISPLAY_W = 1600
DISPLAY_H = 900

PAN_MIN, PAN_MAX = 20, 160
TILT_MIN, TILT_MAX = 20, 160

pan_angle = 90.0
tilt_angle = 90.0

PAN_SIGN = -1
TILT_SIGN = -1

DEADZONE_X = 0.08
DEADZONE_Y = 0.11

KP_X = 2.4
KP_Y = 1.45

KD_X = 0.09
KD_Y = 0.07

MAX_STEP_X = 1.25
MAX_STEP_Y = 0.65

CONTROL_INTERVAL = 0.035

CENTER_ALPHA_X = 0.24
CENTER_ALPHA_Y = 0.12

ERROR_ALPHA_X = 0.16
ERROR_ALPHA_Y = 0.08

MAX_FACE_JUMP = 100
LOST_FRAMES_BEFORE_RESET = 12
SEND_MIN_CHANGE = 1

filtered_error_x = 0.0
filtered_error_y = 0.0
prev_error_x = 0.0
prev_error_y = 0.0

smoothed_center_x = None
smoothed_center_y = None
last_face_center = None

lost_frames = 0
last_control = time.time()
last_sent_pan = None
last_sent_tilt = None

WINDOW_NAME = "Robo Prototype Face Tracking"


def clamp(value, min_value, max_value):
    return float(np.clip(value, min_value, max_value))


def pd_control(angle, error, prev_error, dt, sign, kp, kd, max_step, deadzone, min_angle, max_angle):
    if abs(error) < deadzone:
        return angle, prev_error * 0.65

    dt = max(dt, 0.001)

    d_error = (error - prev_error) / dt
    d_error = clamp(d_error, -5.0, 5.0)

    delta = sign * ((kp * error) + (kd * d_error))
    delta = clamp(delta, -max_step, max_step)

    angle = clamp(angle + delta, min_angle, max_angle)
    return angle, error


def select_best_face(faces, last_center):
    if len(faces) == 0:
        return None

    if last_center is None:
        return max(faces, key=lambda f: f[2] * f[3])

    lx, ly = last_center

    def score(f):
        x, y, w, h = f
        cx = x + w / 2
        cy = y + h / 2
        dist = (cx - lx) ** 2 + (cy - ly) ** 2
        area = w * h
        return dist - area * 0.25

    return min(faces, key=score)


def face_jump_too_large(raw_x, raw_y, last_center):
    if last_center is None:
        return False

    lx, ly = last_center
    jump = np.sqrt((raw_x - lx) ** 2 + (raw_y - ly) ** 2)
    return jump > MAX_FACE_JUMP


def send_servo_command(ser, pan, tilt):
    global last_sent_pan, last_sent_tilt

    send_pan = int(round(pan))
    send_tilt = int(round(tilt))

    pan_changed = last_sent_pan is None or abs(send_pan - last_sent_pan) >= SEND_MIN_CHANGE
    tilt_changed = last_sent_tilt is None or abs(send_tilt - last_sent_tilt) >= SEND_MIN_CHANGE

    if pan_changed or tilt_changed:
        command = f"PAN:{send_pan},TILT:{send_tilt}\n"
        ser.write(command.encode())
        last_sent_pan = send_pan
        last_sent_tilt = send_tilt


try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    time.sleep(2)
    print(f"Arduino verbunden auf {SERIAL_PORT}")
except serial.SerialException as e:
    print(f"Fehler: Arduino/Serial-Port nicht erreichbar: {e}")
    raise SystemExit


cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    ser.close()
    raise SystemExit


cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Fehler: Haarcascade konnte nicht geladen werden.")
    cap.release()
    ser.close()
    raise SystemExit

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Kein Kamerabild.")
            time.sleep(0.1)
            continue

        h, w, _ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=7,
            minSize=(80, 80)
        )

        face = select_best_face(faces, last_face_center)
        face_detected = face is not None

        if face_detected:
            x, y, width, height = face

            raw_center_x = x + width / 2
            raw_center_y = y + height / 2

            if face_jump_too_large(raw_center_x, raw_center_y, last_face_center):
                face_detected = False
                lost_frames += 1

                cv2.putText(
                    frame,
                    "Sprung ignoriert",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

            else:
                lost_frames = 0

                if smoothed_center_x is None:
                    smoothed_center_x = raw_center_x
                    smoothed_center_y = raw_center_y
                else:
                    smoothed_center_x = (1 - CENTER_ALPHA_X) * smoothed_center_x + CENTER_ALPHA_X * raw_center_x
                    smoothed_center_y = (1 - CENTER_ALPHA_Y) * smoothed_center_y + CENTER_ALPHA_Y * raw_center_y

                center_x = int(smoothed_center_x)
                center_y = int(smoothed_center_y)

                last_face_center = (center_x, center_y)

                raw_error_x = (center_x - w / 2) / (w / 2)
                raw_error_y = (center_y - h / 2) / (h / 2)

                filtered_error_x = (1 - ERROR_ALPHA_X) * filtered_error_x + ERROR_ALPHA_X * raw_error_x
                filtered_error_y = (1 - ERROR_ALPHA_Y) * filtered_error_y + ERROR_ALPHA_Y * raw_error_y

                now = time.time()

                if now - last_control >= CONTROL_INTERVAL:
                    dt = now - last_control

                    pan_angle, prev_error_x = pd_control(
                        pan_angle,
                        filtered_error_x,
                        prev_error_x,
                        dt,
                        PAN_SIGN,
                        KP_X,
                        KD_X,
                        MAX_STEP_X,
                        DEADZONE_X,
                        PAN_MIN,
                        PAN_MAX
                    )

                    tilt_angle, prev_error_y = pd_control(
                        tilt_angle,
                        filtered_error_y,
                        prev_error_y,
                        dt,
                        TILT_SIGN,
                        KP_Y,
                        KD_Y,
                        MAX_STEP_Y,
                        DEADZONE_Y,
                        TILT_MIN,
                        TILT_MAX
                    )

                    send_servo_command(ser, pan_angle, tilt_angle)
                    last_control = now

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        if not face_detected:
            filtered_error_x *= 0.85
            filtered_error_y *= 0.85
            prev_error_x *= 0.7
            prev_error_y *= 0.7

            if lost_frames >= LOST_FRAMES_BEFORE_RESET:
                last_face_center = None
                smoothed_center_x = None
                smoothed_center_y = None
                lost_frames = 0

            cv2.putText(
                frame,
                "Kein stabiles Gesicht - keine Bewegung",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        cv2.circle(frame, (w // 2, h // 2), 5, (255, 0, 0), -1)
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (255, 0, 0), 1)

        cv2.putText(
            frame,
            f"PAN: {int(round(pan_angle))} | TILT: {int(round(tilt_angle))}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"errX: {filtered_error_x:.2f} | errY: {filtered_error_y:.2f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "ESC=Ende | R=Reset",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == ord("r"):
            pan_angle = 90.0
            tilt_angle = 90.0

            filtered_error_x = 0.0
            filtered_error_y = 0.0
            prev_error_x = 0.0
            prev_error_y = 0.0

            last_face_center = None
            smoothed_center_x = None
            smoothed_center_y = None
            lost_frames = 0

            send_servo_command(ser, pan_angle, tilt_angle)
            print("Reset auf 90/90")

finally:
    cap.release()
    cv2.destroyAllWindows()
    ser.close()