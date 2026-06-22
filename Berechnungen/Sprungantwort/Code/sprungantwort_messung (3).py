"""
============================================================
 Sprungantwort-Messumgebung  Pan/Tilt-Kamera-System
============================================================

Zweck
-----
Misst die OFFENE Streckenantwort (Servo + Kamera + Bildverarbeitung)
auf einen Winkelsprung. Aus den Daten legst du in MATLAB per
Loopshaping deinen Regler fuer die Gesichtsverfolgung aus.

WICHTIG: Der Arduino laeuft hier OHNE PID (arduino_step_measure.ino).
Es werden direkte Winkel gesendet, KEINE Fehlerwerte.

Bedien-Ablauf (genau so wie gewuenscht)
---------------------------------------
1. Parameter einstellen (Achse, Sprunghoehe, Anzahl Spruenge, ...).
2. "1) Vorschau starten":
     - Arduino verbindet, Kamera oeffnet, Servo faehrt auf Startwinkel.
     - Live-Bild mit Fadenkreuz + erkanntem Blob (farbiger Punkt).
     - Du richtest den schwarzen Kreis aus, bis er stabil + zentriert ist.
     - Ampel: ROT (kein Ziel) / GELB (noch nicht ruhig/zentriert) /
       GRUEN (bereit). Erst bei GRUEN ist Messen erlaubt.
3. "2) Messung starten":
     - Erst JETZT laufen die Spruenge los und werden geloggt.
4. "CSV speichern" -> MATLAB-fertige Datei.
   Auswertung (px/Grad, norm/Grad, T, Td) erscheint im Interface.

Abhaengigkeiten
---------------
    pip install opencv-python numpy pyserial
    (optional fuer Plot-Knopf:)       pip install matplotlib
    (optional fuer MediaPipe-Modus:)  pip install mediapipe

CSV-Spalten
-----------
    run_id, step_id, axis, step_deg, target_angle_deg,
    t_sec, err_px, err_norm, phase
============================================================
"""

import time
import csv
import os
import threading
import queue
from dataclasses import dataclass

import numpy as np
import cv2

import serial
import serial.tools.list_ports

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


# ----------------------------------------------------------
# Konfiguration / Defaults
# ----------------------------------------------------------
DEFAULT_SERIAL_BAUD = 9600
DEFAULT_CAMERA_INDEX = 0
# Niedrigere Aufloesung -> deutlich mehr fps. 640x480 schafft mit den
# meisten Webcams 30 fps, was fuer eine Servo-Sprungantwort wichtig ist.
# 1280x720 faellt bei vielen Cams auf ~10 fps und untersampelt den Sprung!
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30

PAN_MIN, PAN_MAX = 20, 160
TILT_MIN, TILT_MAX = 45, 135

DEFAULT_SETTLE_RECORD_S = 1.2
DEFAULT_PRE_RECORD_S = 0.3

STABILITY_PX = 4.0
CENTER_TOL_PX = 60.0

CAMERA_BACKEND = cv2.CAP_DSHOW   # Windows; bei Linux ggf. cv2.CAP_ANY


# ----------------------------------------------------------
# Datencontainer
# ----------------------------------------------------------
@dataclass
class Sample:
    run_id: int
    step_id: int
    axis: str
    step_deg: float
    target_angle: float
    t_sec: float
    err_px: float
    err_norm: float
    phase: str


@dataclass
class MeasureConfig:
    serial_port: str = "COM4"
    serial_baud: int = DEFAULT_SERIAL_BAUD
    camera_index: int = DEFAULT_CAMERA_INDEX
    axis: str = "PAN"
    step_deg: float = 5.0
    start_angle: float = 90.0
    n_steps: int = 6
    bidirectional: bool = True
    pre_record_s: float = DEFAULT_PRE_RECORD_S
    settle_record_s: float = DEFAULT_SETTLE_RECORD_S
    tracker: str = "BLOB"
    show_window: bool = True
    blob_min_area: int = 200
    blob_thresh_mode: str = "AUTO"   # AUTO oder FIXED
    blob_thresh_val: int = 90
    blob_dark_target: bool = True
    blob_min_circ: float = 0.6
    show_mask: bool = False
    cam_width: int = DEFAULT_CAMERA_WIDTH
    cam_height: int = DEFAULT_CAMERA_HEIGHT
    cam_fps: int = DEFAULT_CAMERA_FPS


# ----------------------------------------------------------
# Tracker
# ----------------------------------------------------------
class BlobTracker:
    """Trackt einen runden Blob. Standard: dunkler Kreis auf hellem Grund.

    Verbesserungen gegenueber reinem Otsu:
      - Schwelle waehlbar: AUTO (Otsu), oder fester Wert (zuverlaessiger,
        wenn die Beleuchtung mal schwankt).
      - Polaritaet waehlbar: dunkles Ziel auf hell ODER hell auf dunkel.
      - Rundheits-Filter: nimmt nur Konturen, die wirklich kreisfoermig
        sind. Das verhindert, dass ein Kabel/Schatten/Aermel faelschlich
        als Ziel erkannt wird.
      - Debug-Maske: das Binaerbild kann angezeigt werden, damit man
        SIEHT, was die Kamera als Ziel haelt.
    """
    name = "BLOB"

    def __init__(self, min_area=200, thresh_mode="AUTO", thresh_val=90,
                 dark_target=True, min_circularity=0.6):
        self.min_area = min_area
        self.thresh_mode = thresh_mode      # "AUTO" oder "FIXED"
        self.thresh_val = int(thresh_val)
        self.dark_target = bool(dark_target)
        self.min_circularity = float(min_circularity)
        self.last_area = 0.0
        self.last_mask = None               # fuers Debug-Fenster

    def find(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # dunkles Ziel -> BINARY_INV (dunkel wird weiss in der Maske)
        if self.dark_target:
            inv = cv2.THRESH_BINARY_INV
        else:
            inv = cv2.THRESH_BINARY

        if self.thresh_mode == "FIXED":
            _, th = cv2.threshold(gray, self.thresh_val, 255, inv)
        else:
            _, th = cv2.threshold(gray, 0, 255, inv | cv2.THRESH_OTSU)

        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        self.last_mask = th

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.last_area = 0.0
            return None

        # Bestes rundes Objekt ueber Mindestflaeche suchen
        best = None
        best_area = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            # Rundheit: 1.0 = perfekter Kreis
            circ = 4.0 * np.pi * area / (perim * perim)
            if circ < self.min_circularity:
                continue
            if area > best_area:
                best_area = area
                best = c

        if best is None:
            self.last_area = 0.0
            return None

        self.last_area = best_area
        M = cv2.moments(best)
        if M["m00"] == 0:
            return None
        return (M["m10"] / M["m00"], M["m01"] / M["m00"])


class MediaPipeTracker:
    name = "MEDIAPIPE"

    def __init__(self, confidence=0.45):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("mediapipe ist nicht installiert.")
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=confidence)
        self.last_area = 0.0

    def find(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)
        if not res.detections:
            return None
        bb = res.detections[0].location_data.relative_bounding_box
        return ((bb.xmin + bb.width / 2.0) * w,
                (bb.ymin + bb.height / 2.0) * h)


# ----------------------------------------------------------
# Arduino
# ----------------------------------------------------------
class ArduinoLink:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.2)
        time.sleep(2.0)
        self.ser.reset_input_buffer()

    def ping(self):
        self.ser.write(b"PING\n")
        t0 = time.time()
        while time.time() - t0 < 1.5:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line == "PONG":
                return True
        return False

    def set_angle(self, axis, angle):
        self.ser.write(f"{axis}:{int(round(angle))}\n".encode("utf-8"))

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


# ----------------------------------------------------------
# Mess-Engine: 2 Phasen (Vorschau -> Messung)
# ----------------------------------------------------------
class MeasureEngine:
    def __init__(self, cfg: MeasureConfig, status_q: queue.Queue):
        self.cfg = cfg
        self.q = status_q
        self.samples: list[Sample] = []
        self.analysis = {}

        self._stop = threading.Event()
        self._start_measure = threading.Event()
        self._abort_measure = threading.Event()

        self.thread = None
        self.ready = False
        self.measured_fps = 0.0
        self._ard = None
        self._cap = None
        self._tracker = None

    def log(self, m): self.q.put(("log", m))
    def status(self, m): self.q.put(("status", m))
    def ampel(self, s): self.q.put(("ampel", s))

    def start_preview(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def trigger_measurement(self):
        self._start_measure.set()

    def abort_measurement(self):
        self._abort_measure.set()

    def stop_all(self):
        self._stop.set()

    def _make_tracker(self):
        if self.cfg.tracker == "MEDIAPIPE":
            return MediaPipeTracker()
        return BlobTracker(
            min_area=self.cfg.blob_min_area,
            thresh_mode=self.cfg.blob_thresh_mode,
            thresh_val=self.cfg.blob_thresh_val,
            dark_target=self.cfg.blob_dark_target,
            min_circularity=self.cfg.blob_min_circ)

    def _err_along_axis(self, cx, cy, w, h):
        if self.cfg.axis == "PAN":
            e = cx - w / 2.0
            return e, e / (w / 2.0)
        e = cy - h / 2.0
        return e, e / (h / 2.0)

    def _clamp(self, axis, ang):
        if axis == "PAN":
            return max(PAN_MIN, min(PAN_MAX, ang))
        return max(TILT_MIN, min(TILT_MAX, ang))

    def _measure_fps(self, n=30):
        """Misst die echte Framerate, indem n Frames gegrabbt werden.
        Warnt, wenn die fps fuer eine Sprungantwort zu niedrig sind."""
        ok = 0
        t0 = time.perf_counter()
        for _ in range(n):
            ret, _ = self._cap.read()
            if ret:
                ok += 1
        dt = time.perf_counter() - t0
        fps = ok / dt if dt > 0 else 0.0
        self.measured_fps = fps
        self.log(f"Gemessene Framerate: {fps:.1f} fps")
        if fps < 20:
            self.log("WARNUNG: <20 fps -> Sprungflanke wird grob "
                     "abgetastet! Kleinere Sprunghoehe + 640x480 nutzen, "
                     "Kamerafenster ggf. aus.")
        return fps

    def _run(self):
        cfg = self.cfg
        try:
            self.status("Verbinde Arduino...")
            self._ard = ArduinoLink(cfg.serial_port, cfg.serial_baud)
            if self._ard.ping():
                self.log("Arduino antwortet (PONG).")
            else:
                self.log("WARN: kein PONG  fahre trotzdem fort.")
        except Exception as e:
            self.log(f"FEHLER Arduino: {e}")
            self.status("Abgebrochen (Arduino)")
            self.q.put(("done", None))
            return

        try:
            self.status("Oeffne Kamera...")
            self._cap = cv2.VideoCapture(cfg.camera_index, CAMERA_BACKEND)
            # MJPG anfordern: viele Webcams liefern nur damit hohe fps,
            # bei roher YUY2-Uebertragung brechen sie auf ~10 fps ein.
            self._cap.set(cv2.CAP_PROP_FOURCC,
                          cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
            self._cap.set(cv2.CAP_PROP_FPS, cfg.cam_fps)
            if not self._cap.isOpened():
                raise RuntimeError(f"Kamera {cfg.camera_index} nicht erreichbar")
            self._tracker = self._make_tracker()
            # Tatsaechliche Werte auslesen + kurze fps-Messung
            aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.log(f"Kamera offen: {aw}x{ah}, angefordert {cfg.cam_fps} fps")
            self._measure_fps()
        except Exception as e:
            self.log(f"FEHLER Kamera/Tracker: {e}")
            self.status("Abgebrochen (Kamera)")
            if self._ard:
                self._ard.close()
            self.q.put(("done", None))
            return

        axis = cfg.axis
        base = self._clamp(axis, cfg.start_angle)
        self._ard.set_angle(axis, base)
        self.log(f"Startwinkel {axis}={base} gesetzt.")
        time.sleep(0.5)

        stopped_in_preview = False
        try:
            self.status("Vorschau: Kreis zentrieren, bis GRUEN.")
            self._preview_loop(axis)
            if self._stop.is_set():
                stopped_in_preview = True
            else:
                self.status("Messung laeuft...")
                self._measure_loop(axis, base)
        except Exception as e:
            self.log(f"FEHLER: {e}")
        finally:
            try:
                self._ard.set_angle(axis, base)
                time.sleep(0.2)
            except Exception:
                pass
            if self._cap:
                self._cap.release()
            if cfg.show_window:
                cv2.destroyAllWindows()
            if self._ard:
                self._ard.close()

        if stopped_in_preview:
            self.status("Vorschau gestoppt.")
        else:
            self._analyze()
        # done IMMER senden, damit die GUI die Buttons zuruecksetzt
        self.q.put(("done", self.analysis if not stopped_in_preview else None))

    def _preview_loop(self, axis):
        buf = []
        while not self._stop.is_set() and not self._start_measure.is_set():
            ret, frame = self._cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            pos = self._tracker.find(frame)
            if pos is None:
                buf.clear()
                self.ready = False
                self.ampel("red")
                self._draw(frame, None, axis, "KEIN ZIEL", "red")
                continue
            cx, cy = pos
            err_px, _ = self._err_along_axis(cx, cy, w, h)
            buf.append(err_px)
            if len(buf) > 12:
                buf.pop(0)
            spread = float(np.std(buf)) if len(buf) >= 8 else 999.0
            centered = abs(err_px) < CENTER_TOL_PX
            stable = spread < STABILITY_PX and len(buf) >= 8
            if centered and stable:
                self.ready = True
                amp, txt = "green", "BEREIT - Messung moeglich"
            elif not centered:
                self.ready = False
                amp, txt = "yellow", "nicht zentriert"
            else:
                self.ready = False
                amp, txt = "yellow", "noch nicht ruhig"
            self.ampel(amp)
            area = getattr(self._tracker, "last_area", 0.0)
            self._draw(frame, pos, axis, txt, amp,
                       extra=f"err={err_px:+.0f}px  std={spread:.1f}  "
                             f"area={int(area)}")

    def _measure_loop(self, axis, base):
        cfg = self.cfg
        run_id = int(time.time())
        targets = []
        cur = base
        direction = +1
        for _ in range(max(1, cfg.n_steps)):
            nxt = self._clamp(axis, cur + direction * cfg.step_deg)
            targets.append((cur, nxt))
            cur = nxt
            if cfg.bidirectional:
                direction *= -1

        step_counter = 0
        for (from_ang, to_ang) in targets:
            if self._stop.is_set() or self._abort_measure.is_set():
                break
            self._ard.set_angle(axis, from_ang)
            if not self._wait_stable(axis):
                self.log("Ziel nicht stabil  Sprung uebersprungen.")
                continue
            step_signed = to_ang - from_ang
            if abs(step_signed) < 0.5:
                self.log("Sprunghoehe ~0 (Grenze?)  uebersprungen.")
                continue
            self.log(f"Sprung {step_counter}: {axis} {from_ang}->{to_ang} "
                     f"({step_signed:+.0f} Grad)")
            self._record(axis, run_id, step_counter, step_signed, from_ang,
                         cfg.pre_record_s, "pre", zero=False)
            t0 = time.perf_counter()
            self._ard.set_angle(axis, to_ang)
            self._record(axis, run_id, step_counter, step_signed, to_ang,
                         cfg.settle_record_s, "step", zero=True, t_zero=t0)
            step_counter += 1
        self.log(f"Messung fertig: {step_counter} Spruenge, "
                 f"{len(self.samples)} Punkte.")

    def _wait_stable(self, axis, timeout=5.0):
        buf = []
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._stop.is_set() or self._abort_measure.is_set():
                return False
            ret, frame = self._cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            pos = self._tracker.find(frame)
            if pos is None:
                self._draw(frame, None, axis, "warte (kein Ziel)", "red")
                continue
            cx, cy = pos
            err_px, _ = self._err_along_axis(cx, cy, w, h)
            buf.append(err_px)
            if len(buf) > 10:
                buf.pop(0)
            spread = float(np.std(buf)) if len(buf) >= 6 else 999.0
            self._draw(frame, pos, axis, "stabilisiere...", "yellow",
                       extra=f"std={spread:.1f}")
            if spread < STABILITY_PX and len(buf) >= 6:
                return True
        return False

    def _record(self, axis, run_id, step_id, step_deg, target_ang,
                duration, phase, zero, t_zero=None):
        t_start = time.perf_counter()
        while True:
            if self._stop.is_set() or self._abort_measure.is_set():
                return
            now = time.perf_counter()
            t_rel = (now - t_zero) if zero else (now - t_start)
            if t_rel > duration:
                return
            ret, frame = self._cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            pos = self._tracker.find(frame)
            if pos is None:
                self._draw(frame, None, axis, f"{phase} (kein Ziel)", "red")
                continue
            cx, cy = pos
            err_px, err_norm = self._err_along_axis(cx, cy, w, h)
            self.samples.append(Sample(
                run_id, step_id, axis, step_deg, target_ang,
                (t_rel if zero else t_rel - duration),
                err_px, err_norm, phase))
            self._draw(frame, pos, axis, f"{phase} t={t_rel:.2f}s", "green",
                       extra=f"err={err_px:+.0f}px")

    def _draw(self, frame, pos, axis, label, ampel, extra=""):
        if not self.cfg.show_window:
            return
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2, 0), (w // 2, h), (90, 90, 90), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (90, 90, 90), 1)
        cv2.circle(frame, (w // 2, h // 2), int(CENTER_TOL_PX),
                   (70, 70, 70), 1)
        col = {"red": (60, 60, 230), "yellow": (40, 200, 230),
               "green": (80, 220, 90)}.get(ampel, (200, 200, 200))
        if pos is not None:
            cx, cy = int(pos[0]), int(pos[1])
            cv2.circle(frame, (cx, cy), 9, col, -1)
            if axis == "PAN":
                cv2.line(frame, (cx, h // 2), (w // 2, h // 2), col, 2)
            else:
                cv2.line(frame, (w // 2, cy), (w // 2, h // 2), col, 2)
        cv2.circle(frame, (30, 30), 14, col, -1)
        cv2.putText(frame, f"{axis} | {label}", (55, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        if extra:
            cv2.putText(frame, extra, (55, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        if self.measured_fps > 0:
            fcol = (80, 220, 90) if self.measured_fps >= 20 else (60, 60, 230)
            cv2.putText(frame, f"{self.measured_fps:.0f} fps", (w - 130, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, fcol, 2)
        cv2.imshow("Sprungantwort Messung", frame)
        # Debug-Maske: zeigt das Binaerbild, das der Tracker auswertet.
        # Hier muss DEIN Kreis als einzige weisse Flaeche zu sehen sein.
        if self.cfg.show_mask:
            mask = getattr(self._tracker, "last_mask", None)
            if mask is not None:
                cv2.imshow("Blob-Maske (weiss = erkannt)", mask)
        cv2.waitKey(1)

    def _analyze(self):
        step_samples = [s for s in self.samples if s.phase == "step"]
        if not step_samples:
            self.analysis = {"ok": False, "msg": "Keine Sprung-Messpunkte."}
            return
        by_step = {}
        for s in step_samples:
            by_step.setdefault(s.step_id, []).append(s)
        gains_px, gains_norm, Ts, Tds = [], [], [], []
        for samps in by_step.values():
            samps = sorted(samps, key=lambda s: s.t_sec)
            step_deg = samps[0].step_deg
            if abs(step_deg) < 0.5:
                continue
            t = np.array([s.t_sec for s in samps])
            ypx = np.array([s.err_px for s in samps])
            yn = np.array([s.err_norm for s in samps])
            n = len(t)
            if n < 4:
                continue
            tail = max(1, n // 3)
            head = max(1, n // 10)
            y0 = float(np.mean(ypx[:head]))
            yend = float(np.mean(ypx[-tail:]))
            d = yend - y0
            gains_px.append(d / step_deg)
            gains_norm.append((float(np.mean(yn[-tail:])) -
                               float(np.mean(yn[:head]))) / step_deg)
            target63 = y0 + 0.632 * d
            thr = abs(d) * 0.05
            for i in range(n):
                if abs(ypx[i] - y0) > thr:
                    Tds.append(float(t[i]))
                    break
            for i in range(1, n):
                a, b = ypx[i - 1], ypx[i]
                if (a - target63) * (b - target63) <= 0 and (b - a) != 0:
                    frac = (target63 - a) / (b - a)
                    Ts.append(float(t[i - 1] + frac * (t[i] - t[i - 1])))
                    break
        if not gains_px:
            self.analysis = {"ok": False, "msg": "Spruenge zu kurz."}
            return
        # Wie viele Messpunkte liegen im Anstieg (bis 63%)? Grobe
        # Qualitaetskennzahl gegen Untersampling der Sprungflanke.
        rise_pts = []
        for samps in by_step.values():
            samps = sorted(samps, key=lambda s: s.t_sec)
            ypx = np.array([s.err_px for s in samps])
            if len(ypx) < 2:
                continue
            y0 = float(np.mean(ypx[:max(1, len(ypx) // 10)]))
            yend = float(np.mean(ypx[-max(1, len(ypx) // 3):]))
            tgt = y0 + 0.632 * (yend - y0)
            cnt = 0
            for v in ypx:
                cnt += 1
                if (yend - y0) >= 0 and v >= tgt:
                    break
                if (yend - y0) < 0 and v <= tgt:
                    break
            rise_pts.append(cnt)
        self.analysis = {
            "ok": True,
            "n_steps": len(gains_px),
            "gain_px_per_deg": float(np.mean(gains_px)),
            "gain_px_per_deg_std": float(np.std(gains_px)),
            "gain_norm_per_deg": float(np.mean(gains_norm)),
            "T_63_s": float(np.mean(Ts)) if Ts else None,
            "Td_s": float(np.mean(Tds)) if Tds else None,
            "fps": float(getattr(self, "measured_fps", 0.0)),
            "rise_points": float(np.mean(rise_pts)) if rise_pts else 0.0,
        }

    def save_csv(self, path):
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["run_id", "step_id", "axis", "step_deg",
                         "target_angle_deg", "t_sec", "err_px",
                         "err_norm", "phase"])
            for s in self.samples:
                wr.writerow([s.run_id, s.step_id, s.axis,
                             f"{s.step_deg:.3f}", f"{s.target_angle:.3f}",
                             f"{s.t_sec:.6f}", f"{s.err_px:.4f}",
                             f"{s.err_norm:.6f}", s.phase])


# ----------------------------------------------------------
# GUI
# ----------------------------------------------------------
BG = "#15171c"
PANEL = "#1e2128"
FIELD = "#272b34"
TEXT = "#e8ebf0"
MUTED = "#9aa3b2"
ACCENT = "#5cc8a0"
WARN = "#e0b450"
GREEN = "#5cc86a"
YELLOW = "#e0c050"
RED = "#e06b6b"


class App:
    def __init__(self, root):
        self.root = root
        root.title("Sprungantwort-Messung  Pan/Tilt")
        root.configure(bg=BG)
        root.geometry("580x900")

        self.q = queue.Queue()
        self.engine = None
        self.last_engine = None
        self.phase = "idle"   # idle / preview / measuring

        self._build()
        self._poll()

    def _build(self):
        head = tk.Frame(self.root, bg=BG)
        head.pack(fill="x", padx=18, pady=(16, 4))
        tk.Label(head, text="Sprungantwort-Messung", bg=BG, fg=TEXT,
                 font=("Segoe UI", 17, "bold")).pack(anchor="w")
        tk.Label(head, text="Offene Strecke (Servo + Kamera) fuer Loopshaping",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w")

        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        sb = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        body = tk.Frame(canvas, bg=BG)
        win = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(
                            int(-e.delta / 120), "units"))
        bi = tk.Frame(body, bg=BG)
        bi.pack(fill="both", expand=True, padx=18, pady=8)

        # Verbindung
        c1 = self._card(bi, "Verbindung")
        self.var_port = tk.StringVar(value=self._guess_port())
        self.var_baud = tk.IntVar(value=DEFAULT_SERIAL_BAUD)
        self.var_cam = tk.IntVar(value=DEFAULT_CAMERA_INDEX)
        self._combo(c1, "Serial Port", self.var_port, self._ports())
        self._spin(c1, "Baudrate", self.var_baud, 9600, 250000, 9600)
        self._spin(c1, "Kamera-Index", self.var_cam, 0, 10, 1)
        # Aufloesung (niedriger = mehr fps) + fps
        self.var_res = tk.StringVar(value="640x480")
        rr = tk.Frame(c1, bg=PANEL); rr.pack(fill="x", pady=4)
        self._lab(rr, "Aufloesung").pack(side="left")
        ttk.Combobox(rr, textvariable=self.var_res, state="readonly",
                     values=["640x480", "800x600", "1280x720", "1920x1080"],
                     width=12).pack(side="left")
        self.var_fps = tk.IntVar(value=DEFAULT_CAMERA_FPS)
        self._spin(c1, "Kamera fps (Soll)", self.var_fps, 5, 120, 5)

        # Messung
        c2 = self._card(bi, "Messung")
        self.var_axis = tk.StringVar(value="PAN")
        r = tk.Frame(c2, bg=PANEL); r.pack(fill="x", pady=4)
        self._lab(r, "Achse", 16).pack(side="left")
        for a in ("PAN", "TILT"):
            tk.Radiobutton(r, text=a, value=a, variable=self.var_axis,
                           bg=PANEL, fg=TEXT, selectcolor=FIELD,
                           activebackground=PANEL, activeforeground=TEXT,
                           font=("Segoe UI", 10)).pack(side="left", padx=6)
        self.var_step = tk.DoubleVar(value=5.0)
        r = tk.Frame(c2, bg=PANEL); r.pack(fill="x", pady=4)
        self._lab(r, "Sprunghoehe (Grad)", 16).pack(side="left")
        for v in (1, 5, 10):
            tk.Button(r, text=str(v), width=3,
                      command=lambda val=v: self.var_step.set(float(val)),
                      bg=FIELD, fg=TEXT, relief="flat",
                      activebackground=ACCENT).pack(side="left", padx=3)
        tk.Spinbox(r, textvariable=self.var_step, from_=0.5, to=40,
                   increment=0.5, width=6, format="%.1f").pack(side="left",
                                                               padx=8)
        self.var_start = tk.DoubleVar(value=90.0)
        self._spin_f(c2, "Startwinkel (Grad)", self.var_start, 20, 160, 1)
        self.var_nsteps = tk.IntVar(value=6)
        self._spin(c2, "Anzahl Spruenge", self.var_nsteps, 1, 50, 1)
        self.var_bidir = tk.BooleanVar(value=True)
        tk.Checkbutton(c2, text="Hin und zurueck springen (mitteln)",
                       variable=self.var_bidir, bg=PANEL, fg=TEXT,
                       selectcolor=FIELD, activebackground=PANEL,
                       activeforeground=TEXT,
                       font=("Segoe UI", 10)).pack(anchor="w", pady=4)
        self.var_pre = tk.DoubleVar(value=DEFAULT_PRE_RECORD_S)
        self.var_settle = tk.DoubleVar(value=DEFAULT_SETTLE_RECORD_S)
        self._spin_f(c2, "Vorlauf (s)", self.var_pre, 0.0, 2.0, 0.1)
        self._spin_f(c2, "Aufnahme nach Sprung (s)", self.var_settle,
                     0.3, 5.0, 0.1)

        # Ziel-Erkennung
        c3 = self._card(bi, "Ziel-Erkennung")
        self.var_tracker = tk.StringVar(value="BLOB")
        tk.Radiobutton(c3, text="Kontrast-Blob: schwarzer Kreis (empfohlen)",
                       value="BLOB", variable=self.var_tracker, bg=PANEL,
                       fg=TEXT, selectcolor=FIELD, activebackground=PANEL,
                       activeforeground=TEXT,
                       font=("Segoe UI", 10)).pack(anchor="w")
        mp_state = "normal" if MEDIAPIPE_AVAILABLE else "disabled"
        tk.Radiobutton(c3, text="MediaPipe-Gesicht (festes Foto!)",
                       value="MEDIAPIPE", variable=self.var_tracker,
                       bg=PANEL, fg=TEXT, selectcolor=FIELD,
                       activebackground=PANEL, activeforeground=TEXT,
                       state=mp_state,
                       font=("Segoe UI", 10)).pack(anchor="w")
        if not MEDIAPIPE_AVAILABLE:
            tk.Label(c3, text="(mediapipe nicht installiert)", bg=PANEL,
                     fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")
        self.var_minarea = tk.IntVar(value=200)
        self._spin(c3, "Blob Mindestflaeche (px2)", self.var_minarea,
                   50, 5000, 50)

        # Schwelle: AUTO (Otsu) oder fester Wert
        self.var_threshmode = tk.StringVar(value="AUTO")
        tr = tk.Frame(c3, bg=PANEL); tr.pack(fill="x", pady=4)
        self._lab(tr, "Schwelle").pack(side="left")
        for m in ("AUTO", "FIXED"):
            tk.Radiobutton(tr, text=m, value=m, variable=self.var_threshmode,
                           bg=PANEL, fg=TEXT, selectcolor=FIELD,
                           activebackground=PANEL, activeforeground=TEXT,
                           font=("Segoe UI", 9)).pack(side="left", padx=4)
        self.var_threshval = tk.IntVar(value=90)
        self._spin(c3, "  fester Wert (0-255)", self.var_threshval, 0, 255, 5)

        # Polaritaet
        self.var_dark = tk.BooleanVar(value=True)
        tk.Checkbutton(c3, text="Ziel ist dunkel auf hell "
                                "(aus = hell auf dunkel)",
                       variable=self.var_dark, bg=PANEL, fg=TEXT,
                       selectcolor=FIELD, activebackground=PANEL,
                       activeforeground=TEXT,
                       font=("Segoe UI", 9)).pack(anchor="w", pady=2)

        # Rundheit
        self.var_circ = tk.DoubleVar(value=0.6)
        self._spin_f(c3, "Min. Rundheit (0-1)", self.var_circ, 0.0, 1.0, 0.05)

        self.var_show = tk.BooleanVar(value=True)
        tk.Checkbutton(c3, text="Kamerafenster anzeigen (zum Zentrieren)",
                       variable=self.var_show, bg=PANEL, fg=TEXT,
                       selectcolor=FIELD, activebackground=PANEL,
                       activeforeground=TEXT,
                       font=("Segoe UI", 10)).pack(anchor="w", pady=4)
        self.var_mask = tk.BooleanVar(value=False)
        tk.Checkbutton(c3, text="Blob-Maske anzeigen (Debug: was wird "
                                "erkannt?)",
                       variable=self.var_mask, bg=PANEL, fg=TEXT,
                       selectcolor=FIELD, activebackground=PANEL,
                       activeforeground=TEXT,
                       font=("Segoe UI", 10)).pack(anchor="w", pady=2)

        # Ablauf: Ampel + Buttons
        ctrl = self._card(bi, "Ablauf")
        amp = tk.Frame(ctrl, bg=PANEL); amp.pack(fill="x", pady=(0, 8))
        self.canvas_amp = tk.Canvas(amp, width=26, height=26, bg=PANEL,
                                    highlightthickness=0)
        self.canvas_amp.pack(side="left", padx=(0, 8))
        self.amp_dot = self.canvas_amp.create_oval(4, 4, 22, 22,
                                                   fill="#555", outline="")
        self.lbl_amp = tk.Label(amp, text="Bereit zum Start.", bg=PANEL,
                                fg=MUTED, font=("Segoe UI", 10, "bold"))
        self.lbl_amp.pack(side="left")

        brow = tk.Frame(ctrl, bg=PANEL); brow.pack(fill="x")
        self.btn_preview = tk.Button(brow, text="1) Vorschau starten",
                                     command=self.on_preview, bg=ACCENT,
                                     fg="#11221b", relief="flat", pady=10,
                                     font=("Segoe UI", 10, "bold"))
        self.btn_preview.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_measure = tk.Button(brow, text="2) Messung starten",
                                     command=self.on_measure, bg=FIELD,
                                     fg=MUTED, relief="flat", pady=10,
                                     state="disabled",
                                     font=("Segoe UI", 10, "bold"))
        self.btn_measure.pack(side="left", fill="x", expand=True, padx=4)

        brow2 = tk.Frame(ctrl, bg=PANEL); brow2.pack(fill="x", pady=(6, 0))
        self.btn_stop = tk.Button(brow2, text="Stop / Abbrechen",
                                  command=self.on_stop, bg=FIELD, fg=TEXT,
                                  relief="flat", pady=8, state="disabled",
                                  font=("Segoe UI", 10, "bold"))
        self.btn_stop.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_plot = tk.Button(brow2, text="Plot ansehen",
                                  command=self.on_plot, bg=FIELD, fg=MUTED,
                                  relief="flat", pady=8, state="disabled",
                                  font=("Segoe UI", 10, "bold"))
        self.btn_plot.pack(side="left", fill="x", expand=True, padx=4)
        self.btn_save = tk.Button(brow2, text="CSV speichern",
                                  command=self.on_save, bg=FIELD, fg=MUTED,
                                  relief="flat", pady=8, state="disabled",
                                  font=("Segoe UI", 10, "bold"))
        self.btn_save.pack(side="left", fill="x", expand=True, padx=4)

        # Speicherordner waehlbar
        srow = tk.Frame(ctrl, bg=PANEL); srow.pack(fill="x", pady=(8, 0))
        self._lab(srow, "Speicherordner", 13).pack(side="left")
        self.var_savedir = tk.StringVar(value=os.getcwd())
        tk.Entry(srow, textvariable=self.var_savedir, bg=FIELD, fg=TEXT,
                 relief="flat", insertbackground=TEXT).pack(
                     side="left", fill="x", expand=True, padx=(4, 4))
        tk.Button(srow, text="...", command=self.on_choose_dir, bg=FIELD,
                  fg=TEXT, relief="flat", width=3).pack(side="left")

        # Auswertung
        ana = self._card(bi, "Auswertung (nach Lauf)")
        self.lbl_ana = tk.Label(ana, text="Noch keine Messung.", bg=PANEL,
                                fg=MUTED, font=("Consolas", 10),
                                justify="left", anchor="w")
        self.lbl_ana.pack(anchor="w", fill="x")

        # Status/Log
        self.lbl_status = tk.Label(bi, text="Bereit.", bg=BG, fg=ACCENT,
                                   font=("Segoe UI", 10, "bold"), anchor="w")
        self.lbl_status.pack(fill="x", pady=(8, 2))
        self.txt_log = tk.Text(bi, height=6, bg="#0e0f13", fg=MUTED,
                               font=("Consolas", 9), relief="flat",
                               wrap="word")
        self.txt_log.pack(fill="both", expand=False)

    def _card(self, parent, title):
        wrap = tk.Frame(parent, bg=BG); wrap.pack(fill="x", pady=6)
        tk.Label(wrap, text=title, bg=BG, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))
        card = tk.Frame(wrap, bg=PANEL); card.pack(fill="x")
        inner = tk.Frame(card, bg=PANEL); inner.pack(fill="x", padx=12, pady=10)
        return inner

    def _lab(self, parent, text, width=16):
        return tk.Label(parent, text=text, bg=parent["bg"], fg=TEXT,
                        font=("Segoe UI", 10), anchor="w", width=width)

    def _combo(self, parent, label, var, values):
        r = tk.Frame(parent, bg=PANEL); r.pack(fill="x", pady=4)
        self._lab(r, label).pack(side="left")
        ttk.Combobox(r, textvariable=var, values=values,
                     width=22).pack(side="left", fill="x", expand=True)

    def _spin(self, parent, label, var, frm, to, inc):
        r = tk.Frame(parent, bg=PANEL); r.pack(fill="x", pady=4)
        self._lab(r, label).pack(side="left")
        tk.Spinbox(r, textvariable=var, from_=frm, to=to, increment=inc,
                   width=10).pack(side="left")

    def _spin_f(self, parent, label, var, frm, to, inc):
        r = tk.Frame(parent, bg=PANEL); r.pack(fill="x", pady=4)
        self._lab(r, label).pack(side="left")
        tk.Spinbox(r, textvariable=var, from_=frm, to=to, increment=inc,
                   width=10, format="%.1f").pack(side="left")

    def _ports(self):
        return [p.device for p in serial.tools.list_ports.comports()]

    def _guess_port(self):
        p = self._ports()
        return p[0] if p else "COM4"

    def _read_cfg(self):
        return MeasureConfig(
            serial_port=self.var_port.get().strip(),
            serial_baud=int(self.var_baud.get()),
            camera_index=int(self.var_cam.get()),
            axis=self.var_axis.get(),
            step_deg=float(self.var_step.get()),
            start_angle=float(self.var_start.get()),
            n_steps=int(self.var_nsteps.get()),
            bidirectional=bool(self.var_bidir.get()),
            pre_record_s=float(self.var_pre.get()),
            settle_record_s=float(self.var_settle.get()),
            tracker=self.var_tracker.get(),
            show_window=bool(self.var_show.get()),
            blob_min_area=int(self.var_minarea.get()),
            blob_thresh_mode=self.var_threshmode.get(),
            blob_thresh_val=int(self.var_threshval.get()),
            blob_dark_target=bool(self.var_dark.get()),
            blob_min_circ=float(self.var_circ.get()),
            show_mask=bool(self.var_mask.get()),
            cam_width=int(self.var_res.get().split("x")[0]),
            cam_height=int(self.var_res.get().split("x")[1]),
            cam_fps=int(self.var_fps.get()),
        )

    def on_preview(self):
        if self.engine is not None:
            return
        # Falls ein vorheriger Thread noch auslaeuft (Kamera/Serial werden
        # gerade freigegeben), kurz darauf warten, damit der Kamera-Index
        # beim Neustart wieder frei ist.
        if (self.last_engine is not None
                and self.last_engine.thread is not None
                and self.last_engine.thread.is_alive()):
            self.last_engine.stop_all()
            self.last_engine.thread.join(timeout=2.0)
        # Alte, noch in der Queue haengende Events (insb. 'done' vom
        # gestoppten Lauf) verwerfen, damit sie den neuen Lauf nicht
        # sofort wieder beenden.
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        cfg = self._read_cfg()
        self.txt_log.delete("1.0", "end")
        self.engine = MeasureEngine(cfg, self.q)
        self.last_engine = self.engine
        self.phase = "preview"
        self.btn_preview.config(state="disabled", bg=FIELD, fg=MUTED)
        self.btn_stop.config(state="normal")
        self.btn_save.config(state="disabled", fg=MUTED, bg=FIELD)
        self.btn_plot.config(state="disabled", fg=MUTED, bg=FIELD)
        self.btn_measure.config(state="disabled", fg=MUTED, bg=FIELD)
        self.engine.start_preview()

    def on_measure(self):
        if self.engine is None or self.phase != "preview":
            return
        if not self.engine.ready:
            messagebox.showwarning(
                "Noch nicht bereit",
                "Der Kreis ist noch nicht stabil zentriert (Ampel nicht "
                "GRUEN). Richte das Ziel aus, bis die Ampel gruen ist.")
            return
        self.phase = "measuring"
        self.btn_measure.config(state="disabled", fg=MUTED, bg=FIELD)
        self.engine.trigger_measurement()

    def on_stop(self):
        if self.engine is None:
            return
        if self.phase == "measuring":
            self.engine.abort_measurement()
            self._log("Messung abgebrochen.")
        self.engine.stop_all()

    def on_choose_dir(self):
        d = filedialog.askdirectory(
            initialdir=self.var_savedir.get() or os.getcwd(),
            title="Speicherordner fuer CSV-Dateien waehlen")
        if d:
            self.var_savedir.set(d)
            self._log(f"Speicherordner: {d}")

    def on_save(self):
        if self.last_engine is None or not self.last_engine.samples:
            messagebox.showwarning("Keine Daten", "Keine Messdaten vorhanden.")
            return
        cfg = self.last_engine.cfg
        default = (f"sprungantwort_{cfg.axis}_{cfg.step_deg:.0f}deg_"
                   f"{time.strftime('%Y%m%d_%H%M%S')}.csv")
        init_dir = self.var_savedir.get().strip() or os.getcwd()
        if not os.path.isdir(init_dir):
            init_dir = os.getcwd()
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=default,
            initialdir=init_dir,
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            self.last_engine.save_csv(path)
            # gewaehlten Ordner als neuen Default merken
            self.var_savedir.set(os.path.dirname(path))
            self._log(f"CSV gespeichert: {path}")
            messagebox.showinfo("Gespeichert", f"CSV gespeichert:\n{path}")
        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    def on_plot(self):
        """Schnelle Sichtpruefung: alle Sprung-Antworten uebereinander,
        Zeit ab Sprung gegen Pixelfehler. Nur zur Kontrolle, die
        eigentliche Analyse machst du in MATLAB aus der CSV."""
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showwarning(
                "matplotlib fehlt",
                "Zum Plotten:  pip install matplotlib")
            return
        if self.last_engine is None or not self.last_engine.samples:
            messagebox.showwarning("Keine Daten", "Keine Messdaten vorhanden.")
            return

        samples = self.last_engine.samples
        cfg = self.last_engine.cfg
        step_samples = [s for s in samples if s.phase == "step"]
        if not step_samples:
            messagebox.showwarning("Keine Daten",
                                   "Keine Sprung-Messpunkte vorhanden.")
            return

        # nach step_id gruppieren
        by_step = {}
        for s in step_samples:
            by_step.setdefault(s.step_id, []).append(s)

        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            for step_id, samps in sorted(by_step.items()):
                samps = sorted(samps, key=lambda s: s.t_sec)
                t = [s.t_sec * 1000.0 for s in samps]   # ms
                y = [s.err_px for s in samps]
                sign = "+" if samps[0].step_deg >= 0 else "-"
                ax.plot(t, y, marker=".", linewidth=1,
                        label=f"Sprung {step_id} ({sign}{abs(samps[0].step_deg):.0f}\u00b0)")
            ax.axhline(0, color="#888", linewidth=0.8)
            ax.set_xlabel("Zeit ab Sprung [ms]")
            ax.set_ylabel("Pixelfehler entlang Achse [px]")
            ax.set_title(f"Sprungantwort  {cfg.axis}  "
                         f"{cfg.step_deg:.0f}\u00b0  "
                         f"({len(by_step)} Spruenge)")
            ax.grid(True, alpha=0.3)
            if len(by_step) <= 12:
                ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            plt.show(block=False)
            self._log("Plot geoeffnet.")
        except Exception as e:
            messagebox.showerror("Plot-Fehler", str(e))

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "status":
                    self.lbl_status.config(text=payload)
                elif kind == "ampel":
                    self._set_ampel(payload)
                elif kind == "done":
                    self._on_done(payload)
        except queue.Empty:
            pass
        if (self.engine is not None and self.phase == "preview"
                and self.engine.ready):
            self.btn_measure.config(state="normal", fg="#11221b", bg=ACCENT)
        elif self.phase == "preview":
            self.btn_measure.config(state="disabled", fg=MUTED, bg=FIELD)
        self.root.after(100, self._poll)

    def _set_ampel(self, state):
        col = {"red": RED, "yellow": YELLOW, "green": GREEN}.get(state, "#555")
        txt = {"red": "Kein Ziel erkannt.",
               "yellow": "Ziel erkannt  noch nicht bereit.",
               "green": "Bereit  Messung moeglich."}.get(state, "")
        self.canvas_amp.itemconfig(self.amp_dot, fill=col)
        self.lbl_amp.config(text=txt, fg=col)

    def _on_done(self, analysis):
        had_samples = bool(self.last_engine and self.last_engine.samples)
        self.engine = None
        self.phase = "idle"
        self.btn_preview.config(state="normal", bg=ACCENT, fg="#11221b")
        self.btn_measure.config(state="disabled", fg=MUTED, bg=FIELD)
        self.btn_stop.config(state="disabled")
        self.canvas_amp.itemconfig(self.amp_dot, fill="#555")
        self.lbl_amp.config(text="Bereit zum Start.", fg=MUTED)
        if had_samples:
            self.btn_save.config(state="normal", fg=TEXT, bg=ACCENT)
            self.btn_plot.config(state="normal", fg=TEXT, bg=FIELD)
        # Auswertungsbox nur aktualisieren, wenn es echte Daten gibt.
        # Bei reinem Stop in der Vorschau (analysis None, keine Samples)
        # bleibt die Box unveraendert statt "Keine Auswertung" zu zeigen.
        if analysis is not None or had_samples:
            self._show_analysis(analysis)

    def _show_analysis(self, a):
        if not a or not a.get("ok"):
            msg = a.get("msg", "Keine Auswertung.") if a else "Keine Auswertung."
            self.lbl_ana.config(text=msg, fg=WARN)
            return
        gpx = a["gain_px_per_deg"]; gpxs = a["gain_px_per_deg_std"]
        gn = a["gain_norm_per_deg"]; T = a["T_63_s"]; Td = a["Td_s"]
        fps = a.get("fps", 0.0); rise = a.get("rise_points", 0.0)
        lines = [
            f"Spruenge ausgewertet : {a['n_steps']}",
            f"Verstaerkung K       : {gpx:+.2f} px/Grad  (std {gpxs:.2f})",
            f"                       {gn:+.4f} norm/Grad",
            "Zeitkonstante T(63%) : " + (f"{T*1000:.0f} ms" if T else "n/v"),
            "Totzeit Td (grob)    : " + (f"{Td*1000:.0f} ms" if Td else "n/v"),
            f"Framerate / Anstieg  : {fps:.0f} fps, ~{rise:.0f} Punkte",
            "",
            "-> Startschaetzung MATLAB tfest (PT1+Totzeit):",
        ]
        if T:
            lines.append(f"   K={gpx:.2f} px/Grad, T={T*1000:.0f} ms, "
                         + (f"Td={Td*1000:.0f} ms" if Td else "Td~0"))
        else:
            lines.append("   K bekannt, T/Td unklar (mehr Spruenge messen)")
        # Untersampling-Warnung
        warn = (rise > 0 and rise < 4) or (0 < fps < 20)
        if warn:
            lines.append("")
            lines.append("WARNUNG: Sprungflanke zu grob abgetastet!")
            lines.append("Kleinere Sprunghoehe (1-5 Grad), 640x480,")
            lines.append("Kamerafenster aus -> mehr fps.")
        self.lbl_ana.config(text="\n".join(lines),
                            fg=(WARN if warn else TEXT))

    def _log(self, msg):
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
