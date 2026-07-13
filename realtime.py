import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
from ai_edge_litert.interpreter import Interpreter
from processor import DriverStateProcessor
from flask import Flask, jsonify
import threading

app = Flask(__name__)

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD TFLITE MODEL
# ─────────────────────────────────────────────────────────────
interpreter = Interpreter(model_path="eye_model.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_eye_v1(eye_crop):
    gray    = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32))
    norm    = resized.astype(np.float32) / 255.0
    tensor  = norm.reshape(1, 32, 32, 1)
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke()
    prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return float(prob)

# ─────────────────────────────────────────────────────────────
# STEP 1b — LOAD V2 TFLITE MODEL (MobileNetV2)
# ─────────────────────────────────────────────────────────────
interpreter_v2 = Interpreter(model_path="eye_model_v2.tflite")
interpreter_v2.allocate_tensors()
input_details_v2  = interpreter_v2.get_input_details()
output_details_v2 = interpreter_v2.get_output_details()

def predict_eye_v2(eye_crop):
    # V2 expects: 96x96 RGB, normalised -1 to 1
    resized = cv2.resize(eye_crop, (96, 96))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm    = rgb.astype(np.float32)
    norm    = (norm / 127.5) - 1.0
    tensor  = norm.reshape(1, 96, 96, 3)
    interpreter_v2.set_tensor(input_details_v2[0]['index'], tensor)
    interpreter_v2.invoke()
    prob = interpreter_v2.get_tensor(output_details_v2[0]['index'])[0][0]
    return float(prob)

# ─────────────────────────────────────────────────────────────
# STEP 2 — MEDIAPIPE SETUP
# ─────────────────────────────────────────────────────────────
BaseOptions        = mp.tasks.BaseOptions
FaceLandmarker     = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
RunningMode        = mp.tasks.vision.RunningMode

LEFT_EYE      = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE     = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_EAR_PTS  = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_PTS = [33,  160, 158, 133, 153, 144]

options = FaceLandmarkerOpts(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task",
        delegate=BaseOptions.Delegate.CPU
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ─────────────────────────────────────────────────────────────
# STEP 3 — EAR + HELPERS
# ─────────────────────────────────────────────────────────────
def compute_ear(landmarks, pts, h, w):
    def p(i):
        lm = landmarks[pts[i]]
        return np.array([lm.x * w, lm.y * h])
    p1,p2,p3,p4,p5,p6 = [p(i) for i in range(6)]
    return round((np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)) / (2.0 * np.linalg.norm(p1-p4)), 3)

def final_state(cnn_prob, ear_val):
    if ear_val > 0.25:   return "OPEN",   "EAR"
    elif ear_val < 0.18: return "CLOSED", "EAR"
    else:                return ("OPEN" if cnn_prob > 0.5 else "CLOSED"), "CNN"

def get_eye_crop(frame, landmarks, eye_indices, h, w, pad=12):
    xs = [int(landmarks[i].x * w) for i in eye_indices]
    ys = [int(landmarks[i].y * h) for i in eye_indices]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad)
    y2 = min(h, max(ys) + pad)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def draw_eye_sharp(frame, landmarks, indices, color, h, w):
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        pts.append((x, y))
        cv2.circle(frame, (x, y), 1, color, -1, cv2.LINE_AA)
    for i in range(len(pts)):
        cv2.line(frame, pts[i], pts[(i+1) % len(pts)], color, 1, cv2.LINE_AA)

def draw_label(frame, text, pos, color):
    x, y = pos
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(frame, (x-2, y-th-4), (x+tw+2, y+2), (0,0,0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
# STEP 4 — GRAPH PANEL DRAWING
# Right panel — 400x480 black canvas drawn with OpenCV
# ─────────────────────────────────────────────────────────────
PANEL_W = 400
PANEL_H = 480
GRAPH_H = 150    # height of PERCLOS graph area
GRAPH_Y = 120    # y position of graph top
GRAPH_X = 40     # x position of graph left
GRAPH_W = 320    # width of graph

# rolling history for graph line
perclos_history = deque(maxlen=GRAPH_W)

STATE_COLORS = {
    "CALIBRATING": (150, 150, 150),
    "ACTIVE":      (0,   200, 80),
    "DROWSY":      (0,   165, 255),
    "DANGER":      (0,   0,   255),
}

def draw_graph_panel(state, perclos, votes, frame_count, calib_n=300):
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (15, 15, 15)   # dark background

    s_color = STATE_COLORS.get(state, (150,150,150))

    # ── State label (top) ─────────────────────────────────────
    cv2.rectangle(panel, (10, 10), (PANEL_W-10, 60), (30,30,30), -1)
    cv2.rectangle(panel, (10, 10), (PANEL_W-10, 60), s_color, 2)
    cv2.putText(panel, state,
                (20, 47), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                s_color, 2, cv2.LINE_AA)

    # ── PERCLOS value ─────────────────────────────────────────
    cv2.putText(panel, f"PERCLOS: {perclos:.1f}%",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Votes: {votes}/4",
                (220, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200,200,200), 1, cv2.LINE_AA)

    # ── PERCLOS graph area ────────────────────────────────────
    # background
    cv2.rectangle(panel,
                  (GRAPH_X, GRAPH_Y),
                  (GRAPH_X+GRAPH_W, GRAPH_Y+GRAPH_H),
                  (30,30,30), -1)

    # threshold lines
    danger_y = GRAPH_Y + int(GRAPH_H * (1 - 25/60))
    drowsy_y = GRAPH_Y + int(GRAPH_H * (1 - 15/60))
    cv2.line(panel, (GRAPH_X, danger_y), (GRAPH_X+GRAPH_W, danger_y), (0,0,200), 1)
    cv2.line(panel, (GRAPH_X, drowsy_y), (GRAPH_X+GRAPH_W, drowsy_y), (0,130,255), 1)

    # labels for threshold lines
    cv2.putText(panel, "25%", (GRAPH_X+GRAPH_W+5, danger_y+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,200), 1)
    cv2.putText(panel, "15%", (GRAPH_X+GRAPH_W+5, drowsy_y+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,130,255), 1)

    # graph border
    cv2.rectangle(panel,
                  (GRAPH_X, GRAPH_Y),
                  (GRAPH_X+GRAPH_W, GRAPH_Y+GRAPH_H),
                  (60,60,60), 1)

    # graph title
    cv2.putText(panel, "PERCLOS over time",
                (GRAPH_X, GRAPH_Y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

    # draw PERCLOS line
    perclos_history.append(min(perclos, 100))
    if len(perclos_history) > 1:
        pts = list(perclos_history)
        for i in range(1, len(pts)):
            x1 = GRAPH_X + i - 1
            x2 = GRAPH_X + i
            y1 = GRAPH_Y + int(GRAPH_H * (1 - pts[i-1]/100))
            y2 = GRAPH_Y + int(GRAPH_H * (1 - pts[i]/100))
            y1 = max(GRAPH_Y, min(GRAPH_Y+GRAPH_H, y1))
            y2 = max(GRAPH_Y, min(GRAPH_Y+GRAPH_H, y2))

            # color line by current perclos value
            if pts[i] > 25:   lcolor = (0,0,255)
            elif pts[i] > 15: lcolor = (0,165,255)
            else:              lcolor = (0,200,80)
            cv2.line(panel, (x1, y1), (x2, y2), lcolor, 2)

    # ── Model votes bars ──────────────────────────────────────
    model_y = GRAPH_Y + GRAPH_H + 20
    cv2.putText(panel, "Model votes",
                (20, model_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
    model_y += 18

    model_labels = ["KMeans", "GMM   ", "IsoFor", "SVM   "]
    for idx, label in enumerate(model_labels):
        y = model_y + idx * 22
        voted = idx < votes   # simple visual — first N models voted closed

        color = (0,0,255) if voted else (0,180,80)
        text  = "CLOSED" if voted else "OPEN  "

        cv2.putText(panel, f"{label} [{text}]",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    color, 1, cv2.LINE_AA)

        # bar
        bar_w = 80 if voted else 40
        cv2.rectangle(panel, (220, y-10), (220+bar_w, y+2), color, -1)

    # ── Calibration progress ──────────────────────────────────
    if state == "CALIBRATING":
        prog   = min(frame_count / calib_n, 1.0)
        prog_y = PANEL_H - 40
        cv2.putText(panel, f"Calibrating... {int(prog*100)}%",
                    (20, prog_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
        cv2.rectangle(panel, (20, prog_y+8), (PANEL_W-20, prog_y+20),
                      (50,50,50), -1)
        cv2.rectangle(panel, (20, prog_y+8),
                      (20 + int((PANEL_W-40)*prog), prog_y+20),
                      (0,200,80), -1)
    else:
        cv2.putText(panel, f"Frame: {frame_count}",
                    (20, PANEL_H-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1)

    return panel


# ─────────────────────────────────────────────────────────────
# STEP 5 — PROCESSOR
# ─────────────────────────────────────────────────────────────
processor = DriverStateProcessor()


# ── Flask API — serves state to laptop ROS2 bridge ───────────
@app.route('/state', methods=['GET'])
def get_state():
    return jsonify({
        "state":   processor.get_state(),
        "perclos": round(processor.get_perclos(), 1),
        "votes":   processor.get_votes()
    })

def start_api():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

api_thread = threading.Thread(target=start_api, daemon=True)
api_thread.start()
print("API running on http://0.0.0.0:5000")



# ── TEMPORARY TEST — confirm both models loaded correctly ─────
print(f"V1 input shape: {input_details[0]['shape']}")
print(f"V2 input shape: {input_details_v2[0]['shape']}")



# ─────────────────────────────────────────────────────────────
# STEP 6 — MAIN LOOP
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture("/home/raspberrypi/car/driver2.webm")
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("Running — press Q to quit.")

with FaceLandmarker.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            # video ended — restart from beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 480))
        h, w  = frame.shape[:2]
        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_img, ts_ms)

        ear_l_val = 0.0
        ear_r_val = 0.0
        cnn_l_val = 0.5
        cnn_r_val = 0.5

        if result.face_landmarks:
            lms = result.face_landmarks[0]

            # draw landmarks
            draw_eye_sharp(frame, lms, LEFT_EYE,  (0, 255, 80),  h, w)
            draw_eye_sharp(frame, lms, RIGHT_EYE, (0, 220, 255), h, w)

            # compute EAR
            ear_l_val = compute_ear(lms, LEFT_EAR_PTS,  h, w)
            ear_r_val = compute_ear(lms, RIGHT_EAR_PTS, h, w)

            # get eye crops and CNN prediction
            l_crop, (lx1,ly1,lx2,ly2) = get_eye_crop(frame, lms, LEFT_EYE,  h, w)
            r_crop, (rx1,ry1,rx2,ry2) = get_eye_crop(frame, lms, RIGHT_EYE, h, w)

            cnn_l_val = predict_eye_v1(l_crop) if l_crop.size > 0 else 0.5
            cnn_r_val = predict_eye_v1(r_crop) if r_crop.size > 0 else 0.5

            # hybrid EAR + CNN decision
            l_state, l_src = final_state(cnn_l_val, ear_l_val)
            r_state, r_src = final_state(cnn_r_val, ear_r_val)

            l_color = (0,255,80)  if l_state == "OPEN" else (0,0,255)
            r_color = (0,220,255) if r_state == "OPEN" else (0,0,255)

            cv2.rectangle(frame, (lx1,ly1), (lx2,ly2), l_color, 1)
            cv2.rectangle(frame, (rx1,ry1), (rx2,ry2), r_color, 1)

            draw_label(frame, f"L:{l_state} {ear_l_val:.2f}[{l_src}]", (lx1, ly1-5), l_color)
            draw_label(frame, f"R:{r_state} {ear_r_val:.2f}[{r_src}]", (rx1, ry1-5), r_color)

        else:
            draw_label(frame, "No face detected", (10, 30), (0,0,255))

        # ── update processor every frame ──────────────────────
        processor.update(ear_l_val, ear_r_val, cnn_l_val, cnn_r_val)
        state   = processor.get_state()
        perclos = processor.get_perclos()
        votes   = processor.get_votes()

        # ── draw graph panel ──────────────────────────────────
        panel = draw_graph_panel(state, perclos, votes, processor.frame_count)

        # ── combine webcam + panel side by side ───────────────
        combined = np.hstack([frame, panel])

        cv2.imshow("Driver Monitor", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
from ai_edge_litert.interpreter import Interpreter
from processor import DriverStateProcessor

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD TFLITE MODEL
# ─────────────────────────────────────────────────────────────
interpreter = Interpreter(model_path="eye_model.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_eye(eye_crop):
    gray    = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32))
    norm    = resized.astype(np.float32) / 255.0
    tensor  = norm.reshape(1, 32, 32, 1)
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke()
    prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return float(prob)


# ─────────────────────────────────────────────────────────────
# STEP 2 — MEDIAPIPE SETUP
# ─────────────────────────────────────────────────────────────
BaseOptions        = mp.tasks.BaseOptions
FaceLandmarker     = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
RunningMode        = mp.tasks.vision.RunningMode

LEFT_EYE      = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE     = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_EAR_PTS  = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_PTS = [33,  160, 158, 133, 153, 144]

options = FaceLandmarkerOpts(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task",
        delegate=BaseOptions.Delegate.CPU
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ─────────────────────────────────────────────────────────────
# STEP 3 — EAR + HELPERS
# ─────────────────────────────────────────────────────────────
def compute_ear(landmarks, pts, h, w):
    def p(i):
        lm = landmarks[pts[i]]
        return np.array([lm.x * w, lm.y * h])
    p1,p2,p3,p4,p5,p6 = [p(i) for i in range(6)]
    return round((np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)) / (2.0 * np.linalg.norm(p1-p4)), 3)

def final_state(cnn_prob, ear_val):
    if ear_val > 0.25:   return "OPEN",   "EAR"
    elif ear_val < 0.18: return "CLOSED", "EAR"
    else:                return ("OPEN" if cnn_prob > 0.5 else "CLOSED"), "CNN"

def get_eye_crop(frame, landmarks, eye_indices, h, w, pad=12):
    xs = [int(landmarks[i].x * w) for i in eye_indices]
    ys = [int(landmarks[i].y * h) for i in eye_indices]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad)
    y2 = min(h, max(ys) + pad)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def draw_eye_sharp(frame, landmarks, indices, color, h, w):
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        pts.append((x, y))
        cv2.circle(frame, (x, y), 1, color, -1, cv2.LINE_AA)
    for i in range(len(pts)):
        cv2.line(frame, pts[i], pts[(i+1) % len(pts)], color, 1, cv2.LINE_AA)

def draw_label(frame, text, pos, color):
    x, y = pos
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(frame, (x-2, y-th-4), (x+tw+2, y+2), (0,0,0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
# STEP 4 — GRAPH PANEL DRAWING
# Right panel — 400x480 black canvas drawn with OpenCV
# ─────────────────────────────────────────────────────────────
PANEL_W = 400
PANEL_H = 480
GRAPH_H = 150    # height of PERCLOS graph area
GRAPH_Y = 120    # y position of graph top
GRAPH_X = 40     # x position of graph left
GRAPH_W = 320    # width of graph

# rolling history for graph line
perclos_history = deque(maxlen=GRAPH_W)

STATE_COLORS = {
    "CALIBRATING": (150, 150, 150),
    "ACTIVE":      (0,   200, 80),
    "DROWSY":      (0,   165, 255),
    "DANGER":      (0,   0,   255),
}

def draw_graph_panel(state, perclos, votes, frame_count, calib_n=300):
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (15, 15, 15)   # dark background

    s_color = STATE_COLORS.get(state, (150,150,150))

    # ── State label (top) ─────────────────────────────────────
    cv2.rectangle(panel, (10, 10), (PANEL_W-10, 60), (30,30,30), -1)
    cv2.rectangle(panel, (10, 10), (PANEL_W-10, 60), s_color, 2)
    cv2.putText(panel, state,
                (20, 47), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                s_color, 2, cv2.LINE_AA)

    # ── PERCLOS value ─────────────────────────────────────────
    cv2.putText(panel, f"PERCLOS: {perclos:.1f}%",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Votes: {votes}/4",
                (220, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200,200,200), 1, cv2.LINE_AA)

    # ── PERCLOS graph area ────────────────────────────────────
    # background
    cv2.rectangle(panel,
                  (GRAPH_X, GRAPH_Y),
                  (GRAPH_X+GRAPH_W, GRAPH_Y+GRAPH_H),
                  (30,30,30), -1)

    # threshold lines
    danger_y = GRAPH_Y + int(GRAPH_H * (1 - 57/100))
    drowsy_y = GRAPH_Y + int(GRAPH_H * (1 - 20/100))
    cv2.line(panel, (GRAPH_X, danger_y), (GRAPH_X+GRAPH_W, danger_y), (0,0,200), 1)
    cv2.line(panel, (GRAPH_X, drowsy_y), (GRAPH_X+GRAPH_W, drowsy_y), (0,130,255), 1)

    # labels for threshold lines
    cv2.putText(panel, "50%", (GRAPH_X+GRAPH_W+5, danger_y+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,200), 1)
    cv2.putText(panel, "20%", (GRAPH_X+GRAPH_W+5, drowsy_y+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,130,255), 1)

    # graph border
    cv2.rectangle(panel,
                  (GRAPH_X, GRAPH_Y),
                  (GRAPH_X+GRAPH_W, GRAPH_Y+GRAPH_H),
                  (60,60,60), 1)

    # graph title
    cv2.putText(panel, "PERCLOS over time",
                (GRAPH_X, GRAPH_Y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

    # draw PERCLOS line
    perclos_history.append(min(perclos, 60))
    if len(perclos_history) > 1:
        pts = list(perclos_history)
        for i in range(1, len(pts)):
            x1 = GRAPH_X + i - 1
            x2 = GRAPH_X + i
            y1 = GRAPH_Y + int(GRAPH_H * (1 - pts[i-1]/60))
            y2 = GRAPH_Y + int(GRAPH_H * (1 - pts[i]/60))
            y1 = max(GRAPH_Y, min(GRAPH_Y+GRAPH_H, y1))
            y2 = max(GRAPH_Y, min(GRAPH_Y+GRAPH_H, y2))

            # color line by current perclos value
            if pts[i] > 57:   lcolor = (0,0,255)
            elif pts[i] > 20: lcolor = (0,165,255)
            else:             lcolor = (0,200,80)

    # ── Model votes bars ──────────────────────────────────────
    model_y = GRAPH_Y + GRAPH_H + 20
    cv2.putText(panel, "Model votes",
                (20, model_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
    model_y += 18

    model_labels = ["KMeans", "GMM   ", "IsoFor", "SVM   "]
    for idx, label in enumerate(model_labels):
        y = model_y + idx * 22
        voted = idx < votes   # simple visual — first N models voted closed

        color = (0,0,255) if voted else (0,180,80)
        text  = "CLOSED" if voted else "OPEN  "

        cv2.putText(panel, f"{label} [{text}]",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    color, 1, cv2.LINE_AA)

        # bar
        bar_w = 80 if voted else 40
        cv2.rectangle(panel, (220, y-10), (220+bar_w, y+2), color, -1)

    # ── Calibration progress ──────────────────────────────────
    if state == "CALIBRATING":
        prog   = min(frame_count / calib_n, 1.0)
        prog_y = PANEL_H - 40
        cv2.putText(panel, f"Calibrating... {int(prog*100)}%",
                    (20, prog_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
        cv2.rectangle(panel, (20, prog_y+8), (PANEL_W-20, prog_y+20),
                      (50,50,50), -1)
        cv2.rectangle(panel, (20, prog_y+8),
                      (20 + int((PANEL_W-40)*prog), prog_y+20),
                      (0,200,80), -1)
    else:
        cv2.putText(panel, f"Frame: {frame_count}",
                    (20, PANEL_H-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1)

    return panel


# ─────────────────────────────────────────────────────────────
# STEP 5 — PROCESSOR
# ─────────────────────────────────────────────────────────────
processor = DriverStateProcessor()


# ─────────────────────────────────────────────────────────────
# STEP 6 — MAIN LOOP
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("Running — press Q to quit.")

with FaceLandmarker.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        h, w  = frame.shape[:2]
        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_img, ts_ms)

        ear_l_val = 0.0
        ear_r_val = 0.0
        cnn_l_val = 0.5
        cnn_r_val = 0.5

        if result.face_landmarks:
            lms = result.face_landmarks[0]

            # draw landmarks
            draw_eye_sharp(frame, lms, LEFT_EYE,  (0, 255, 80),  h, w)
            draw_eye_sharp(frame, lms, RIGHT_EYE, (0, 220, 255), h, w)

            # compute EAR
            ear_l_val = compute_ear(lms, LEFT_EAR_PTS,  h, w)
            ear_r_val = compute_ear(lms, RIGHT_EAR_PTS, h, w)

            # get eye crops and CNN prediction
            l_crop, (lx1,ly1,lx2,ly2) = get_eye_crop(frame, lms, LEFT_EYE,  h, w)
            r_crop, (rx1,ry1,rx2,ry2) = get_eye_crop(frame, lms, RIGHT_EYE, h, w)

            cnn_l_val = predict_eye(l_crop) if l_crop.size > 0 else 0.5
            cnn_r_val = predict_eye(r_crop) if r_crop.size > 0 else 0.5

            # hybrid EAR + CNN decision
            l_state, l_src = final_state(cnn_l_val, ear_l_val)
            r_state, r_src = final_state(cnn_r_val, ear_r_val)

            l_color = (0,255,80)  if l_state == "OPEN" else (0,0,255)
            r_color = (0,220,255) if r_state == "OPEN" else (0,0,255)

            cv2.rectangle(frame, (lx1,ly1), (lx2,ly2), l_color, 1)
            cv2.rectangle(frame, (rx1,ry1), (rx2,ry2), r_color, 1)

            draw_label(frame, f"L:{l_state} {ear_l_val:.2f}[{l_src}]", (lx1, ly1-5), l_color)
            draw_label(frame, f"R:{r_state} {ear_r_val:.2f}[{r_src}]", (rx1, ry1-5), r_color)

        else:
            draw_label(frame, "No face detected", (10, 30), (0,0,255))

        # ── update processor every frame ──────────────────────
        processor.update(ear_l_val, ear_r_val, cnn_l_val, cnn_r_val)
        state   = processor.get_state()
        perclos = processor.get_perclos()
        votes   = processor.get_votes()

        # ── draw graph panel ──────────────────────────────────
        panel = draw_graph_panel(state, perclos, votes, processor.frame_count)

        # ── combine webcam + panel side by side ───────────────
        combined = np.hstack([frame, panel])

        cv2.imshow("Driver Monitor", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
