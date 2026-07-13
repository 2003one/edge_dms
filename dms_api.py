from flask import Flask, jsonify
import threading
import cv2
import mediapipe as mp
import numpy as np
import os
from ai_edge_litert.interpreter import Interpreter
from processor import DriverStateProcessor

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"

app = Flask(__name__)

# ── shared state ──────────────────────────────────────────────
current_state   = {"state": "CALIBRATING", "perclos": 0.0, "votes": 0}
state_lock      = threading.Lock()

# ── load model ────────────────────────────────────────────────
interpreter = Interpreter(model_path="eye_model.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_eye(crop):
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32))
    norm    = resized.astype(np.float32) / 255.0
    tensor  = norm.reshape(1, 32, 32, 1)
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke()
    return float(interpreter.get_tensor(output_details[0]['index'])[0][0])
    
# ── V2 model ──────────────────────────────────────────────────
interpreter_v2 = Interpreter(model_path="eye_model_v2.tflite")
interpreter_v2.allocate_tensors()
input_details_v2  = interpreter_v2.get_input_details()
output_details_v2 = interpreter_v2.get_output_details()

def predict_eye_v2(crop):
    resized = cv2.resize(crop, (96, 96))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm    = (rgb.astype(np.float32) / 127.5) - 1.0
    tensor  = norm.reshape(1, 96, 96, 3)
    interpreter_v2.set_tensor(input_details_v2[0]['index'], tensor)
    interpreter_v2.invoke()
    return float(interpreter_v2.get_tensor(
        output_details_v2[0]['index'])[0][0])

def cascade_predict(crop, ear_val):
    prob_v1 = predict_eye(crop)

    # V2 triggers on ambiguous V1 or EAR/CNN disagreement
    if 0.3 < prob_v1 < 0.7:
        prob_v2 = predict_eye_v2(crop)
        return (prob_v1 + prob_v2) / 2
    elif ear_val < 0.20 and prob_v1 > 0.5:
        prob_v2 = predict_eye_v2(crop)
        return prob_v2
    else:
        return prob_v1

# ── API endpoint ──────────────────────────────────────────────
@app.route('/state', methods=['GET'])
def get_state():
    with state_lock:
        return jsonify(current_state)

# ── DMS thread — runs realtime inference ──────────────────────
def dms_thread():
    global current_state

    BaseOptions    = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FLOptions      = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode    = mp.tasks.vision.RunningMode

    LEFT_EAR_PTS  = [362, 385, 387, 263, 373, 380]
    RIGHT_EAR_PTS = [33,  160, 158, 133, 153, 144]
    LEFT_EYE      = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
    RIGHT_EYE     = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]

    def compute_ear(lms, pts, h, w):
        def p(i):
            lm = lms[pts[i]]
            return np.array([lm.x * w, lm.y * h])
        p1,p2,p3,p4,p5,p6 = [p(i) for i in range(6)]
        return round((np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)) / (2.0 * np.linalg.norm(p1-p4)), 3)

    def get_crop(frame, lms, indices, h, w, pad=12):
        xs = [int(lms[i].x * w) for i in indices]
        ys = [int(lms[i].y * h) for i in indices]
        x1 = max(0, min(xs)-pad); y1 = max(0, min(ys)-pad)
        x2 = min(w, max(xs)+pad); y2 = min(h, max(ys)+pad)
        return frame[y1:y2, x1:x2]

    options = FLOptions(
        base_options=BaseOptions(
            model_asset_path="face_landmarker.task",
            delegate=BaseOptions.Delegate.CPU),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5)

    processor = DriverStateProcessor()
    cap       = cv2.VideoCapture("driver2.webm")

    with FaceLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame  = cv2.resize(frame, (640, 480))
            h, w   = frame.shape[:2]
            ts_ms  = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_img, ts_ms)

            ear_l = ear_r = 0.0
            cnn_l = cnn_r = 0.5

            if result.face_landmarks:
                lms   = result.face_landmarks[0]
                ear_l = compute_ear(lms, LEFT_EAR_PTS,  h, w)
                ear_r = compute_ear(lms, RIGHT_EAR_PTS, h, w)
                l_crop = get_crop(frame, lms, LEFT_EYE,  h, w)
                r_crop = get_crop(frame, lms, RIGHT_EYE, h, w)
                if l_crop.size > 0: cnn_l = cascade_predict(l_crop, ear_l)
                if r_crop.size > 0: cnn_r = cascade_predict(r_crop, ear_r)
            processor.update(ear_l, ear_r, cnn_l, cnn_r)
            
             # ── show video with state overlay ─────────────────
            state   = processor.get_state()
            perclos = processor.get_perclos()

            color = (0,255,0) if state=="ACTIVE" else \
                    (0,165,255) if state=="DROWSY" else (0,0,255)

            cv2.putText(frame, f"{state} | PERCLOS:{perclos:.1f}%",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)
            cv2.putText(frame, f"EAR L:{ear_l:.2f} R:{ear_r:.2f}",
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 1)
            cv2.putText(frame, f"CNN L:{cnn_l:.2f} R:{cnn_r:.2f}",
                        (10,85), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 1)

            cv2.imshow("DMS Debug", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
                
            with state_lock:
                current_state = {
                    "state":   processor.get_state(),
                    "perclos": round(processor.get_perclos(), 1),
                    "votes":   processor.get_votes()
                }

if __name__ == '__main__':
    t = threading.Thread(target=dms_thread, daemon=True)
    t.start()
    print("DMS API running on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
