from collections import deque
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
BUFFER_SIZE   = 900    # 30 seconds at 30fps
CALIBRATION_N = 300    # first 10 seconds = calibration phase
UPDATE_EVERY  = 30     # re-run models every 30 frames (1 second)


class DriverStateProcessor:

    def __init__(self):
        self.buffer           = deque(maxlen=BUFFER_SIZE)
        self.eye_state_buffer = deque(maxlen=900)   # 30 seconds
        self.frame_count      = 0
        self.calibrated       = False
        self.state            = "CALIBRATING"
        self.kmeans           = MiniBatchKMeans(n_clusters=2, random_state=42)
        self.gmm              = None
        self.iso              = None
        self.svm              = None
        self.threshold        = 0.20
        self.last_perclos     = 0.0
        self.last_votes       = 0

    # ─────────────────────────────────────────────────────────
    # UPDATE — called every frame from realtime.py
    # ─────────────────────────────────────────────────────────
    def update(self, ear_l, ear_r, cnn_l, cnn_r):

        # build feature vector and push to buffer
        feature = [ear_l, ear_r, cnn_l, cnn_r]
        self.buffer.append(feature)
        self.frame_count += 1

        # update KMeans every frame incrementally
        if self.frame_count >= 2:
            self.kmeans.partial_fit(np.array(list(self.buffer)[-2:]))
            ear_centers = sorted(self.kmeans.cluster_centers_[:, 0])
            gap = ear_centers[1] - ear_centers[0]
            if gap > 0.05:
                self.threshold = (ear_centers[0] + ear_centers[1]) / 2

        # push eye state every frame for PERCLOS
        avg_ear  = (ear_l + ear_r) / 2
        eye_open = 1 if avg_ear > self.threshold else 0
        self.eye_state_buffer.append(eye_open)

        # at frame 300 — calibrate all models
        if self.frame_count == CALIBRATION_N:
            self._calibrate()

        # every 30 frames — compute driver state
        if self.calibrated and self.frame_count % UPDATE_EVERY == 0:
            self.state = self._compute_state()

        return self.state

    # ─────────────────────────────────────────────────────────
    # CALIBRATE — runs once at frame 300
    # fits GMM, Isolation Forest, One-Class SVM
    # ─────────────────────────────────────────────────────────
    def _calibrate(self):
        data = np.array(self.buffer)

        self.gmm = GaussianMixture(n_components=2, random_state=42)
        self.gmm.fit(data)
        gmm_means           = self.gmm.means_[:, 0]
        self.gmm_closed_idx = int(np.argmin(gmm_means))
        self.gmm_open_idx   = int(np.argmax(gmm_means))

        self.iso = IsolationForest(contamination=0.05, random_state=42)
        self.iso.fit(data)

        self.svm = OneClassSVM(kernel='rbf', nu=0.05)
        self.svm.fit(data)

        self.calibrated = True

        # ── reset PERCLOS buffer ──────────────────────────────────
        # calibration used wrong threshold → clear bad data
        # PERCLOS starts fresh from 0% after calibration
        self.eye_state_buffer.clear()

        print("Calibration complete — monitoring started")
    # ─────────────────────────────────────────────────────────
    # PERCLOS — percentage of time eyes were closed
    # ─────────────────────────────────────────────────────────
    def _compute_perclos(self):
        if len(self.eye_state_buffer) < 30:
            return 0.0
        total  = len(self.eye_state_buffer)
        closed = sum(1 for s in self.eye_state_buffer if s == 0)
        return (closed / total) * 100

    # ─────────────────────────────────────────────────────────
    # COMPUTE STATE — runs every second after calibration
    # all 4 models vote → final driver state
    # ─────────────────────────────────────────────────────────
    def _compute_state(self):
            latest  = np.array(self.buffer[-1])
            ear_l   = latest[0]
            ear_r   = latest[1]
            avg_ear = (ear_l + ear_r) / 2

            # Model 1 — KMeans
            kmeans_state = "OPEN" if avg_ear > self.threshold else "CLOSED"

            # Model 2 — GMM
            probs      = self.gmm.predict_proba([latest])[0]
            open_prob  = probs[self.gmm_open_idx]
            gmm_state  = "OPEN" if open_prob > 0.5 else "CLOSED"

            # Model 3 — Isolation Forest
            iso_result = self.iso.predict([latest])[0]
            iso_state  = "NORMAL" if iso_result == 1 else "ANOMALY"

            # Model 4 — One-Class SVM
            svm_result = self.svm.predict([latest])[0]
            svm_state  = "NORMAL" if svm_result == 1 else "ANOMALY"

            # PERCLOS — computed after all models
            perclos = self._compute_perclos()

            # ── WEIGHTED VOTES ────────────────────────────────────────
            # Anomaly models (IsoForest + SVM) → weight 2 (more reliable)
            # Cluster models (KMeans + GMM)    → weight 1 (weaker)
            # Max possible score = 6
            closed_score = 0
            if kmeans_state == "CLOSED":  closed_score += 1
            if gmm_state    == "CLOSED":  closed_score += 1
            if iso_state    == "ANOMALY": closed_score += 2
            if svm_state    == "ANOMALY": closed_score += 2

            # save for realtime.py to read
            self.last_perclos = perclos
            self.last_votes   = closed_score   # now reflects weighted score

            # ── FINAL DECISION ────────────────────────────────────────
            # PERCLOS uses tighter window (300 frames = 10s) → reacts fast
            # Weighted score needs anomaly models to confirm danger
            if perclos > 57:
                final_state = "DANGER"
            elif perclos > 20:
                final_state = "DROWSY"
            elif closed_score >= 4:   # both anomaly models must agree
                final_state = "DANGER"
            elif closed_score >= 2:   # at least one anomaly or both clusters
                final_state = "DROWSY"
            else:
                final_state = "ACTIVE"

            return final_state

    # ─────────────────────────────────────────────────────────
    # GETTER — realtime.py reads these every frame
    # ─────────────────────────────────────────────────────────
    def get_state(self):
        return self.state

    def get_perclos(self):
        return self.last_perclos

    def get_votes(self):
        return self.last_votes
