import requests
import time
from gpiozero import Buzzer      # controls the buzzer pin

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DMS_URL = "http://localhost:5000/state"   # realtime.py's API (same machine)
CAR_IP  = "10.190.216.223"                # your ESP car's IP
CAR_URL = f"http://{CAR_IP}/dms"

buzzer = Buzzer(18)                        # buzzer on GPIO18 (physical pin 12)

# ─────────────────────────────────────────────
# BRIDGE LOOP
# ─────────────────────────────────────────────
last_sent = None
print("=" * 45)
print("DMS -> Car bridge (with buzzer)")
print(f"  reading state from : {DMS_URL}")
print(f"  sending to car     : {CAR_URL}")
print("=" * 45)

while True:
    try:
        # 1. read drowsiness state from realtime.py
        resp  = requests.get(DMS_URL, timeout=1)
        state = resp.json().get("state", "UNKNOWN")

        # 2. BUZZER — runs every loop so it reflects the live state
        #    DROWSY or DANGER -> buzz to alert the driver
        #    anything else    -> silent
        if state in ("DROWSY", "DANGER"):
            buzzer.on()
        else:
            buzzer.off()

        # 3. CAR — only send when the state CHANGES (avoids spamming)
        if state != last_sent:
            print(f"[state change] {last_sent} -> {state}")
            try:
                requests.get(f"{CAR_URL}?s={state}", timeout=0.5)
                if state == "DANGER":
                    print("   >> sent STOP to car + BUZZER ON")
                elif state == "DROWSY":
                    print("   >> BUZZER ON (warning)")
                else:
                    print("   >> car unlocked + BUZZER OFF")
                last_sent = state
            except Exception as e:
                print(f"   !! car unreachable: {e}")

    except Exception as e:
        # realtime.py not up yet, or API not responding
        print(f"[waiting for DMS API] {e}")
        buzzer.off()      # stay silent if we lose the DMS

    time.sleep(0.3)   # check ~3 times per second
