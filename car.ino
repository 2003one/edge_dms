#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <Wire.h>
#include <7Semi_ICM20948.h>
#include "ICM20948_regs.h"

// ── Motor driver pins (active-LOW: LOW = that channel ON) ──
#define IN1 14       // D5
#define IN2 12       // D6
#define IN3 13       // D7
#define IN4 16       // D0

// ── IMU I2C pins ──
#define I2C_SCL 5    // D1
#define I2C_SDA 4    // D2
#define ICM_ADDR 0x69

// ── WiFi hotspot credentials ──
const char* ssid     = "Anup's A34";
const char* password = "12345678";

ESP8266WebServer server(80);     // web server on port 80
ICM20948_7Semi imu;              // IMU object

// ── IMU state ──
float gxBias = 0, gyBias = 0, gzBias = 0;   // gyro calibration offsets
bool imuOK = false;                          // true if IMU initialised
unsigned long lastImuPrint = 0;              // timer for IMU printing

// ── DMS safety lock ──
// When true, the car is force-stopped by the drowsiness system
// and ignores all phone commands until the driver is alert again.
bool dmsLock = false;


// ─────────────────────────────────────────────
// MOTOR COMMANDS  (active-LOW driver)
// ─────────────────────────────────────────────
void forward()  { digitalWrite(IN1,LOW);  digitalWrite(IN2,HIGH); digitalWrite(IN3,LOW);  digitalWrite(IN4,HIGH); }
void backward() { digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW);  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW);  }
void left()     { digitalWrite(IN1,LOW);  digitalWrite(IN2,HIGH); digitalWrite(IN3,HIGH); digitalWrite(IN4,HIGH); }
void right()    { digitalWrite(IN1,HIGH); digitalWrite(IN2,HIGH); digitalWrite(IN3,LOW);  digitalWrite(IN4,HIGH); }
void stopMotors(){ digitalWrite(IN1,HIGH); digitalWrite(IN2,HIGH); digitalWrite(IN3,HIGH); digitalWrite(IN4,HIGH); }


// ─────────────────────────────────────────────
// WEB PAGE  (the control buttons shown on phone)
// ─────────────────────────────────────────────
const char* page =
"<!DOCTYPE html><html><head>"
"<meta name='viewport' content='width=device-width,initial-scale=1'>"
"<style>"
"body{font-family:sans-serif;text-align:center;background:#111;color:#fff;user-select:none}"
"button{width:110px;height:110px;margin:8px;font-size:20px;border:none;border-radius:16px;background:#2a6;color:#fff}"
"button:active{background:#7d4}"
"#stop{background:#c33}"
"</style></head><body>"
"<h2>Car Control</h2>"
"<div><button onmousedown=\"s('F')\" onmouseup=\"s('S')\" ontouchstart=\"s('F')\" ontouchend=\"s('S')\">FWD</button></div>"
"<div>"
"<button onmousedown=\"s('L')\" onmouseup=\"s('S')\" ontouchstart=\"s('L')\" ontouchend=\"s('S')\">LEFT</button>"
"<button id='stop' onclick=\"s('S')\">STOP</button>"
"<button onmousedown=\"s('R')\" onmouseup=\"s('S')\" ontouchstart=\"s('R')\" ontouchend=\"s('S')\">RIGHT</button>"
"</div>"
"<div><button onmousedown=\"s('B')\" onmouseup=\"s('S')\" ontouchstart=\"s('B')\" ontouchend=\"s('S')\">BACK</button></div>"
"<script>function s(c){fetch('/cmd?d='+c);}</script>"
"</body></html>";


// ─────────────────────────────────────────────
// WEB HANDLERS
// ─────────────────────────────────────────────

// Serves the control page when phone opens the car's IP
void handleRoot() {
    server.send(200, "text/html", page);
}

// Handles drive commands from the phone (/cmd?d=F etc.)
void handleCmd() {
    // SAFETY: if the DMS has locked the car (driver in DANGER),
    // ignore the phone and keep the car stopped.
    if (dmsLock) {
        stopMotors();
        server.send(200, "text/plain", "locked");
        return;
    }

    // Normal driving — act on the command letter
    char c = server.arg("d")[0];
    if      (c=='F') forward();
    else if (c=='B') backward();
    else if (c=='L') left();
    else if (c=='R') right();
    else             stopMotors();

    server.send(200, "text/plain", "ok");
}

// Handles drowsiness state from the Pi bridge (/dms?s=DANGER etc.)
void handleDMS() {
    String s = server.arg("s");

    if (s == "DANGER") {
        // Driver asleep — lock and stop the car
        dmsLock = true;
        stopMotors();
    } else {
        // Driver alert (ACTIVE/DROWSY) — release the lock,
        // phone driving allowed again
        dmsLock = false;
    }

    server.send(200, "text/plain", "ok");
}


// ─────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(2000);

    // Motors OFF before making pins outputs (active-LOW, avoids boot twitch)
    digitalWrite(IN1,HIGH); digitalWrite(IN2,HIGH);
    digitalWrite(IN3,HIGH); digitalWrite(IN4,HIGH);
    pinMode(IN1,OUTPUT); pinMode(IN2,OUTPUT);
    pinMode(IN3,OUTPUT); pinMode(IN4,OUTPUT);

    // Start IMU and calibrate gyro (keep car still during this)
    Wire.begin(I2C_SDA, I2C_SCL);
    if (imu.begin(Wire, ICM_ADDR) && imu.applyBasicDefaults()) {
        imuOK = true;
        calibrateGyro();
    }

    // Connect to the hotspot
    WiFi.begin(ssid, password);
    Serial.print("Connecting");
    while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
    Serial.println();
    Serial.print("Open this on your phone: http://");
    Serial.println(WiFi.localIP());   // <-- this IP goes in dms_bridge.py

    // Register all web routes
    server.on("/",    handleRoot);    // control page
    server.on("/cmd", handleCmd);     // phone driving
    server.on("/dms", handleDMS);     // drowsiness stop from Pi
    server.begin();
}


// ─────────────────────────────────────────────
// MAIN LOOP
// ─────────────────────────────────────────────
void loop() {
    server.handleClient();            // handle incoming web requests

    // Print IMU data ~10x/sec (non-blocking)
    if (imuOK && millis() - lastImuPrint >= 100) {
        lastImuPrint = millis();
        printImu();
    }
}


// ─────────────────────────────────────────────
// IMU — print bias-corrected readings
// ─────────────────────────────────────────────
void printImu() {
    float ax,ay,az,gx,gy,gz;
    if (imu.readAccel(ax,ay,az) && imu.readGyro(gx,gy,gz)) {
        gx-=gxBias; gy-=gyBias; gz-=gzBias;   // remove calibration offset
        Serial.print("ax:");Serial.print(ax,3);
        Serial.print(" gz:");Serial.print(gz,2);
        Serial.println();
    }
}


// ─────────────────────────────────────────────
// IMU — measure gyro bias at startup (car must be STILL)
// ─────────────────────────────────────────────
void calibrateGyro() {
    float sx=0,sy=0,sz=0; int n=0;
    for(int i=0;i<200;i++){
        float gx,gy,gz;
        if(imu.readGyro(gx,gy,gz)){ sx+=gx;sy+=gy;sz+=gz;n++; }
        delay(5);
    }
    if(n>0){ gxBias=sx/n; gyBias=sy/n; gzBias=sz/n; }
}