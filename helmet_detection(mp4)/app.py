from flask import Flask, render_template, Response, request
import cv2
import math
import cvzone
from ultralytics import YOLO
import time

app = Flask(__name__)

model = YOLO("D:/Helmet_Detect___/best.pt")

classNames = ['With Helmet', 'Without Helmet']

new_width = 1024
new_height = 576

is_paused = False

def generate_frames():
    global is_paused
    video_path = "D:/Helmet_Detect___/traffic.mp4"
    cap = cv2.VideoCapture(video_path)

    while True:
        if is_paused:
            time.sleep(0.1)
            continue

        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (new_width, new_height))

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=color)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control():
    global is_paused
    action = request.form.get('action')
    if action == 'play':
        is_paused = False
    elif action == 'pause':
        is_paused = True
    return '', 204

if __name__ == "__main__":
    app.run(debug=True)
