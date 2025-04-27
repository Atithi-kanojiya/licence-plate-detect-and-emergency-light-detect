from flask import Flask, render_template, jsonify, Response
import mysql.connector
import cv2
from datetime import datetime
from detecti import detect_license_plate_and_emergency_light  # This imports your function

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('front.html')  # Correct path to your HTML

@app.route('/start_detection', methods=['GET'])
def start_detection():
    # Connect to the database
    db_connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="a123",
        database="alpr"
    )
    cursor = db_connection.cursor()

    cap = cv2.VideoCapture(0)

    def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Call your detect function from detecti.py
            plate, status, timestamp = detect_license_plate_and_emergency_light(frame)

            # Insert into the database if plate detected
            if plate:
                query = "INSERT INTO vehicle_log (license_plate, date, time, status) VALUES (%s, %s, %s, %s)"
                data = (plate, timestamp.date(), timestamp.time(), status)
                cursor.execute(query, data)
                db_connection.commit()

            # Send result back to browser as byte stream
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
