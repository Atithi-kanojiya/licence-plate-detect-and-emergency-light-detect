import cv2
import pytesseract
import numpy as np
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

red1 = (np.array([0, 70, 50]), np.array([10, 255, 255]))
red2 = (np.array([160, 70, 50]), np.array([180, 255, 255]))
blue = (np.array([90, 50, 50]), np.array([140, 255, 255]))

def extract_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return pytesseract.image_to_string(binary, config=config).strip()

def correct_rotation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    angle = 0.0
    if lines is not None:
        for rho, theta in lines[0]:
            angle = np.rad2deg(theta) - 90
            break

    rows, cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (cols, rows))

    return rotated_img

# Function to detect emergency lights in the frame
def detect_emergency(frame):
    roi = frame[:int(frame.shape[0]*0.3), :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = sum([cv2.inRange(hsv, *color) for color in [red1, red2, blue]])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c) > 250 for c in contours)

# Function to detect license plate and emergency light
def detect_license_plate_and_emergency_light(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in plates:
        plate_img = frame[y:y+h, x:x+w]

        plate_img = correct_rotation(plate_img)

        plate = extract_text(plate_img)

        if plate:
            now = datetime.now()
            timestamp = now

            status = "Emergency Vehicle" if detect_emergency(frame) else "Normal Vehicle"
            return plate, status, timestamp

    return None, None, None
