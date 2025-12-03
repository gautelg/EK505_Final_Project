# src/detection/crack_detector.py
import cv2
import numpy as np

def detect_cracks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = frame.copy()
    boxes = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return output, edges, boxes
