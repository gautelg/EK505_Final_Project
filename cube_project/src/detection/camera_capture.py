# src/detection/camera_capture.py
import cv2

def get_camera(camera_index=1):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    return cap
