# camera_crack_detection.py
import cv2
import numpy as np

# Camera setup
CAM_INDEX = 1  # Logitech webcam index
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Cannot open camera {CAM_INDEX}")
    exit()

print("Press 'q' to quit.")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Find contours for cracks/dents
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours / bounding boxes
    output_frame = frame.copy()
    for cnt in contours:
        # Optional: filter by area to remove small noise
        if cv2.contourArea(cnt) < 50:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show frames
    cv2.imshow("Crack Detection", output_frame)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
