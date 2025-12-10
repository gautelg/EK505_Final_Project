# src/pipeline/run_detection.py

import cv2
from src.detection.camera_capture import get_camera
from src.detection.crack_detector import detect_cracks
from src.detection.model_loader import load_model

def run_detection(config, path):
    print("[DETECTION] Starting anomaly detection...")

    model = load_model()

    camera_index = config["detection"].get("camera_index", 1)
    cap = get_camera(camera_index)

    save_output = config["detection"].get("save_output", True)
    save_path = "data/outputs/detection_results.avi"

    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, edges, boxes = detect_cracks(frame)

        cv2.imshow("Detection", annotated)
        cv2.imshow("Edges", edges)

        if save_output:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(save_path, fourcc, 20.0, (w, h))
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[DETECTION] Finished.")
