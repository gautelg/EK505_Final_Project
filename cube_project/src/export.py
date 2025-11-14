import json
import os
import logging

def save_viewpoints_json(filename, viewpoints, path):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = {
        "viewpoints": viewpoints.tolist(),
        "path": path
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Saved viewpoints and path to {filename}")
