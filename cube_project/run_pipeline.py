import yaml
import sys
import os

# add src to sys.path so Python can find the src package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.pipeline.run_full_pipeline import run_full_pipeline


if __name__ == "__main__":
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)

    run_full_pipeline(config)
