import os
import json
from PIL import Image
from ultralytics import YOLO 

# Config
MODEL_PATH = "runs/train/piece_yolov8/weights/best.pt"
INPUT_JSON = "input.json"
OUTPUT_FILE = "output.json"

# Get image file paths from input JSON
def get_image_list(json_path):
    with open(json_path, "r") as f:
        input_data = json.load(f)
    return [os.path.join(os.path.dirname(json_path), path) for path in input_data["image_files"]]

# Load YOLO model
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Run inference and save results to JSON
def generate_json(model, image_paths):
    results = []
    for img_path in image_paths:
        # Run inference
        detections = model(img_path)[0]  # First result (YOLO returns a list)

        # Count number of detected pieces (number of bounding boxes)
        num_pieces = len(detections.boxes)

        results.append({
            "image": img_path,
            "num_pieces": num_pieces,
        })

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✅ Results saved to {OUTPUT_FILE}")

# Main entry point
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"❌ Input JSON not found: {INPUT_JSON}")

    model = load_model(MODEL_PATH)
    image_paths = get_image_list(INPUT_JSON)
    generate_json(model, image_paths)

if __name__ == "__main__":
    main()
