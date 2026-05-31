import os
import json
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

# Config
MODEL_PATH = "best_model.pth"
INPUT_JSON = "input.json"
OUTPUT_FILE = "output.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get image file paths from input JSON
def get_image_list(json_path):
    with open(json_path, "r") as f:
        input_data = json.load(f)
    return [os.path.join(os.path.dirname(json_path), path) for path in input_data["image_files"]]

# Image preprocessing
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Shape: (1, C, H, W)

# Scaled sigmoid layer to limit the output range
class ScaledSigmoid(nn.Module):
    def __init__(self, scale=32.0):
        super(ScaledSigmoid, self).__init__()
        self.scale = scale

    def forward(self, x):
        return torch.sigmoid(x) * self.scale

# Load model with EfficientNetB0 backbone
def load_model(model_path):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        ScaledSigmoid(scale=32.0)
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Run inference and save results to JSON
def generate_json(model, image_paths):
    results = []
    with torch.no_grad():
        for img_path in image_paths:
            img_tensor = load_image(img_path).to(DEVICE)
            output = model(img_tensor).item()
            results.append({
                "image": img_path,
                "num_pieces": output,
            })

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
