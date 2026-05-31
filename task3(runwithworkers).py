import os
import json
import shutil
import random
from pathlib import Path
from pprint import pprint

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import models
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as transforms
from torch.amp import autocast, GradScaler




random.seed(42)

data_aug = transforms.Compose([
    transforms.ToImage(),    
    transforms.Resize((256, 256)),
    #transforms.CenterCrop((224, 224)),
    #transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

data_in = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((256, 256)),
    #transforms.CenterCrop((224, 224)),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# get cpu or gpu device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

scaler = GradScaler(device=device)

class CocoBoardDataset(Dataset):
    def __init__(self, root_dir, partition, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(os.path.join(root_dir, 'annotations.json'), 'r') as f:
            anns = json.load(f)

        self.images = anns['images']
        self.corners = anns['annotations']['corners']
        self.split_ids = anns['splits']['chessred2k'][partition]['image_ids']

        # Filter images to current split
        self.images = [img for img in self.images if img['id'] in self.split_ids]
        self.image_id_to_path = {img['id']: img['path'] for img in self.images}

        # Group corners by image_id
        self.corners_by_image = {}
        for ann in self.corners:
            self.corners_by_image[ann['image_id']] = ann['corners']

        self.image_ids = list(self.image_id_to_path.keys())

        print(f"Number of {partition} images (board masks): {len(self.image_ids)}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, self.image_id_to_path[img_id])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]

        image = image.astype(np.float32) / 255.0

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image).permute(2, 0, 1).float()

        corners_dict = self.corners_by_image[img_id]

        polygon = np.array([
            corners_dict['top_left'],
            corners_dict['top_right'],
            corners_dict['bottom_right'],
            corners_dict['bottom_left']
        ], dtype=np.int32)

        # Generate mask
        mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)  # Mask will have values 0 or 1
        mask = cv2.resize(mask, (image.shape[-1], image.shape[-2]), interpolation=cv2.INTER_NEAREST)

        x_min = np.min(polygon[:, 0])
        y_min = np.min(polygon[:, 1])
        x_max = np.max(polygon[:, 0])
        y_max = np.max(polygon[:, 1])

        boxes = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)  # 1 class: board
        masks = torch.tensor(mask[None, ...], dtype=torch.uint8)  # Shape (1, H, W)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id])
        }

        return image, target


# === IoU computation function ===
def compute_iou(pred_mask, target_mask):
    intersection = (pred_mask & target_mask).float().sum()
    union = (pred_mask | target_mask).float().sum()
    iou = intersection / union if union > 0 else 0.0
    return iou.item()

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    print("Running the script...")

    root_dir = ''
    batch_size = 2
    num_workers = 4
    print(f"Using device: {device}")

    train_dataset_board = CocoBoardDataset(root_dir, partition='train', transform=data_aug)
    valid_dataset_board = CocoBoardDataset(root_dir, partition='val', transform=data_in)
    test_dataset_board = CocoBoardDataset(root_dir, partition='test', transform=data_in)

    train_dataloader_board = DataLoader(train_dataset_board, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    valid_dataloader_board = DataLoader(valid_dataset_board, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_dataloader_board = DataLoader(test_dataset_board, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, 2)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2.5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # === Training loop
    num_epochs = 50
    patience = 5
    epochs_no_improvement = 0

    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        ##########################
        # TRAIN LOOP
        ##########################
        model.train()
        running_loss = 0.0
        
        for images, targets in tqdm(train_dataloader_board, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with autocast(device_type=device.type):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(images)

        epoch_train_loss = running_loss / len(train_dataloader_board.dataset)

        ##########################
        # VALIDATION LOOP
        ##########################
        model.train()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, targets in tqdm(valid_dataloader_board, desc=f"Epoch {epoch+1}/{num_epochs} [Validation - Loss]"):
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Get loss_dict → must be in train mode
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

                val_loss += loss.item() * len(images)

        # Second step → now switch to eval to get predictions for IoU
        model.eval()
        with torch.no_grad():
            for images, targets in tqdm(valid_dataloader_board, desc=f"Epoch {epoch+1}/{num_epochs} [Validation - IoU]"):
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                preds = model(images)

                batch_ious = []
                for pred, target in zip(preds, targets):
                    if len(pred["masks"]) == 0:
                        continue
                    # Optional: use largest mask if multiple
                    if len(pred["masks"]) > 1:
                        areas = [(m[0] > 0.5).sum().item() for m in pred["masks"]]
                        max_idx = np.argmax(areas)
                        pred_mask = (pred["masks"][max_idx, 0] > 0.5).cpu().int()
                    else:
                        pred_mask = (pred["masks"][0, 0] > 0.5).cpu().int()

                    gt_mask = (target["masks"][0] > 0.5).cpu().int()
                    iou = compute_iou(pred_mask, gt_mask)
                    batch_ious.append(iou)

                if len(batch_ious) > 0:
                    val_iou += np.mean(batch_ious) * len(images)

        epoch_val_loss = val_loss / len(valid_dataloader_board.dataset)
        epoch_val_iou = val_iou / len(valid_dataloader_board.dataset)

        scheduler.step(epoch_val_loss)

        ##########################
        # Save metrics
        ##########################
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_ious.append(epoch_val_iou)

        ##########################
        # Save best model
        ##########################
        if epoch_val_loss < best_val_loss:
            print(f"🔥 New best model found at epoch {epoch+1} with validation loss {epoch_val_loss:.6f} - saving model.")
            best_val_loss = epoch_val_loss
            os.makedirs("board_model", exist_ok=True)
            torch.save(model.state_dict(), f"board_model/maskrcnn_best.pth")

            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= patience:
                print(f"⏳ Early stopping triggered after {patience} epochs without improvement.")
                break

        ##########################
        # Print epoch summary
        ##########################
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Train Loss: {epoch_train_loss:.6f} - "
            f"Val Loss: {epoch_val_loss:.6f} - Val IoU: {epoch_val_iou:.6f} - "
            f"LR: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

    # === Save metrics
    os.makedirs("metrics_board", exist_ok=True)
    np.save("metrics_board/train_losses.npy", np.array(train_losses))
    np.save("metrics_board/val_losses.npy", np.array(val_losses))
    np.save("metrics_board/val_ious.npy", np.array(val_ious))

    print("✅ Metrics saved to 'metrics_board' folder.")
