# utils.py
import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from config import MODEL_NAME, DEVICE
import torch.nn.functional as F


# ✅ 1. 이미지 로드 (PIL + RGB)
def load_image_from_path(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


# ✅ 2. 기본 모델 inference 수행
def run_inference(image_path, return_tensor=False):
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(DEVICE)
    
    image = load_image_from_path(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]["predicted_depth"].clamp(min=0)  # (1, H, W)

    if return_tensor:
        return post_processed.squeeze(0), image, inputs
    else:
        return post_processed.squeeze().cpu().numpy()


# ✅ 3. bbox json 파일 로드
def load_bbox_json(json_path, exclude=[]):
    with open(json_path, 'r') as f:
        data = json.load(f)

    bboxes = []
    for ann in data.get("annotations", []):
        if ann.get("category", "").lower() in exclude:
            continue

        x, y, w, h = ann["bbox"]
        bboxes.append([int(x), int(y), int(x + w), int(y + h)])  # [x1, y1, x2, y2]

    return bboxes

# ✅ 4. bbox → soft mask 생성
def generate_soft_mask_from_bbox(bboxes, image_shape, margin=50, alpha=0.5):
    """
    bboxes: List of [x1, y1, x2, y2]
    image_shape: (H, W)
    return: torch.Tensor of shape [1, 1, H, W]
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.float32)

    for x1, y1, x2, y2 in bboxes:
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(W, x2 + margin)
        y2 = min(H, y2 + margin)
        mask[y1:y2, x1:x2] += 1.0  # 중복 영역은 점점 weight 증가

    mask = mask / (mask.max() + 1e-6)  # normalize to [0, 1]
    mask = (mask * alpha).astype(np.float32)
    return torch.from_numpy(mask)[None, None, :, :]  # shape: [1, 1, H, W]


# ✅ 5. depth 맵 저장 (정규화 + PNG)
def save_depth_map(depth_map, save_path):
    depth_map = np.asarray(depth_map)
    norm_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    vis = (norm_depth * 255).astype(np.uint8)
    cv2.imwrite(save_path, vis)
