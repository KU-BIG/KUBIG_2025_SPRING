import torch
import numpy as np
from PIL import Image
from transformers import (pipeline, AutoImageProcessor, AutoModelForDepthEstimation)
from config import MODEL_NAME, DEVICE
from utils import load_image_from_path
import cv2
from ultralytics import YOLO


# 1. 모델 준비
clip_classifier = pipeline(
    task="zero-shot-image-classification",
    model="openai/clip-vit-large-patch14",
    device=0 if torch.cuda.is_available() else -1
)
important_labels = ["car", "bus", "person", "bicycle", "motorcycle", "traffic sign"]


seg_model = YOLO("yolov8n-seg.pt")

def run_light_segmentation(crop_image_pil):
    results = seg_model.predict(source=np.array(crop_image_pil), verbose=False)
    if results[0].masks is None:
        return None
    mask = results[0].masks.data[0].cpu().numpy()  # 첫 객체 mask
    return mask

def create_feathered_mask(h, w, feather=20):
    mask = np.ones((h, w), dtype=np.float32)
    for i in range(feather):
        ratio = i / feather
        mask[i, :] *= ratio
        mask[-(i + 1), :] *= ratio
        mask[:, i] *= ratio
        mask[:, -(i + 1)] *= ratio
    return mask


# 2. depth 모델 준비
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
depth_model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(DEVICE)
depth_model.eval()

def run_patch_inference_on_important_objects(model, processor, image_pil, original_depth_map, important_bboxes, 
                                             margin=100):
    refined_depth_map = original_depth_map.copy()

    for obj in important_bboxes:
        x1, y1, x2, y2 = obj["bbox"]
        original_bbox_depth = obj["original_depth"]
        h0, w0 = y2 - y1, x2 - x1
        
        # 확장 bbox
        x1e, y1e = max(x1 - margin, 0), max(y1 - margin, 0)
        x2e, y2e = min(x2 + margin, image_pil.width), min(y2 + margin, image_pil.height)

        cropped = image_pil.crop((x1e, y1e, x2e, y2e))
        
        # segmentation 먼저 수행
        mask = run_light_segmentation(cropped)
        if mask is None:
            continue
        
        h, w = (y2e - y1e), (x2e - x1e)
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        mask_resized /= mask_resized.max() + 1e-6
        
        # depth estimation
        inputs = processor(images=cropped, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        target_size = [(cropped.size[1], cropped.size[0])]
        pred = processor.post_process_depth_estimation(outputs, target_size)[0]
        patch_depth = pred['predicted_depth'].squeeze().cpu().numpy()

        patch_depth_resized = cv2.resize(patch_depth, (w, h), interpolation=cv2.INTER_LINEAR)
        
        patch_bbox_depth = patch_depth_resized[(y1 - y1e):(y2 - y1e), (x1 - x1e):(x2 - x1e)]
        mask_bbox_resized = mask_resized[(y1 - y1e):(y2 - y1e), (x1 - x1e):(x2 - x1e)]
        
        # depth ratio 기반 global weight (추가)
        orig_median = np.median(original_bbox_depth)
        patch_median = np.median(patch_bbox_depth)
        depth_ratio = orig_median / (patch_median + 1e-6)
        patch_weight_global = np.clip(depth_ratio / (1 + depth_ratio), 0.1, 0.9)
        
        # local confidence (gradient variance) : 추가
        gy_p, gx_p = np.gradient(patch_bbox_depth)
        patch_var = gx_p**2 + gy_p**2
        conf_patch = np.exp(-patch_var / (np.mean(patch_var) + 1e-6))
        
        # 최종 weight
        patch_weight = patch_weight_global * conf_patch
        orig_weight = 1.0 - patch_weight
        
        # blending
        combined_depth = patch_weight * patch_bbox_depth + orig_weight * original_bbox_depth

        # patch와 original depth average or weighted sum
        #combined_depth = 0.5 * patch_bbox_depth + 0.5 * original_bbox_depth  # weight는 조절 가능
        
        # polygon mask binary
        polygon_mask_binary = (mask_bbox_resized > 0.5).astype(np.float32)
        
        # refiined depth map update (polygon 내부만)
        refined_depth_map[y1:y2, x1:x2] = (
            polygon_mask_binary * combined_depth +
            (1 - polygon_mask_binary) * refined_depth_map[y1:y2, x1:x2]
        )
        
    return refined_depth_map
