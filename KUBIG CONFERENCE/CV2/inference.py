# inference.py
import os
import torch
import numpy as np
import cv2
from glob import glob
from config import MODEL_NAME, DEVICE
from transformers import (AutoImageProcessor, AutoModelForDepthEstimation,
                          SamModel, SamImageProcessor)

from utils import (
    run_inference,
    load_bbox_json,
    save_depth_map
)

from tta import tta_update
from loss import compute_edge_aware_loss
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import pipeline
from PIL import Image
from run_patch_inference_on_important_objects import (run_patch_inference_on_important_objects,
                                                      run_light_segmentation)
from ultralytics import YOLO

# 설정
IMAGE_DIR = "images/train"
JSON_DIR  = "labels_bbox/train"
#SAVE_DIR  = "output_average"
SAVE_DIR = 'output_weght_adaptive'
NUM_TTA   = 10
OVERLAY_ALPHA = 0.5
os.makedirs(SAVE_DIR, exist_ok=True)

# 모델 로드
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(DEVICE)

# llm 모델 로드
clip_classifier = pipeline(
    task="zero-shot-image-classification",
    model="openai/clip-vit-large-patch14"
)
important_labels = ["car", "person", "bicycle", "bus", "motorcycle"]

# YOLO segmentation model
seg_model = YOLO("yolov8n-seg.pt")

# 이미지 리스트 불러오기
image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.png")))

for img_path in image_paths:
    base_name = os.path.splitext(os.path.basename(img_path))[0].replace("_leftImg8bit", "")
    print(f"\n[INFO] Processing {base_name}")

    # 1. Base inference
    base_depth, pil_img, inputs = run_inference(img_path, return_tensor=True)
    base_depth_map = base_depth.squeeze(0).cpu().numpy()

    # 2. Load bbox json & generate soft mask
    json_candidates = glob(os.path.join(JSON_DIR, f"{base_name}_*.json"))
    if not json_candidates:
        print(f"[WARNING] JSON not found for {base_name}, skipping...")
        continue

    json_path = json_candidates[0]

    exclude_categories = [
    "road", "parking", "sidewalk", "sky", "out of roi", "license plate",
    "ego vehicle", "rectification border", "ground", "terrian",
    "guard rail", "polegroup", "rail track", "truckgroup",
    "cargroup", "bicyclegroup", "motorcyclegroup",
    "persongroup", "ridegroup"
    ]

    bboxes = load_bbox_json(json_path, exclude=exclude_categories)
    
    important_bboxes = []
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        crop = pil_img.crop((x1, y1, x2, y2))
        preds = clip_classifier(crop, candidate_labels=important_labels)
        top_label, score = preds[0]["label"], preds[0]["score"]
        if score > 0.5:
            original_bbox_depth = base_depth_map[y1:y2, x1:x2]
            important_bboxes.append({
                "bbox":bbox,
                "label":top_label,
                "score":score,
                "original_depth":original_bbox_depth  # 원래 depth map에서 bbox depth 정보 저장
            })
    print(f"Selected {len(important_bboxes)} important bboxes based on CLIP.")
    
    # post-processing refinement 
    refined_depth_map = run_patch_inference_on_important_objects(
        model = model,
        processor=processor,
        image_pil = pil_img,
        original_depth_map = base_depth_map,
        important_bboxes = important_bboxes,
        margin=100
    )

    # 5. Save depth map
    save_path = os.path.join(SAVE_DIR, f"{base_name}_depth_average.png")
    save_depth_map(refined_depth_map, save_path)
    print(f"[SAVED] {save_path}")
