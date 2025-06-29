from metric import compute_affine_invariant_metrics, disparity_to_depth
import cv2
import numpy as np
import os
import math

def main():
    gt_dir = "GT"
    pred_dir = "output_weght_adaptive"  # evaluate ours
    #pred_dir = "output1/base_model"  # evaluate base_model
    
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".png")])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".png")])

    all_metrics = []

    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_path = os.path.join(gt_dir, gt_file)
        pred_path = os.path.join(pred_dir, pred_file)

        gt_raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        pred_depth = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        print(f"Raw GT min: {gt_raw.min()}, max: {gt_raw.max()}")

        # CASE 1: GT가 depth PNG, mm 단위일 경우
        if gt_raw.max() > 10000:  # mm 단위라면 값이 수천 이상일 것
            gt_depth = gt_raw / 1000.0  # mm -> m
        # CASE 2: GT가 disparity PNG (raw/256)라면
        else:
            gt_disparity = gt_raw / 256.0
            gt_depth = disparity_to_depth(gt_disparity)

        # pred scaling 필요하면
        if pred_depth.max() < 1.0:
            pred_depth = pred_depth * 100  # 예: 25cm -> 25m로 스케일 맞춤
        elif pred_depth.max() > 100:
            pred_depth = pred_depth / 1000.0  # mm -> m 변환

        valid_mask = (gt_depth > 1.0) & (gt_depth < 20.0)
        metrics = compute_affine_invariant_metrics(gt_depth, pred_depth, mask=valid_mask)
        all_metrics.append(metrics)

        print(f"\n=== {gt_file} ===")
        for k, v in metrics.items():
            v_scaled = math.floor(v*1000) / 1000
            print(f"{k}: {v_scaled}")    

    final = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    print("\n=== Final Averaged Metrics ===")
    for k, v in final.items():
        v_scaled = math.floor(v*1000) / 1000
        print(f"{k}: {v_scaled}")

if __name__ == "__main__":
    main()
