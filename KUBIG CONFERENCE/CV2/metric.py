import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# Disparity -> Depth 변환
def disparity_to_depth(disparity_map, focal_length=2262.52, baseline=0.209313):
    """
    disparity_map: (H, W) float32 ndarray
    focal_length: focal length in pixels
    baseline: baseline in meters
    
    Returns:
    depth_map: (H, W) float32 ndarray (meters)
    """
    # avoid division by zero
    disparity_map = np.clip(disparity_map, 1e-6, None)
    depth_map = (focal_length * baseline) / disparity_map
    return depth_map

def align_depth(d):
    t = np.median(d)
    s = np.mean(np.abs(d - t))
    d_aligned = (d - t) / (s + 1e-6)  # avoid division by zero
    return d_aligned


def aggregate_metrics(metrics_list):
    final = {}
    for key in metrics_list[0].keys():
        final[key] = np.mean([m[key] for m in metrics_list])
    return final


# metric 계산
def compute_affine_invariant_metrics(gt, pred, mask=None):
    if mask is None:
        mask = (gt > 1.0) & (gt < 20.0)

    gt = gt[mask]
    pred = pred[mask]

    # scale + shift alignment (least squares fit)
    A = np.vstack([pred, np.ones_like(pred)]).T
    s, t = np.linalg.lstsq(A, gt, rcond=None)[0]
    pred_aligned = s * pred + t

    gt = np.clip(gt, 1e-6, None)
    pred_aligned = np.clip(pred_aligned, 1e-6, None)

    abs_error = np.abs(gt - pred_aligned)
    abs_error = np.clip(abs_error, 0, 10)
    abs_rel = np.mean(abs_error / gt)

    rmse = np.sqrt(np.mean((gt - pred_aligned) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred_aligned)) ** 2))

    ratio = np.maximum(gt / pred_aligned, pred_aligned / gt)
    delta1 = np.mean(ratio < 1.3)
    delta2 = np.mean(ratio < 1.3 ** 2)
    delta3 = np.mean(ratio < 1.3 ** 3)
    
    ssim_val = ssim(
        (gt - gt.min()) / (gt.max() - gt.min() + 1e-6),
        (pred_aligned - pred_aligned.min()) / (pred_aligned.max() - pred_aligned.min() + 1e-6),
        data_range=1.0
    )

    return {
        'AbsRel': abs_rel,
        'RMSE': rmse,
        'RMSE_log': rmse_log,
        'delta1': delta1,
        'delta2': delta2,
        'delta3': delta3,
        'SSIM' : ssim_val
    }
