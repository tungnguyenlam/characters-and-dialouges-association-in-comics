"""
expect the result of the segmentation task to be 

binary mask: true/false (numpy array)

bbox: xyxy (numpy array)

"""
import torch
import numpy as np
from typing import List, Dict, Union, Optional
from scipy.optimize import linear_sum_assignment

def yolo_style_ap(recalls, precisions):
    # ensure to be numpy arrays
    recalls = np.array(recalls)
    precisions = np.array(precisions)

    # add boundary values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # precision envelope (monotonic decreasing)
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    # recall COCO: 101 points
    recall_levels = np.linspace(0, 1, 101)

    # interpolated precision
    ap = 0.0
    for r in recall_levels:
        ap += np.max(precisions[recalls >= r])

    return ap / 101


    

class EvalSeg:
    def __init__(self, 
                 gt_masks: List[List[torch.Tensor]], 
                 gt_bboxes: List[List[List[float]]], 
                 pred_masks: List[List[torch.Tensor]], 
                 pred_bboxes: List[List[List[float]]],
                pred_probs: List[List[float]]):
        """
        Args:
            gt_masks: List of ground truth masks, each item shape [[H, W], ...]
            gt_bboxes: List of ground truth bboxes, each item: [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
            pred_masks: List of predicted masks, each item shape [[H, W], ...]
            pred_bboxes: List of predicted bboxes, each item: [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
        """
        self.gt_masks = gt_masks
        self.gt_bboxes = gt_bboxes
        self.pred_masks = pred_masks
        self.pred_bboxes = pred_bboxes
        self.pred_probs= pred_probs
        
    @staticmethod
    def iou_bbox(box1: Union[np.ndarray, List], box2: Union[np.ndarray, List]) -> float:
        """calculate IoU between 2 bounding boxes [x1, y1, x2, y2]"""
        if isinstance(box1, torch.Tensor):
            box1 = box1.cpu().numpy()
        if isinstance(box2, torch.Tensor):
            box2 = box2.cpu().numpy()
        
        box1 = np.array(box1)
        box2 = np.array(box2)
        
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        
        inter = max(0, xB - xA) * max(0, yB - yA)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - inter
        
        iou = inter / union if union > 0 else 0
        return iou
    
    @staticmethod
    def iou_mask(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """Tính IoU cho masks (hỗ trợ batch)"""
        mask1 = mask1.bool()
        mask2 = mask2.bool()
        
        inter = (mask1 & mask2).sum(dim=(-2, -1))
        union = (mask1 | mask2).sum(dim=(-2, -1))
        iou = torch.where(unionfollowing confidence descending
        all_pred.sort(key=lambda x: -x["conf"])
    
        aps = []
        results_per_threshold = {}
        mean_ious_list = []
    
        for thr in iou_thresholds:
            TP, FP = [], []
            used_gt = {}  # image_idx -> set of matched gt indices
            matched_ious = []
    
            for pred in all_pred:
                img = pred["image"]
                pmask = pred["mask"]
    
                gt_of_img = [g for g in all_gt if g["image"] == img]
    
                if len(gt_of_img) == 0:
                    FP.append(1)
                    TP.append(0)
                    continue
    
                # Compute IoU list
                ious = np.array([self.iou_mask(pmask, g["mask"]) for g in gt_of_img])
                best = ious.max()
                best_gt_idx = ious.argmax()
    
                if best >= thr:
                    if img not in used_gt:
                        used_gt[img] = set()
    
                    if best_gt_idx not in used_gt[img]:
                        TP.append(1)
                        FP.append(0)
                        used_gt[img].add(best_gt_idx)
                        matched_ious.append(best)
                    else:
                        FP.append(1)
                        TP.append(0)
                else:
                    FP.append(1)
                    TP.append(0)
    
            mean_ious_list.append(np.mean(matched_ious) if matched_ious else 0.0)
    
            TP = np.array(TP)
            FP = np.array(FP)
            cum_TP = np.cumsum(TP)
            cum_FP = np.cumsum(FP)
            total_gt = len(all_gt)
    
            recall = cum_TP / (total_gt + 1e-6)
            precision = cum_TP / (cum_TP + cum_FP + 1e-6)
    
            ap = yolo_style_ap(recall, precision)
            aps.append(ap)
    
            results_per_threshold[f"AP@{thr:.2f}"] = ap
            results_per_threshold[f"P@{thr:.2f}"] = precision[-1]
            results_per_threshold[f"R@{thr:.2f}"] = recall[-1]
            results_per_threshold[f"mIoU@{thr:.2f}"] = mean_ious_list[-1]
    
        return {
            "mAP50": results_per_threshold.get("AP@0.50", 0.0),
            "mAP50-95": np.mean(aps),
            "precision": results_per_threshold.get("P@0.50", 0.0),
            "recall": results_per_threshold.get("R@0.50", 0.0),
            "mean_iou": np.mean(mean_ious_list)
        }



        
    
    def eval_all(self, iou_thresholds: Optional[List[float]] = None) -> Dict:
        """Evaluate bbox & mask, return dict"""
        bbox_metrics = self.eval_bbox(iou_thresholds)
        mask_metrics = self.eval_mask()
        
        return {
            "bbox": bbox_metrics,
            "mask": mask_metrics
        }


def print_results(results: Dict):
    bbox = results['bbox']
    print("\n BBOX METRICS:")
    print(f"  Mean IoU:        {bbox.get('mean_iou', 0.0):.4f}")
    print(f"  mAP50:           {bbox.get('mAP50', 0.0):.4f}")
    print(f"  mAP50-95:        {bbox.get('mAP50-95', 0.0):.4f}")
    print(f"  Precision:       {bbox.get('precision', 0.0):.4f}")
    print(f"  Recall:          {bbox.get('recall', 0.0):.4f}")
    print(f"  F1 Score:        {bbox.get('f1', 0.0):.4f}")
    
    mask = results['mask']
    print("\n MASK METRICS:")
    print(f"  Mean IoU:        {mask.get('mean_iou', 0.0):.4f}")
    print(f"  mAP50:           {mask.get('mAP50', 0.0):.4f}")
    print(f"  mAP50-95:        {mask.get('mAP50-95', 0.0):.4f}")
    print(f"  Precision:       {mask.get('precision', 0.0):.4f}")
    print(f"  Recall:          {mask.get('recall', 0.0):.4f}")
