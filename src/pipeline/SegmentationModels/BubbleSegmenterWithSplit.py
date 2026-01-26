# bubble_splitting.py or keep in BubbleSegmenter.py
import cv2
import numpy as np
import sys
from typing import Tuple, List
from .YoloSeg import YoloBubbleSeg
from typing import Optional

sys.setrecursionlimit(2000)


def split_connected_bubbles(
    masks: List[np.ndarray],
    confs: List[float],
    image_shape: Tuple[int, int],
    min_defect_depth: float = 13,
    max_angle_deg: float = 170,
    min_dist_between_defects: float = 20,
    max_split_depth: int = 2
) -> Tuple[List[List[float]], List[np.ndarray], List[float]]:
    """
    Split connected bubbles in masks.
    
    Args:
        masks: List of mask arrays from segmentation
        confs: List of confidence scores
        image_shape: (height, width) of target image
        
    Returns:
        bboxes: List of [x1, y1, x2, y2]
        masks: List of split masks
        confs: List of confidence scores
    """
    if len(masks) == 0:
        return [], [], []
    
    h, w = image_shape
    split_bboxes = []
    split_masks = []
    split_confs = []

    for i, mask in enumerate(masks):
        # Resize mask to image size if needed
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        initial_mask = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)

        # Recursive split with depth limit
        result_masks = _split_recursive(
            initial_mask, min_defect_depth, max_angle_deg, min_dist_between_defects, max_split_depth
        )

        # Extract bbox for each split mask
        for single_mask in result_masks:
            contours, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(contour)
            
            split_bboxes.append([x, y, x + bw, y + bh])
            split_masks.append(single_mask)
            split_confs.append(confs[i])

    # Sort: Top-to-bottom, then Right-to-left (Manga reading order)
    if split_bboxes:
        sorted_indices = sorted(
            range(len(split_bboxes)), 
            key=lambda i: (split_bboxes[i][1], -split_bboxes[i][0])
        )
        split_bboxes = [split_bboxes[i] for i in sorted_indices]
        split_masks = [split_masks[i] for i in sorted_indices]
        split_confs = [split_confs[i] for i in sorted_indices]
    
    return split_bboxes, split_masks, split_confs


def _attempt_split_once(bubble_mask, min_defect_depth, max_angle_deg, min_dist_between_defects):
    """Core math logic for finding convexity defects and cutting."""
    contours, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [bubble_mask]
    
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 500:
        return [bubble_mask]

    try:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 3:
            return [bubble_mask]
        defects = cv2.convexityDefects(contour, hull_indices)
    except:
        return [bubble_mask]

    if defects is None:
        return [bubble_mask]

    candidates = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = d / 256.0
        
        if depth > min_defect_depth:
            far = tuple(contour[f][0])
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            
            v1 = np.array(start) - np.array(far)
            v2 = np.array(end) - np.array(far)
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            
            if n1 == 0 or n2 == 0:
                continue
            
            cosine = np.dot(v1, v2) / (n1 * n2)
            angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            
            if angle < max_angle_deg:
                candidates.append({'point': far, 'depth': depth})

    # Clustering
    candidates.sort(key=lambda x: x['depth'], reverse=True)
    unique_candidates = []
    for cand in candidates:
        is_distinct = True
        for exist in unique_candidates:
            dist = np.linalg.norm(np.array(cand['point']) - np.array(exist['point']))
            if dist < min_dist_between_defects:
                is_distinct = False
                break
        if is_distinct:
            unique_candidates.append(cand)

    # Execute Cut
    split_mask = bubble_mask.copy()
    cut_happened = False
    
    if len(unique_candidates) >= 2:
        best_pair = None
        min_cut_dist = float('inf')
        consider_list = unique_candidates[:4]
        
        for i in range(len(consider_list)):
            for j in range(i + 1, len(consider_list)):
                p1 = consider_list[i]['point']
                p2 = consider_list[j]['point']
                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                if dist < min_cut_dist:
                    min_cut_dist = dist
                    best_pair = (p1, p2)
        
        if best_pair:
            cv2.line(split_mask, best_pair[0], best_pair[1], 0, 3)
            cut_happened = True
            
    elif len(unique_candidates) == 1:
        p1 = unique_candidates[0]['point']
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            dx, dy = cx - p1[0], cy - p1[1]
            target = (int(cx + dx*0.5), int(cy + dy*0.5))
            cv2.line(split_mask, p1, target, 0, 3)
            cut_happened = True

    if cut_happened:
        new_contours, _ = cv2.findContours(split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_masks = []
        for c in new_contours:
            if cv2.contourArea(c) > 100: 
                m = np.zeros_like(bubble_mask)
                cv2.drawContours(m, [c], -1, 255, -1)
                new_masks.append(m)
        if len(new_masks) > 1:
            return new_masks

    return [bubble_mask]


def _split_recursive(bubble_mask, min_defect_depth, max_angle_deg, min_dist_between_defects, max_split_depth, depth=0):
    """Recursive wrapper with depth limit."""
    # Stop recursion if max depth reached
    if depth >= max_split_depth:
        return [bubble_mask]
    
    initial_results = _attempt_split_once(
        bubble_mask, min_defect_depth, max_angle_deg, min_dist_between_defects
    )
    if len(initial_results) == 1:
        return initial_results
    
    final_bubbles = []
    for sub_mask in initial_results:
        sub_results = _split_recursive(
            sub_mask, min_defect_depth, max_angle_deg, min_dist_between_defects, max_split_depth, depth + 1
        )
        final_bubbles.extend(sub_results)
    return final_bubbles

class BubbleSegmenterWithSplit(YoloBubbleSeg):
    def __init__(
        self,
        model_path: Optional[str] = None,
        variant: str = "v8n",
        device: str = 'auto',
        verbose: bool = False,
        plot: bool = False,
        min_defect_depth: float = 13,
        max_angle_deg: float = 170,
        min_dist_between_defects: float = 20,
        max_split_depth: int = 2
    ):
        super().__init__(model_path, variant, device, verbose, plot)
        self.min_defect_depth = min_defect_depth
        self.max_angle_deg = max_angle_deg
        self.min_dist_between_defects = min_dist_between_defects
        self.max_split_depth = max_split_depth
        self._image_shape = None

    def preprocess(self, image):
        self._image_shape = image.shape[:2]
        return image

    def postprocess(self, outputs):
        """Apply bubble splitting with depth limit."""
        
        bboxes, masks, confs = outputs
        return split_connected_bubbles(
            masks, confs, self._image_shape,
            self.min_defect_depth,
            self.max_angle_deg,
            self.min_dist_between_defects,
            self.max_split_depth
        )