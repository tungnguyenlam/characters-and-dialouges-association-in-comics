# src/pipeline/SegmentationModels/BubbleSegmentationWithOrdering.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from typing import List, Tuple, Dict, Any, Optional
from shapely.geometry import box as shapely_box
from .BaseSegmentationModel import BaseSegmentationModel


class BubbleSegmentationWithOrdering(BaseSegmentationModel):
    """
    Wrapper that orders bubble detections in manga reading order.
    
    Optionally uses panel detection to group bubbles by panel first,
    then sorts within each panel (right-to-left, top-to-bottom).
    """
    
    def __init__(
        self,
        bubble_detector: BaseSegmentationModel,
        panel_detector: Optional[BaseSegmentationModel] = None,
        num_panel_rows: int = 4,
        merge_threshold: float = 0.2,
        right_to_left: bool = True,
        handle_double_page: bool = True,
        device: str = 'auto',
        verbose: bool = False,
        plot: bool = False
    ):
        super().__init__(
            model_path=bubble_detector.model_path,
            device=device,
            verbose=verbose,
            plot=plot
        )
        self.bubble_detector = bubble_detector
        self.panel_detector = panel_detector
        self.num_panel_rows = num_panel_rows
        self.merge_threshold = merge_threshold
        self.right_to_left = right_to_left
        self.handle_double_page = handle_double_page

    @property
    def is_loaded(self) -> bool:
        bubble_loaded = self.bubble_detector.is_loaded
        panel_loaded = self.panel_detector.is_loaded if self.panel_detector else True
        return bubble_loaded and panel_loaded

    def load_model(self) -> None:
        self.bubble_detector.load_model()
        if self.panel_detector:
            self.panel_detector.load_model()
        self._log("Models loaded")

    def unload_model(self) -> None:
        self.bubble_detector.unload_model()
        if self.panel_detector:
            self.panel_detector.unload_model()
        self._log("Models unloaded")

    def _inference(
        self, 
        image: np.ndarray, 
        conf_threshold: float = 0.5,
        **kwargs
    ) -> Tuple[List[List[float]], List[np.ndarray], List[float]]:
        """Detect bubbles and return in manga reading order."""
        # Check for double-page spread
        if self.handle_double_page and self._is_double_page(image):
            self._log("Detected double-page spread, processing separately...")
            return self._process_double_page(image, conf_threshold, **kwargs)
        
        return self._process_single_page(image, conf_threshold, **kwargs)

    def _process_single_page(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5,
        **kwargs
    ) -> Tuple[List[List[float]], List[np.ndarray], List[float]]:
        """Process a single page."""
        h, w = image.shape[:2]
        
        # Detect bubbles - use predict() to include preprocess/postprocess (e.g., splitting)
        bubble_bboxes, bubble_masks, bubble_confs = self.bubble_detector.predict(
            image, conf_threshold=conf_threshold, **kwargs
        )
        
        if len(bubble_bboxes) == 0:
            return [], [], []
        
        # Resize masks to image size if needed
        bubble_masks = self._resize_masks(bubble_masks, h, w)
        
        # Get ordering
        sorted_panel_bboxes: Optional[List[List[float]]] = None
        sorted_panel_masks: Optional[List[np.ndarray]] = None
        if self.panel_detector:
            # Use panel-aware ordering
            panel_bboxes, panel_masks, _ = self.panel_detector.predict(
                image, conf_threshold=conf_threshold, **kwargs
            )
            panel_masks = self._resize_masks(panel_masks, h, w)
            
            # Sort panels and get ordered bubble indices
            ordered_indices, sorted_panel_bboxes, sorted_panel_masks = self._order_with_panels(
                bubble_bboxes, panel_bboxes, panel_masks, h
            )
        else:
            # Simple ordering without panels
            ordered_indices = self._order_simple(bubble_bboxes, h)
        
        # Reorder outputs
        ordered_bboxes = [bubble_bboxes[i] for i in ordered_indices]
        ordered_masks = [bubble_masks[i] for i in ordered_indices]
        ordered_confs = [bubble_confs[i] for i in ordered_indices]

        if self.plot:
            self.plot_panels_and_bubbles(image, sorted_panel_bboxes or [], ordered_bboxes, sorted_panel_masks)
        
        return ordered_bboxes, ordered_masks, ordered_confs

    # === DOUBLE-PAGE HANDLING ===

    def _is_double_page(self, image: np.ndarray) -> bool:
        """Check if image is a double-page spread (width > height)."""
        height, width = image.shape[:2]
        return width / height > 1.0

    def _split_double_page(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split double-page into right and left halves (manga reading order)."""
        height, width = image.shape[:2]
        mid = width // 2
        right_page = image[:, mid:]  # Right page first for manga
        left_page = image[:, :mid]
        return right_page, left_page

    def _process_double_page(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5,
        **kwargs
    ) -> Tuple[List[List[float]], List[np.ndarray], List[float]]:
        """Process double-page spread by splitting and merging results."""
        height, full_width = image.shape[:2]
        mid_x = full_width // 2
        
        # Split into right and left pages
        right_page, left_page = self._split_double_page(image)
        
        self._log("Processing right page...")
        right_bboxes, right_masks, right_confs = self._process_single_page(
            right_page, conf_threshold, **kwargs
        )
        
        self._log("Processing left page...")
        left_bboxes, left_masks, left_confs = self._process_single_page(
            left_page, conf_threshold, **kwargs
        )
        
        # Adjust right page coordinates (add mid_x offset)
        adjusted_right_bboxes = [
            [x1 + mid_x, y1, x2 + mid_x, y2]
            for x1, y1, x2, y2 in right_bboxes
        ]
        adjusted_right_masks = [
            self._adjust_mask_for_right_page(mask, full_width, mid_x)
            for mask in right_masks
        ]
        
        # Adjust left page masks (pad to full width)
        adjusted_left_masks = [
            self._adjust_mask_for_left_page(mask, full_width)
            for mask in left_masks
        ]
        
        # Merge: right page first, then left page (manga reading order)
        all_bboxes = adjusted_right_bboxes + left_bboxes
        all_masks = adjusted_right_masks + adjusted_left_masks
        all_confs = right_confs + left_confs
        
        self._log(f"Combined {len(right_bboxes)} + {len(left_bboxes)} = {len(all_bboxes)} bubbles")
        
        return all_bboxes, all_masks, all_confs

    def _adjust_mask_for_right_page(
        self, 
        mask: np.ndarray, 
        full_width: int,
        mid_x: int
    ) -> np.ndarray:
        """Pad mask to fit in right side of full image."""
        height = mask.shape[0]
        page_width = mask.shape[1]
        full_mask = np.zeros((height, full_width), dtype=mask.dtype)
        full_mask[:, mid_x:mid_x + page_width] = mask
        return full_mask

    def _adjust_mask_for_left_page(
        self, 
        mask: np.ndarray, 
        full_width: int
    ) -> np.ndarray:
        """Pad mask to fit in left side of full image."""
        height = mask.shape[0]
        page_width = mask.shape[1]
        full_mask = np.zeros((height, full_width), dtype=mask.dtype)
        full_mask[:, :page_width] = mask
        return full_mask

    def _resize_masks(self, masks: List[np.ndarray], h: int, w: int) -> List[np.ndarray]:
        """Resize masks to target dimensions."""
        resized = []
        for mask in masks:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            resized.append(mask)
        return resized

    # === ORDERING LOGIC ===

    def _order_simple(self, bboxes: List[List[float]], image_height: int) -> List[int]:
        """Order bubbles without panel information (row-based)."""
        row_height = image_height / self.num_panel_rows
        
        # Group by row
        row_groups: Dict[int, List[Tuple[int, float]]] = {}
        for i, bbox in enumerate(bboxes):
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            row_idx = int(center_y // row_height)
            
            if row_idx not in row_groups:
                row_groups[row_idx] = []
            row_groups[row_idx].append((i, center_x))
        
        # Sort each row and concatenate
        ordered = []
        for row_idx in sorted(row_groups.keys()):
            items = row_groups[row_idx]
            items.sort(key=lambda x: x[1], reverse=self.right_to_left)
            ordered.extend([idx for idx, _ in items])
        
        return ordered

    def _order_with_panels(
        self,
        bubble_bboxes: List[List[float]],
        panel_bboxes: List[List[float]],
        panel_masks: List[np.ndarray],
        image_height: int
    ) -> Tuple[List[int], List[List[float]], List[np.ndarray]]:
        """Order bubbles using panel information."""
        # Merge overlapping panels
        panel_bboxes, panel_masks = self._merge_overlapping_panels(panel_bboxes, panel_masks)
        
        # Sort panels in manga order
        sorted_panel_indices = self._sort_panels_manga_order(panel_bboxes, image_height)
        sorted_panels = [panel_bboxes[i] for i in sorted_panel_indices]
        sorted_panel_masks = [panel_masks[i] for i in sorted_panel_indices]
        
        # Map bubbles to panels
        panel_to_bubbles = self._map_bubbles_to_panels(bubble_bboxes, sorted_panels)
        
        # Order bubbles within each panel, then concatenate
        ordered = []
        for panel_idx in range(len(sorted_panels)):
            bubble_indices = panel_to_bubbles.get(panel_idx, [])
            if bubble_indices:
                sorted_in_panel = self._sort_bubbles_in_panel(bubble_bboxes, bubble_indices)
                ordered.extend(sorted_in_panel)
        
        # Add unassigned bubbles at the end
        if -1 in panel_to_bubbles:
            unassigned = self._sort_bubbles_in_panel(bubble_bboxes, panel_to_bubbles[-1])
            ordered.extend(unassigned)
        
        return ordered, sorted_panels, sorted_panel_masks

    def _merge_overlapping_panels(
        self,
        bboxes: List[List[float]],
        masks: List[np.ndarray]
    ) -> Tuple[List[List[float]], List[np.ndarray]]:
        """Merge panels that overlap significantly."""
        bboxes = list(bboxes)
        masks = list(masks)
        
        merge_happened = True
        while merge_happened:
            merge_happened = False
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    ratio = self._intersection_ratio(bboxes[i], bboxes[j])
                    if ratio > self.merge_threshold:
                        # Merge j into i
                        bboxes[i] = self._merge_bboxes(bboxes[i], bboxes[j])
                        masks[i] = np.logical_or(masks[i], masks[j]).astype(masks[i].dtype)
                        bboxes.pop(j)
                        masks.pop(j)
                        merge_happened = True
                        break
                if merge_happened:
                    break
        
        return bboxes, masks

    def _intersection_ratio(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate intersection area / min area ratio."""
        box1 = shapely_box(*bbox1)
        box2 = shapely_box(*bbox2)
        
        if not box1.intersects(box2):
            return 0.0
        
        intersection = box1.intersection(box2).area
        min_area = min(box1.area, box2.area)
        return intersection / min_area if min_area > 0 else 0.0

    def _merge_bboxes(self, bbox1: List[float], bbox2: List[float]) -> List[float]:
        """Merge two bboxes into their union."""
        return [
            min(bbox1[0], bbox2[0]),
            min(bbox1[1], bbox2[1]),
            max(bbox1[2], bbox2[2]),
            max(bbox1[3], bbox2[3])
        ]

    def _sort_panels_manga_order(self, bboxes: List[List[float]], image_height: int) -> List[int]:
        """Return indices of panels sorted in manga reading order.
        
        Uses adaptive Y-gap detection instead of a fixed-grid row height,
        so panels near a grid boundary are not incorrectly split into different rows.
        """
        if not bboxes:
            return []
        
        # Build (index, center_x, center_y, panel_height) tuples
        items = []
        for i, bbox in enumerate(bboxes):
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            ph = bbox[3] - bbox[1]
            items.append((i, cx, cy, ph))
        
        # Sort by center_y to detect natural row breaks
        items_sorted = sorted(items, key=lambda x: x[2])
        
        # Use min panel height * 0.5 as row-break gap threshold
        min_height = min(it[3] for it in items_sorted)
        gap_threshold = min_height * 0.5
        
        rows = []
        current_row = [items_sorted[0]]
        for k in range(1, len(items_sorted)):
            if items_sorted[k][2] - items_sorted[k - 1][2] > gap_threshold:
                rows.append(current_row)
                current_row = [items_sorted[k]]
            else:
                current_row.append(items_sorted[k])
        rows.append(current_row)
        
        # Sort each row right-to-left, concatenate top-to-bottom
        ordered = []
        for row in rows:
            row.sort(key=lambda x: x[1], reverse=self.right_to_left)
            ordered.extend([it[0] for it in row])
        
        return ordered

    def _map_bubbles_to_panels(
        self,
        bubble_bboxes: List[List[float]],
        panel_bboxes: List[List[float]]
    ) -> Dict[int, List[int]]:
        """Map bubble indices to panel indices by maximum overlap area.
        
        Uses overlap-area based assignment instead of center-point containment,
        so bubbles straddling panel borders are assigned to the panel they
        overlap with the most rather than whichever contains their center.
        """
        panel_to_bubbles: Dict[int, List[int]] = {i: [] for i in range(len(panel_bboxes))}
        panel_to_bubbles[-1] = []  # Unassigned
        
        for i, b in enumerate(bubble_bboxes):
            best_panel = -1
            best_area = 0.0
            
            for j, p in enumerate(panel_bboxes):
                ix1 = max(b[0], p[0])
                iy1 = max(b[1], p[1])
                ix2 = min(b[2], p[2])
                iy2 = min(b[3], p[3])
                
                if ix2 > ix1 and iy2 > iy1:
                    area = (ix2 - ix1) * (iy2 - iy1)
                    if area > best_area:
                        best_area = area
                        best_panel = j
            
            panel_to_bubbles[best_panel].append(i)
        
        return panel_to_bubbles

    def _sort_bubbles_in_panel(
        self,
        all_bboxes: List[List[float]],
        bubble_indices: List[int]
    ) -> List[int]:
        """Sort bubble indices within a panel using adaptive row detection.
        
        Uses min bubble height (instead of avg) as the gap threshold basis
        so that one oversized bubble cannot collapse two real rows into one.
        """
        if len(bubble_indices) <= 1:
            return bubble_indices
        
        # Get centers
        centers = []
        for i in bubble_indices:
            bbox = all_bboxes[i]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            height = bbox[3] - bbox[1]
            centers.append((i, cx, cy, height))
        
        # Sort by center_y to find natural row breaks
        centers_sorted = sorted(centers, key=lambda x: x[2])
        
        # Use min height * 0.5 — robust against outlier large bubbles
        min_height = min(c[3] for c in centers_sorted)
        gap_threshold = min_height * 0.5
        
        rows = []
        current_row = [centers_sorted[0]]
        
        for k in range(1, len(centers_sorted)):
            prev_y = centers_sorted[k - 1][2]
            curr_y = centers_sorted[k][2]
            
            if curr_y - prev_y > gap_threshold:
                rows.append(current_row)
                current_row = [centers_sorted[k]]
            else:
                current_row.append(centers_sorted[k])
        
        rows.append(current_row)
        
        # Sort each row right-to-left, concatenate top-to-bottom
        ordered = []
        for row in rows:
            row.sort(key=lambda x: x[1], reverse=self.right_to_left)
            ordered.extend([item[0] for item in row])
        
        return ordered

    # === VISUALIZATION ===

    def _visualize(self, inputs: Any, outputs: Any) -> None:
        """Visualize segmentation results with reading order."""
        image = inputs
        bboxes, masks, confs = outputs
        
        if len(bboxes) == 0:
            self._log("No bubbles to visualize")
            return
        
        self.plot_bubbles_with_order(image, bboxes, masks)
        self.plot_reading_order_arrows(image, bboxes)

    def plot_bubbles_with_order(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        masks: List[np.ndarray],
        panel_bboxes: Optional[List[List[float]]] = None,
        panel_masks: Optional[List[np.ndarray]] = None,
        figsize: Tuple[int, int] = (20, 14),
        title: str = "Bubble Detection with Reading Order"
    ) -> None:
        """Plot image with bubbles and optionally panels, showing reading order."""
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        colors = list(mcolors.TABLEAU_COLORS.values())
        h, w = image.shape[:2]
        
        # Normalize image for display
        img_display = image / 255.0 if image.max() > 1.0 else image
        
        # === Plot 1: Bounding boxes with order numbers ===
        ax1 = axes[0]
        ax1.imshow(img_display)
        
        # Plot panel bboxes if provided
        if panel_bboxes:
            for i, bbox in enumerate(panel_bboxes):
                x1, y1, x2, y2 = bbox
                rect = Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=3, edgecolor='blue', facecolor='none', linestyle='-'
                )
                ax1.add_patch(rect)
                ax1.text(
                    x1 + 5, y1 + 30, f'P{i}',
                    fontsize=14, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round', facecolor='blue', alpha=0.8)
                )
        
        # Plot bubble bboxes with order
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            color = colors[i % len(colors)]
            
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none', linestyle='--'
            )
            ax1.add_patch(rect)
            
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ax1.text(
                cx, cy, f'{i}',
                fontsize=12, fontweight='bold', color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='circle', facecolor=color, alpha=0.9)
            )
        
        ax1.set_title(f"Bounding Boxes (Reading Order 0-{len(bboxes)-1})", fontsize=12)
        ax1.axis('off')
        
        # === Plot 2: Masks overlay ===
        ax2 = axes[1]
        ax2.imshow(img_display)
        
        # Create overlay for masks
        overlay = np.zeros((h, w, 4))
        
        # Panel masks (if provided)
        if panel_masks:
            for i, mask in enumerate(panel_masks):
                color = mcolors.to_rgba('blue', alpha=0.2)
                mask_resized = self._ensure_mask_size(mask, h, w)
                overlay[mask_resized > 0] = color
        
        # Bubble masks
        for i, mask in enumerate(masks):
            color = mcolors.to_rgba(colors[i % len(colors)], alpha=0.4)
            mask_resized = self._ensure_mask_size(mask, h, w)
            overlay[mask_resized > 0] = color
        
        ax2.imshow(overlay)
        ax2.set_title("Segmentation Masks", fontsize=12)
        ax2.axis('off')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_reading_order_arrows(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        figsize: Tuple[int, int] = (12, 16),
        arrow_color: str = 'lime',
        bubble_color: str = 'red',
        title: str = "Bubble Reading Order"
    ) -> None:
        """Plot bubbles with arrows showing reading order."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        img_display = image / 255.0 if image.max() > 1.0 else image
        ax.imshow(img_display)
        
        centers = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centers.append((cx, cy))
            
            # Draw bbox
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=bubble_color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Draw order number
            ax.text(
                cx, cy, f'{i}',
                fontsize=14, fontweight='bold', color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='circle', facecolor=bubble_color, alpha=0.9)
            )
        
        # Draw arrows between consecutive bubbles
        for i in range(len(centers) - 1):
            x1, y1 = centers[i]
            x2, y2 = centers[i + 1]
            
            ax.annotate(
                '',
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->',
                    color=arrow_color,
                    lw=3,
                    connectionstyle='arc3,rad=0.1'
                )
            )
        
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_panels_and_bubbles(
        self,
        image: np.ndarray,
        panel_bboxes: List[List[float]],
        bubble_bboxes: List[List[float]],
        panel_masks: Optional[List[np.ndarray]] = None,
        figsize: Tuple[int, int] = (16, 12),
        panel_color: str = 'blue',
        bubble_color: str = 'red',
        mask_alpha: float = 0.3,
        title: str = "Manga Panel & Bubble Order"
    ) -> None:
        """Plot manga image with panels and bubbles, showing their reading order IDs."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        h, w = image.shape[:2]
        img_display = image / 255.0 if image.max() > 1.0 else image
        ax.imshow(img_display)
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        # Plot panel masks if available
        if panel_masks:
            overlay = np.zeros((h, w, 4))
            for i, mask in enumerate(panel_masks):
                color = mcolors.to_rgba(colors[i % len(colors)], alpha=mask_alpha)
                mask_resized = self._ensure_mask_size(mask, h, w)
                overlay[mask_resized > 0] = color
            ax.imshow(overlay)
        
        # Plot panel bounding boxes
        for i, bbox in enumerate(panel_bboxes):
            x1, y1, x2, y2 = bbox
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor=panel_color, facecolor='none', linestyle='-'
            )
            ax.add_patch(rect)
            ax.text(
                x1 + 5, y1 + 30, f'P{i}',
                fontsize=16, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor=panel_color, alpha=0.8)
            )
        
        # Plot bubble bounding boxes
        for i, bbox in enumerate(bubble_bboxes):
            x1, y1, x2, y2 = bbox
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=bubble_color, facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(
                cx, cy, f'{i}',
                fontsize=12, fontweight='bold', color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='circle', facecolor=bubble_color, alpha=0.9)
            )
        
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=panel_color, linewidth=3, label=f'Panels (P0-P{len(panel_bboxes)-1})'),
            Line2D([0], [0], color=bubble_color, linewidth=2, linestyle='--', label=f'Bubbles (0-{len(bubble_bboxes)-1})')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        plt.show()

    def _ensure_mask_size(self, mask: np.ndarray, h: int, w: int) -> np.ndarray:
        """Ensure mask is the correct size."""
        if mask.shape[:2] != (h, w):
            return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return mask