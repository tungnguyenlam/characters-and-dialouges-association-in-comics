# code/pipeline/Utils/MangaTypesetter.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from typing import List, Tuple, Optional


class MangaTypesetter:
    """Renders translated text onto manga speech bubbles."""
    
    def __init__(
        self, 
        font_families: List[str] = ['Comic Sans MS', 'Chalkboard SE', 'sans-serif'],
        text_color: Tuple[int, int, int] = (0, 0, 0),
        min_font_size: int = 10,
        erosion_kernel_size: int = 6
    ):
        self.font_props = fm.FontProperties(family=font_families)
        self.font_path = fm.findfont(self.font_props)
        self.text_color = text_color
        self.min_font_size = min_font_size
        self.erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)

    def render(
        self, 
        image: np.ndarray, 
        masks: List[np.ndarray], 
        texts: List[str]
    ) -> np.ndarray:
        """
        Render translated text onto manga bubbles.
        
        Args:
            image: RGB numpy array
            masks: List of binary masks (same length as texts)
            texts: List of translated text strings
            
        Returns:
            Image with text rendered
        """
        if len(masks) != len(texts):
            raise ValueError(f"masks ({len(masks)}) and texts ({len(texts)}) must have same length")
        
        image_final = image.copy()
        h, w = image.shape[:2]
        
        # Resize masks if needed
        processed_masks = []
        for mask in masks:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            # Normalize to 0-255
            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
            processed_masks.append(mask)

        # 1. Whitening - clear original text
        for mask, text in zip(processed_masks, texts):
            if not text.strip():
                continue
            eroded_mask = cv2.erode(mask, self.erosion_kernel, iterations=1)
            contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_final, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

        # 2. Text Rendering
        pil_image = Image.fromarray(image_final)
        draw = ImageDraw.Draw(pil_image)

        for mask, text in zip(processed_masks, texts):
            if not text.strip():
                continue
            self._fit_text_in_mask(draw, text, mask)

        return np.array(pil_image)

    def _fit_text_in_mask(self, draw: ImageDraw.Draw, text: str, mask: np.ndarray):
        """Fit text inside mask boundary with optimal font size."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
            
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Calculate centroid for text centering
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2
        
        mask_crop = mask[y:y+h, x:x+w]
        
        # Find best font size
        font_size = min(h, w)
        best_font = None
        best_lines = []
        best_y_start = 0
        
        while font_size >= self.min_font_size:
            font = ImageFont.truetype(self.font_path, font_size)
            lines = self._wrap_text(draw, text, font, w * 0.9)
            
            text_h = sum(draw.textbbox((0, 0), line, font=font)[3] for line in lines)
            text_w = max((draw.textbbox((0, 0), line, font=font)[2] for line in lines), default=0)
            
            if text_h > h or text_w > w:
                font_size -= 2
                continue

            # Check if text fits inside mask (pixel-perfect collision check)
            if self._text_fits_in_mask(draw, lines, font, mask_crop, cx - x, cy - y, text_h):
                best_font = font
                best_lines = lines
                best_y_start = cy - (text_h / 2)
                break
            
            font_size -= 2

        # Draw text
        if best_font:
            self._draw_centered_text(draw, best_lines, best_font, cx, best_y_start)

    def _wrap_text(self, draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: float) -> List[str]:
        """Word-wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            line_width = draw.textbbox((0, 0), test_line, font=font)[2]
            
            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word too long - force character break
                    lines.extend(self._break_long_word(draw, word, font, max_width))
                    current_line = []
                    
        if current_line:
            lines.append(' '.join(current_line))
            
        return lines

    def _break_long_word(self, draw: ImageDraw.Draw, word: str, font: ImageFont.FreeTypeFont, max_width: float) -> List[str]:
        """Break a single long word into multiple lines."""
        lines = []
        current = ""
        for char in word:
            if draw.textbbox((0, 0), current + char, font=font)[2] <= max_width:
                current += char
            else:
                if current:
                    lines.append(current)
                current = char
        if current:
            lines.append(current)
        return lines

    def _text_fits_in_mask(
        self, 
        draw: ImageDraw.Draw, 
        lines: List[str], 
        font: ImageFont.FreeTypeFont, 
        mask_crop: np.ndarray,
        rel_cx: int,
        rel_cy: int,
        text_h: float
    ) -> bool:
        """Check if text fits inside mask without collision."""
        h, w = mask_crop.shape
        text_canvas = np.zeros((h, w), dtype=np.uint8)
        pil_canvas = Image.fromarray(text_canvas)
        canvas_draw = ImageDraw.Draw(pil_canvas)
        
        curr_y = rel_cy - (text_h / 2)
        for line in lines:
            line_w = draw.textbbox((0, 0), line, font=font)[2]
            line_h = draw.textbbox((0, 0), line, font=font)[3]
            canvas_draw.text((rel_cx - line_w / 2, curr_y), line, font=font, fill=255)
            curr_y += line_h
        
        # Check collision: text touching outside of mask
        bubble_wall = cv2.bitwise_not(mask_crop)
        overlap = cv2.bitwise_and(np.array(pil_canvas), bubble_wall)
        return cv2.countNonZero(overlap) == 0

    def _draw_centered_text(
        self, 
        draw: ImageDraw.Draw, 
        lines: List[str], 
        font: ImageFont.FreeTypeFont, 
        cx: int, 
        y_start: float
    ):
        """Draw lines centered at cx."""
        current_y = y_start
        for line in lines:
            line_w = draw.textbbox((0, 0), line, font=font)[2]
            line_h = draw.textbbox((0, 0), line, font=font)[3]
            draw.text((cx - line_w / 2, current_y), line, font=font, fill=self.text_color)
            current_y += line_h