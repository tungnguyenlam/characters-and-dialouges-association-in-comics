# code/pipeline/Utils/MangaTypesetter.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

class MangaTypesetter:
    def __init__(self, font_families=['Comic Sans MS', 'Chalkboard SE', 'sans-serif']):
        self.font_props = fm.FontProperties(family=font_families)
        self.font_path = fm.findfont(self.font_props)
        self.text_color = (0, 0, 0)

    def render(self, original_image_rgb, bubbles):
        """
        Takes the original image and a list of bubbles (with 'translated_text').
        Returns the final numpy image with text typeset.
        """
        image_final = original_image_rgb.copy()
        erosion_kernel = np.ones((6, 6), np.uint8)

        # 1. Whitening (Inpainting) Step
        for bubble in bubbles:
            if not bubble.get('translated_text', '').strip(): continue
            
            # Use original_mask to have better overlay
            mask_to_use = bubble.get('original_mask', bubble['mask'])

            # Erode to clear old text without killing the border
            eroded_mask = cv2.erode(mask_to_use, erosion_kernel, iterations=1)
            contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_final, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

        # 2. Text Rendering Step
        pil_image = Image.fromarray(image_final)
        draw = ImageDraw.Draw(pil_image)

        for bubble in bubbles:
            text = bubble.get('translated_text', '')
            if not text.strip(): continue
            
            self._fit_text_in_mask(draw, text, bubble['mask'])

        return np.array(pil_image)

    def _smart_wrap_text(self, draw, text, font, max_width_px, force_break=False):
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            word_width = draw.textbbox((0, 0), word, font=font)[2]
            if word_width > max_width_px:
                if not force_break:
                    return None

            test_line = ' '.join(current_line + [word]) if current_line else word
            w = draw.textbbox((0, 0), test_line, font=font)[2]
            
            if w <= max_width_px:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word is wider than max width, force break
                    temp_word = ""
                    for char in word:
                        if draw.textbbox((0, 0), temp_word + char, font=font)[2] <= max_width_px:
                            temp_word += char
                        else:
                            if temp_word:
                                lines.append(temp_word)
                            temp_word = char
                    current_line = [temp_word] if temp_word else []
        if current_line:
            lines.append(' '.join(current_line))
        return lines

    def _check_mask_collision(self, text_mask, bubble_mask_crop):
        # Invert bubble: White = Wall
        bubble_wall = cv2.bitwise_not(bubble_mask_crop)
        # Intersection: Text touching Wall
        overlap = cv2.bitwise_and(text_mask, bubble_wall)
        return cv2.countNonZero(overlap) > 0

    def _fit_text_in_mask(self, draw, text, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Calculate Centroid
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2
        
        mask_crop = mask[y:y+h, x:x+w]
        
        # Start searching for best font
        font_size = min(h, w) # Start big
        min_font_size = 12    # Don't go too small or it's unreadable
        best_font = None
        best_lines = []
        best_y_start = 0
        
        # --- ATTEMPT 1: PIXEL PERFECT FIT ---
        # We try to find a size where text stays strictly inside the white mask
        search_font_size = font_size
        while search_font_size >= min_font_size:
            font = ImageFont.truetype(self.font_path, search_font_size)
            # Try to wrap text into the width of the box (0.9 buffer)
            lines = self._smart_wrap_text(draw, text, font, w * 0.95, force_break=False)
            
            if lines is None:
                search_font_size -= 2
                continue

            text_h = sum([draw.textbbox((0, 0), line, font=font)[3] for line in lines])
            
            # Quick check: is the block taller than the bubble?
            if text_h > h:
                search_font_size -= 2
                continue

            # Advanced check: Does the text hit the black walls of the mask?
            # Create a temp canvas for the text
            pil_canvas = Image.new('L', (w, h), 0)
            canvas_draw = ImageDraw.Draw(pil_canvas)
            
            curr_y = (cy - y) - (text_h / 2)
            rel_cx = cx - x
            
            for line in lines:
                line_w = draw.textbbox((0, 0), line, font=font)[2]
                line_h = draw.textbbox((0, 0), line, font=font)[3]
                # Draw text as white on black
                canvas_draw.text((rel_cx - line_w/2, curr_y), line, font=font, fill=255)
                curr_y += line_h
            
            # Check collision (Text pixels vs Mask Wall pixels)
            if not self._check_mask_collision(np.array(pil_canvas), mask_crop):
                best_font = font
                best_lines = lines
                best_y_start = cy - (text_h / 2)
                break
            
            search_font_size -= 2

        # --- ATTEMPT 2: FALLBACK (FORCE DRAW) ---
        # If the text is too long/complex to fit perfectly.
        # We draw it anyway at the minimum size so the user can at least see it.
        if best_font is None:
            print(f"Warning: Text could not fit perfectly in bubble. Forcing render. Text: {text[:20]}...")
            best_font = ImageFont.truetype(self.font_path, min_font_size)
            best_lines = self._smart_wrap_text(draw, text, best_font, w, force_break=True) # Use full width
            
            # Recalculate height
            text_h = sum([draw.textbbox((0, 0), line, font=best_font)[3] for line in best_lines])
            best_y_start = cy - (text_h / 2)

        # Draw to main image
        current_y = best_y_start
        for line in best_lines:
            line_w = draw.textbbox((0, 0), line, font=best_font)[2]
            line_h = draw.textbbox((0, 0), line, font=best_font)[3]
            # Draw with black text
            draw.text((cx - line_w/2, current_y), line, font=best_font, fill=self.text_color)
            current_y += line_h