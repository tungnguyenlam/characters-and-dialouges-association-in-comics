import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DATA_ROOT = Path("data/open-mantra-dataset")
OUTPUT_DIR = Path("output/dataset/open-mantra/visualize")
ANNOTATION_FILE = DATA_ROOT / "annotation.json"

COLORS = [
    "#CC0000", "#007777", "#0066AA", "#228B22", "#CC8800",
    "#8B008B", "#006666", "#B8860B", "#6B238E", "#1E90FF"
]

def load_annotations():
    with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def draw_bbox_with_label(draw, bbox, idx, text_ja, text_en, color, font_ja, font_en, img_width, img_height):
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    
    draw.rectangle([x, y, x + w, y + h], outline=color, width=6)
    
    label = f"[{idx}]"
    label_bbox = draw.textbbox((0, 0), label, font=font_en)
    label_w, label_h = label_bbox[2] - label_bbox[0], label_bbox[3] - label_bbox[1]
    
    draw.rectangle([x, y - label_h - 6, x + label_w + 12, y], fill=color)
    draw.text((x + 6, y - label_h - 3), label, fill="white", font=font_en)
    
    info_y = y + h + 8
    max_width = min(img_width - x - 10, 400)
    
    def wrap_text(text, font, max_width):
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text]
    
    ja_lines = wrap_text(f"JA: {text_ja}", font_ja, max_width)
    en_lines = wrap_text(f"EN: {text_en}", font_en, max_width)
    
    line_height_ja = draw.textbbox((0, 0), "A", font=font_ja)[3] - draw.textbbox((0, 0), "A", font=font_ja)[1] + 4
    line_height_en = draw.textbbox((0, 0), "A", font=font_en)[3] - draw.textbbox((0, 0), "A", font=font_en)[1] + 4
    
    current_y = info_y
    for line in ja_lines:
        if current_y + line_height_ja > img_height - 10:
            break
        draw.text((x, current_y), line, fill=color, font=font_ja)
        current_y += line_height_ja
    
    for line in en_lines:
        if current_y + line_height_en > img_height - 10:
            break
        draw.text((x, current_y), line, fill=color, font=font_en)
        current_y += line_height_en



def visualize_page(image_path, bubbles, output_path):
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size
    
    # Try Japanese-capable fonts first, then Comic Sans for English
    font_ja = None
    font_en = None
    
    # Try Hiragino (common on macOS)
    try:
        font_ja = ImageFont.truetype("/System/Library/Fonts/Supplemental/Hiragino Sans GB.ttc", 18)
        font_en = ImageFont.truetype("/System/Library/Fonts/Supplemental/Comic Sans MS.ttf", 18)
    except:
        try:
            # Try Arial Unicode MS (supports many languages)
            font_ja = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 18)
            font_en = ImageFont.truetype("/System/Library/Fonts/Supplemental/Comic Sans MS.ttf", 18)
        except:
            try:
                # Fallback to any available font
                font_ja = ImageFont.truetype("/System/Library/Fonts/Supplemental/STHeiti Light.ttc", 18)
                font_en = ImageFont.truetype("/System/Library/Fonts/Supplemental/Comic Sans MS.ttf", 18)
            except:
                # Last resort
                font_ja = ImageFont.load_default()
                font_en = ImageFont.load_default()
    
    for idx, bubble in enumerate(bubbles):
        color = COLORS[idx % len(COLORS)]
        draw_bbox_with_label(
            draw, bubble, idx,
            bubble.get('text_ja', ''),
            bubble.get('text_en', ''),
            color, font_ja, font_en, img_width, img_height
        )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

def main():
    annotations = load_annotations()
    
    for book in annotations:
        book_title = book['book_title']
        print(f"Processing: {book_title}")
        
        for page in book['pages']:
            page_idx = page['page_index']
            image_rel_path = page['image_paths'].get('ja', '')
            
            if not image_rel_path:
                continue
                
            image_path = DATA_ROOT / image_rel_path
            bubbles = page.get('text', [])
            
            output_path = OUTPUT_DIR / book_title / f"page_{page_idx:03d}.jpg"
            
            visualize_page(image_path, bubbles, output_path)
            
        print(f"  Saved {len(book['pages'])} pages")
    
    print(f"\nVisualization complete! Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()