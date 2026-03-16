import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
from scipy.ndimage import binary_erosion
from PIL import Image, ImageDraw
import cv2

np.random.seed(3)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.rand(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - x0, box[3] - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def save_image_with_mask(mask, image, save_prefix="image_with_mask", random_color=False, borders=True, resize_to=None):
    """
    Optimized version using PIL and CV2 instead of Matplotlib for speed.
    """
    # Convert PIL Image to RGBA
    if image.mode != 'RGBA':
        base = image.convert('RGBA')
    else:
        base = image.copy()
    
    w, h = base.size
    
    # Process mask
    if random_color:
        # random color in 0-255 range
        color = np.concatenate([np.random.rand(3) * 255, [153]]) # 0.6 alpha -> 153
    else:
        color = np.array([30, 144, 255, 153]) # Blue-ish

    # mask is HxW boolean or binary
    mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_rgba[mask > 0] = color.astype(np.uint8)
    
    mask_layer = Image.fromarray(mask_rgba, mode='RGBA')
    base = Image.alpha_composite(base, mask_layer)
    
    if borders:
        # Find contours using cv2
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        # cv2.findContours modifies the image in some versions, but not recent ones. Safe to pass.
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on a numpy view of the image
        # Note: PIL Image is RGB/RGBA, cv2 usually expects BGR/BGRA?
        # But if we draw (0, 255, 0) on RGBA, it is Green.
        # If we draw on BGRA, it is Green.
        # So (0, 255, 0, 255) is safe for Green.
        
        base_np = np.array(base)
        cv2.drawContours(base_np, contours, -1, (0, 255, 0, 255), 2)
        base = Image.fromarray(base_np, mode='RGBA')

    output_dir = os.path.dirname(save_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Optional resize for visualization export (saves disk space)
    if resize_to is not None:
        # (width, height)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        base = base.resize(resize_to, resample=resampling)

    base.save(f"{save_prefix}.png")

def save_image_with_points(image, points, save_prefix="image_with_points"):
    """
    Optimized version using PIL.
    """
    base = image.copy()
    draw = ImageDraw.Draw(base)
    
    points_np = np.array(points)
    # Assuming all points are positive (green)
    r = 5
    for p in points_np:
        x, y = p
        draw.ellipse((x-r, y-r, x+r, y+r), fill='green', outline='white', width=1)
        
    output_dir = os.path.dirname(save_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    base.save(f"{save_prefix}.png")

def save_image_with_box(boxes, image, save_prefix="image_with_box"):
    """
    Optimized version using PIL.
    """
    base = image.copy()
    draw = ImageDraw.Draw(base)
    
    for box in boxes:
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline='green', width=2)
        
    output_dir = os.path.dirname(save_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    base.save(f"{save_prefix}.png")

def save_image_with_points_and_box(image, points, boxes, save_prefix="image_with_prompts", resize_to=None):
    """
    Optimized version using PIL.
    """
    base = image.copy()
    draw = ImageDraw.Draw(base)
    
    # Points
    if points is not None:
        points_np = np.array(points)
        r = 5
        for p in points_np:
            x, y = p
            draw.ellipse((x-r, y-r, x+r, y+r), fill='green', outline='white', width=1)
            
    # Boxes
    if boxes is not None:
        for box in boxes:
            x0, y0, x1, y1 = box
            draw.rectangle([x0, y0, x1, y1], outline='green', width=2)
        
    output_dir = os.path.dirname(save_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if resize_to is not None:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        base = base.resize(resize_to, resample=resampling)

    base.save(f"{save_prefix}.png")
