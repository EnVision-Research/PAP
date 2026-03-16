"""
PanoAff Pipeline: Affordance-based object detection and segmentation
"""
import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
import concurrent.futures
from tqdm import tqdm
import threading
import json
import requests
import io
import cv2
import base64
import shutil
from PIL import ImageDraw, ImageFont
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.image_utils import save_image_with_points_and_box, save_image_with_mask
from utils.agent_utils import extract_json
from rex_omni import RexOmniWrapper
from utils.pano_utils import process_panorama, restore_panorama, extract_fov, restore_fov_to_panorama
from utils.pano_utils import draw_grid, draw_grid_color
# Import dataset reader
from utils.dataset_utils import PAP_Dataset

import time


class VLMClient:
    """Client for calling OpenAI-compatible API servers (vLLM, cloud APIs, etc.)."""
    
    def __init__(self, api_url, model_name="qwen3-vl-4b", api_key=None, max_retries=3, timeout=300):
        """
        Initialize VLM client for any OpenAI-compatible API server.
        
        Args:
            api_url: Base URL of the API server (e.g., "http://localhost:8000" or "https://api.openai.com")
            model_name: Model name to use in API calls
            api_key: API key for authentication (optional, not needed for local vLLM servers)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.base_url = api_url.rstrip('/')
        self.model_name = model_name
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        
        # Build default headers
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Check server health
        try:
            models_url = f"{self.base_url}/v1/models"
            response = requests.get(models_url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [m.get('id') for m in models_data.get('data', [])]
                if available_models:
                    print(f"✓ Connected to API server")
                    print(f"  Available models: {', '.join(available_models)}")
                    # Use first available model if specified model not found
                    if self.model_name not in available_models:
                        self.model_name = available_models[0]
                        print(f"  Using model: {self.model_name}")
                else:
                    print(f"✓ Connected to API server at {self.base_url}")
            else:
                print(f"⚠ API server responded with status {response.status_code}")
        except Exception as e:
            print(f"⚠ Could not connect to API server: {e}")
            print(f"  Will attempt to use {self.chat_url} anyway")
    
    def _image_to_base64(self, image):
        """Convert PIL Image to base64 data URL."""
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        buffered = io.BytesIO()
        image.save(buffered, format='JPEG')
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    
    def generate(self, image, prompt):
        """
        Generate text response from VLM using OpenAI-compatible API.
        
        Args:
            image: PIL Image
            prompt: Text prompt
            
        Returns:
            str: Generated text response
        """
        for attempt in range(self.max_retries):
            try:
                # Prepare OpenAI-compatible message format
                image_url = self._image_to_base64(image)
                
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url}
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.0,
                }
                
                # Send request to OpenAI-compatible endpoint
                response = requests.post(
                    self.chat_url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )
                if response.status_code != 200:
                    print(f"  [VLM] API error {response.status_code}: {response.text}")
                response.raise_for_status()
                
                # Parse OpenAI-compatible response
                result = response.json()
                text = result['choices'][0]['message']['content'].strip()
                return text
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"  [VLM] Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    continue
                else:
                    print(f"  [VLM] All retry attempts failed: {e}")
                    raise
        
        return None


class PanoAff:
    """PanoAff pipeline for affordance-based object detection and segmentation."""
    
    def __init__(self, 
                 vlm_model=None,
                 rex_model=None, 
                 sam2_model=None,
                 output_root="output",
                 num_workers=8,
                 system_prompt="",
                 clarity_prompt_path="system_prompt/visual_grid_prompting.md",
                 grid_cols=4,
                 grid_rows=3,
                 fov_deg=90,
                 fov_deg_subgrid=60,
                 grid_type="line",
                 grid_alpha=100,
                 line_thickness=5,
                 font_size=50,
                 small_w=2000,
                 small_h=1000):
        """
        Initialize PanoAff pipeline.
        
        Args:
            vlm_model: Vision-Language Model for generating object descriptions
            rex_model: Rex-Omni model for object detection
            sam2_model: SAM2 model for segmentation
            output_root: Root directory for saving outputs
            num_workers: Number of workers for async I/O operations
            system_prompt: System prompt for VLM
            grid_cols: Number of grid columns for visual prompt (horizontal splits)
            grid_rows: Number of grid rows for visual prompt (vertical splits)
            fov_deg: Field of view angle in degrees for FoV extraction (square, default 120)
            fov_deg_subgrid: Field of view angle in degrees for FoV extraction (square, default 60)
            grid_type: Type of grid to draw (line or color)
            grid_alpha: Alpha transparency for color grid (0-255)
        """
        self.vlm_model = vlm_model
        self.rex_model = rex_model
        self.sam2_model = sam2_model
        self.output_root = Path(output_root)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.system_prompt = system_prompt
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.fov_deg = fov_deg
        self.fov_deg_subgrid = fov_deg_subgrid
        self.grid_type = grid_type
        self.grid_alpha = grid_alpha
        self.line_thickness = line_thickness
        self.font_size = font_size
        self.small_w = small_w
        self.small_h = small_h
        # Setup async I/O
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self.save_semaphore = threading.Semaphore(20)
        
    def async_save(self, func, *args, **kwargs):
        """Submit save tasks to thread pool with throttling."""
        self.save_semaphore.acquire()
        def wrapper():
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"\n[Error] Async save failed: {e}")
            finally:
                self.save_semaphore.release()
        self.executor.submit(wrapper)
        
    def async_save_metadata_and_metrics(self, output_dir, metadata):
        """Submit metadata save and metric calculation to thread pool."""
        self.save_semaphore.acquire()
        def wrapper():
            try:
                import cv2
                import time
                from metric import compute_iou
                gt_path = str(output_dir / "gt_mask.png")
                pred_path = str(output_dir / "mask.png")
                
                # Wait for mask.png to be saved by other async tasks
                for _ in range(60):
                    if os.path.exists(pred_path) and os.path.exists(gt_path):
                        break
                    time.sleep(0.5)
                
                if os.path.exists(gt_path) and os.path.exists(pred_path):
                    gt_arr = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    pred_arr = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                    
                    if gt_arr is not None and pred_arr is not None:
                        if gt_arr.shape != pred_arr.shape:
                            pred_arr = cv2.resize(pred_arr, (gt_arr.shape[1], gt_arr.shape[0]), interpolation=cv2.INTER_NEAREST)
                        gt_bool = gt_arr > 127
                        pred_bool = pred_arr > 127
                        iou, intersection, union = compute_iou(pred_bool, gt_bool)
                        
                        metadata['iou'] = float(iou)
                        metadata['intersection'] = int(intersection)
                        metadata['union'] = int(union)
            except Exception as e:
                print(f"\n  [Metric] Async metric calculation failed: {e}")
            finally:
                try:
                    with open(output_dir / "metadata.json", 'w') as f:
                        json.dump(metadata, f, indent=2)
                except Exception as e:
                    print(f"\n  [Error] Async metadata save failed: {e}")
                finally:
                    self.save_semaphore.release()
        self.executor.submit(wrapper)
    
    def _extract_grid_cell(self, image, grid_index):
        """
        Extract a single grid cell from the image by 2D crop.
        grid_index: 1-based index (1 to grid_cols * grid_rows).
        Returns: PIL Image of the cropped cell.
        """
        w, h = image.size
        cell_w = w // self.grid_cols
        cell_h = h // self.grid_rows
        row = (grid_index - 1) // self.grid_cols
        col = (grid_index - 1) % self.grid_cols
        x0 = col * cell_w
        y0 = row * cell_h
        x1 = min(x0 + cell_w, w)
        y1 = min(y0 + cell_h, h)
        return image.crop((x0, y0, x1, y1))
    
    def generate_description(self, image, question, output_dir):
        """
        Use VLM to generate object description based on affordance question.
        When refine is True and object occupies a single cell, performs a second
        VLM round on the cropped-and-enlarged cell with the same sub-grid specifications.
        
        Args:
            image: PIL Image
            question: Affordance question string
            
        Returns:
            tuple: (description, grid_boxes, refine, sub_grid_boxes)
        """
        if self.grid_type == "color":
            draw_img = draw_grid_color(image, grid_cols=self.grid_cols, grid_rows=self.grid_rows, alpha=self.grid_alpha, font_size=self.font_size)
        else:
            draw_img = draw_grid(image, grid_cols=self.grid_cols, grid_rows=self.grid_rows, font_size=self.font_size, line_thickness=self.line_thickness)

        # 保存缓存
        os.makedirs(output_dir/"cache/visualprompt", exist_ok=True)
        cache_path = output_dir/"cache/visualprompt/temp_clarity.jpg"
        draw_img.save(cache_path)

        if self.vlm_model is None:
            print("  [VLM] No VLM model provided, using question as description")
            return question, [], False, None
        
        print(f"  [VLM] Analyzing question: {question[:80]}...")
        
        # Construct prompt for VLM
        prompt = self.system_prompt.replace("TASK", question)
        
        response = self.vlm_model.generate(draw_img, prompt)

        # Save response
        with open(output_dir / "vlm_response.txt", "w") as f:
            f.write(response)

        extracted_data = extract_json(response)
        if extracted_data is None:
            print("  [VLM] Failed to parse JSON response")
            return question, [], False, None
        description = extracted_data.get("object_name", question)
        grid_boxes = extracted_data.get("grid_boxes", [])
        refine = extracted_data.get("small", False)
        print(f"  [VLM] Refine: {refine}, grid_boxes: {grid_boxes}")
        
        sub_grid_boxes = None
        if refine and len(grid_boxes) == 1:
            # Second-round VLM: crop cell, enlarge to original size, overlay specified sub-grid
            grid_idx = int(grid_boxes[0])
            cell_img = self._extract_grid_cell(image, grid_idx)
            # Enlarge cell back to original image size
            cell_enlarged = cell_img.resize(image.size, Image.LANCZOS)
            if self.grid_type == "color":
                cell_draw = draw_grid_color(
                    cell_enlarged,
                    grid_cols=self.grid_cols,
                    grid_rows=self.grid_rows,
                    alpha=self.grid_alpha,
                    font_size=self.font_size
                )
            else:
                cell_draw = draw_grid(
                    cell_enlarged,
                    grid_cols=self.grid_cols,
                    grid_rows=self.grid_rows,
                    font_size=self.font_size,
                    line_thickness=self.line_thickness
                )
            cache_refine_path = output_dir / "cache/visualprompt/temp_clarity_refine.jpg"
            cell_draw.save(cache_refine_path)
            
            sub_prompt = (
                f"This image is a zoomed-in crop of grid box {grid_idx} from a panoramic scene. "
                f"A {self.grid_cols}×{self.grid_rows} sub-grid (1-{self.grid_cols * self.grid_rows}) is overlaid. Task: {question}. Target object: {description}. "
                f"Identify which sub-grid cell(s) (1-{self.grid_cols * self.grid_rows}) contain the target object. "
                f"Output JSON: {{\"sub_grid_boxes\": [index1, index2, ...]}}"
            )
            print(f"  [VLM] Refine round: analyzing sub-grid for cell {grid_idx}")
            response_refine = self.vlm_model.generate(cell_draw, sub_prompt)
            with open(output_dir / "vlm_response_refine.txt", "w") as f:
                f.write(response_refine)
            
            data_refine = extract_json(response_refine)
            if data_refine and "sub_grid_boxes" in data_refine:
                sub_grid_boxes = data_refine["sub_grid_boxes"]
                print(f"  [VLM] Sub-grid boxes: {sub_grid_boxes}")
            else:
                print("  [VLM] Failed to parse sub_grid_boxes, using None")
        
        return description, grid_boxes, refine, sub_grid_boxes
    
    def detect_object(self, image, description, output_dir):
        """
        Use Rex-Omni to detect object in image.
        
        Args:
            image: PIL Image
            description: Object description from VLM
            output_dir: Directory to save visualization
            
        Returns:
            tuple: (bboxes, points) or (None, None) if failed
        """
        print(f"  [Rex] Detecting: {description}")
        
        try:
            # Detection
            results_bbox = self.rex_model.inference(
                images=image, 
                task="detection", 
                categories=[description]
            )
            result = results_bbox[0]
            predictions = result["extracted_predictions"]
            
            bboxes = None
            if description in predictions:
                bboxes = [pred['coords'] for pred in predictions[description]]
            
            # Pointing
            results_pointing = self.rex_model.inference(
                images=image,
                task="pointing",
                categories=[description]
            )
            result = results_pointing[0]
            predictions = result["extracted_predictions"]
            
            points = None
            if description in predictions:
                points = [pred['coords'] for pred in predictions[description]]
            
            if bboxes is None or points is None or len(bboxes) == 0 or len(points) == 0:
                print(f"  [Rex] Failed to detect object")
                return None, None
            
            # Save visualization
            self.async_save(
                save_image_with_points_and_box, 
                image, points, bboxes, 
                save_prefix=f"{output_dir}/rex_detection", 
                resize_to=(4000,2000)
            )
            
            return bboxes, points
            
        except Exception as e:
            print(f"  [Rex] Error: {e}")
            return None, None
    
    def crop_image_for_sam2(self, image, bboxes, points):
        """
        Crop image based on detection results for SAM2 processing.
        
        Args:
            image: PIL Image
            bboxes: List of bounding boxes
            points: List of points
            
        Returns:
            tuple: (cropped_image, adjusted_bboxes, adjusted_points, crop_box)
        """
        W, H = image.size
        crop_box = (0, 0, W, H)
        
        if not bboxes and not points:
            return image, bboxes, points, crop_box
        
        # Gather all coordinates
        all_x, all_y = [], []
        if bboxes:
            for b in bboxes:
                all_x.extend([b[0], b[2]])
                all_y.extend([b[1], b[3]])
        if points:
            for p in points:
                all_x.append(p[0])
                all_y.append(p[1])
        
        if not all_x or not all_y:
            return image, bboxes, points, crop_box
        
        # Calculate crop region
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        box_w = max_x - min_x
        box_h = max_y - min_y
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Make it square with margin
        long_side = max(box_w, box_h)
        margin = max(100, long_side * 0.2)
        square_side = long_side + 2 * margin
        half_side = square_side / 2
        
        crop_x1 = int(center_x - half_side)
        crop_y1 = int(center_y - half_side)
        crop_x2 = int(center_x + half_side)
        crop_y2 = int(center_y + half_side)
        
        # Adjust if out of bounds
        if crop_x1 < 0:
            offset = -crop_x1
            crop_x1 += offset
            crop_x2 += offset
        if crop_y1 < 0:
            offset = -crop_y1
            crop_y1 += offset
            crop_y2 += offset
        if crop_x2 > W:
            offset = crop_x2 - W
            crop_x1 -= offset
            crop_x2 -= offset
        if crop_y2 > H:
            offset = crop_y2 - H
            crop_y1 -= offset
            crop_y2 -= offset
        
        # Clamp to image bounds
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(W, crop_x2)
        crop_y2 = min(H, crop_y2)
        
        crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
        cropped_image = image.crop(crop_box)
        
        # Adjust coordinates
        adjusted_bboxes = None
        adjusted_points = None
        if bboxes:
            adjusted_bboxes = [
                [b[0]-crop_x1, b[1]-crop_y1, b[2]-crop_x1, b[3]-crop_y1] 
                for b in bboxes
            ]
        if points:
            adjusted_points = [
                [p[0]-crop_x1, p[1]-crop_y1] 
                for p in points
            ]
        
        return cropped_image, adjusted_bboxes, adjusted_points, crop_box
            
    
    def segment_object(self, cropped_image, original_image, bboxes, points, crop_box, output_dir):
        """
        Use SAM2 to segment object.
        
        Args:
            cropped_image: Cropped PIL Image for SAM2
            original_image: Original PIL Image
            bboxes: Bounding boxes in cropped image coordinates
            points: Points in cropped image coordinates
            crop_box: Crop coordinates (x1, y1, x2, y2)
            output_dir: Directory to save results
            
        Returns:
            str: Path to saved mask or None if failed
        """
        if not bboxes or not points:
            print("  [SAM2] Missing bboxes or points")
            return None
        
        print("  [SAM2] Running segmentation...")
        
        try:
            with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16):
                # Set image
                image_array = np.array(cropped_image.convert("RGB"))
                self.sam2_model.set_image(image_array)
                
                # Initialize mask
                mask_crop = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=bool)
                
                # Process each detection
                for bbox, point in zip(bboxes, points):
                    masks, scores, _ = self.sam2_model.predict(
                        point_coords=[point],
                        point_labels=[1],
                        box=bbox,
                        multimask_output=True,
                    )
                    # Use best mask
                    best_idx = torch.argmax(torch.tensor(scores))
                    best_mask = masks[best_idx].squeeze()
                    mask_crop = np.logical_or(mask_crop, best_mask)
                
                # Project back to original image size
                W_orig, H_orig = original_image.size
                mask_full = np.zeros((H_orig, W_orig), dtype=bool)
                
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
                h_crop, w_crop = mask_crop.shape
                target_h = min(h_crop, H_orig - crop_y1)
                target_w = min(w_crop, W_orig - crop_x1)
                
                if target_h > 0 and target_w > 0:
                    mask_full[crop_y1:crop_y1+target_h, crop_x1:crop_x1+target_w] = \
                        mask_crop[:target_h, :target_w]
                
                # Save mask
                mask_array = (mask_full * 255).astype(np.uint8)
                mask_image = Image.fromarray(mask_array)
                mask_path = output_dir / "mask.png"
                self.async_save(mask_image.save, str(mask_path))
                #mask_image.save(str(mask_path)) 
                # Save visualization
                self.async_save(
                    save_image_with_mask,
                    mask_full, original_image,
                    save_prefix=str(output_dir / "mask_visualization"),
                    borders=False,
                    resize_to=(4000,2000)
                )
                
                print(f"  [SAM2] Mask saved to {mask_path}")
                return str(mask_path), mask_image
                
        except Exception as e:
            print(f"  [SAM2] Error: {e}")
            import traceback
            traceback.print_exc()
            return None


    
   
    def process_sample(self, sample, skip_existing=False):
        """
        Process a single sample from the dataset.
        
        Args:
            sample: Dictionary containing sample information
            skip_existing: If True, skip samples with existing masks
            
        Returns:
            dict: Processing result
        """
        scene_type = sample['scene_type']
        scene_id = sample['scene_id']
        object_name = sample['object_name']
        question = sample['question']
        image_path = sample['image_path']
        mask_path = sample['mask_path']
        
        # Create output directory
        output_dir = self.output_root / scene_type / scene_id / object_name / question
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(mask_path, output_dir / "gt_mask.png")
        
        # Check if already processed
        if skip_existing and (output_dir / "mask.png").exists():
            return {"status": "skipped", "output_dir": str(output_dir)}
        
        print(f"\nProcessing: {scene_type}/{scene_id}/{object_name}")
        print(f"  Question: {question}")
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            # image_small = image.resize((image.size[0]//9, image.size[1]//9), Image.LANCZOS)
            # image_medium = image.resize((image.size[0]//3, image.size[1]//3), Image.LANCZOS)
            image_small = image.resize((self.small_w, self.small_h), Image.LANCZOS)
            image_medium = image.resize((self.small_w * self.grid_cols, self.small_h * self.grid_rows), Image.LANCZOS)
            # image_small = image.resize((4000, 2000), Image.LANCZOS)
            # image_medium = image.copy()
            #print(f"  [IMAGE SIZE] {image_small.size}")
            #clarity = self.clarity_description(image_small, question, output_dir)
        
            description, grid_boxes, refine, sub_grid_boxes = self.generate_description(image_small, question, output_dir)
            print(f"  [GRID BOXES] {grid_boxes}" + (f", sub_grid: {sub_grid_boxes}" if sub_grid_boxes else ""))
            print(f"  [object name] {description}")
            
            # Initialize refinement flag to avoid UnboundLocalError in return
            use_subgrid = False
            
            if not grid_boxes:
                print(f"  [Error] VLM returned empty grid_boxes")
                return {
                    "status": "failed",
                    "stage": "vlm_localization",
                    "output_dir": str(output_dir)
                }

            if len(grid_boxes) > 1:
                # Step 1: Generate description using VLM
                #description = self.generate_description(image_small, question)
                # Step 2: Detect object using Rex-Omni
                bboxes, points = self.detect_object(image_small, description, output_dir)
                if bboxes is None or points is None:
                    return {
                        "status": "failed",
                        "stage": "detection",
                        "output_dir": str(output_dir)
                    }
                # else:
                #     # Step 3: Crop image for SAM2
                #     cropped_image, adj_bboxes, adj_points, crop_box = \
                #         self.crop_image_for_sam2(image_small, bboxes, points)
                    
                #     # Step 4: Segment object using SAM2
                #     mask_path, _ = self.segment_object(
                #         cropped_image, image_small, adj_bboxes, adj_points, crop_box, output_dir)
            
                # Step 3: Crop image for SAM2
                cropped_image, adj_bboxes, adj_points, crop_box = \
                    self.crop_image_for_sam2(image_small, bboxes, points)
            
                # Step 4: Segment object using SAM2
                mask_path, _ = self.segment_object(
                    cropped_image, image_small, adj_bboxes, adj_points, crop_box, output_dir
                )
            else:
                img_cv = cv2.cvtColor(np.array(image_medium), cv2.COLOR_RGB2BGR)
                
                # Sub-grid zoom refinement logic
                use_subgrid = sub_grid_boxes and len(sub_grid_boxes) == 1
                current_fov_deg = self.fov_deg_subgrid if use_subgrid else self.fov_deg
                current_sub_idx = int(sub_grid_boxes[0]) if use_subgrid else None

                image_rotated = process_panorama(img_cv, int(grid_boxes[0]), sub_grid_index=current_sub_idx, grid_cols=self.grid_cols, grid_rows=self.grid_rows) #  time: 1.1s
                if image_rotated is None:
                    print(f"  [Error] Failed to process panorama")
                    return {
                        "status": "failed",
                        "stage": "panorama",
                        "output_dir": str(output_dir)
                    }
                else:
                    print(f"  [Success] Processed panorama")
                    cv2.imwrite(output_dir/"image_rotated.jpg", image_rotated)

                extracted_fov = extract_fov(image_rotated, fov_deg=current_fov_deg, output_dir=output_dir) # time: 0.03s
                if extracted_fov is None:
                    print(f"  [Error] Failed to extract FoV")
                    return {
                        "status": "failed",
                        "stage": "fov",
                        "output_dir": str(output_dir)
                    }
                else:
                    print(f"  [Success] Extracted FoV")
                    cv2.imwrite(output_dir/"extracted_fov.jpg", extracted_fov)
                    extracted_fov_rgb = cv2.cvtColor(extracted_fov, cv2.COLOR_BGR2RGB)
                    extracted_fov_pil = Image.fromarray(extracted_fov_rgb)
    
                #description = self.generate_description(extracted_fov_pil, question)
                #print(f"  [Description] {description}")
                
                # Step 2: Detect object using Rex-Omni
                bboxes, points = self.detect_object(extracted_fov_pil, description, output_dir)
                if bboxes is None or points is None:
                    print(f"  [Error] Failed to detect object")
                    return {
                        "status": "failed",
                        "stage": "detection",
                        "output_dir": str(output_dir)
                    }
                else:
                    # Step 3: Crop image for SAM2
                    cropped_image, adj_bboxes, adj_points, crop_box = \
                        self.crop_image_for_sam2(extracted_fov_pil, bboxes, points)
                
                    # Step 4: Segment object using SAM2
                    mask_path, mask_image = self.segment_object(
                        cropped_image, extracted_fov_pil, adj_bboxes, adj_points, crop_box, output_dir
                    )

                if mask_path:
                    # 11.33s -> 2s
                    start_time = time.perf_counter()
                    print(f"  [Success] Segmented object")
                    # 1. 读取生成的 Mask (对应 extracted_fov 的分辨率)
                    mask_fov = np.array(mask_image)

                    # 使用 image_medium 的分辨率做逆投影（比原图小 9 倍，速度快约 9 倍）
                    # 最后再 resize 回原图大小
                    pano_w_med, pano_h_med = image_medium.size  # PIL: (width, height)

                    # 2. 第一步还原：逆 FOV 投影 (透视 -> 旋转后的全景坐标)
                    black_bg = np.zeros((pano_h_med, pano_w_med, 3), dtype=np.uint8)

                    # 转换 mask 为 3 通道以便处理
                    if len(mask_fov.shape) == 2:
                        mask_fov_3ch = cv2.cvtColor(mask_fov, cv2.COLOR_GRAY2BGR)
                    else:
                        mask_fov_3ch = mask_fov

                    # remap maps 按 (pano_h, pano_w, fov_h, fov_w, fov_deg) 缓存，后续调用直接复用
                    mask_on_rotated_pano = restore_fov_to_panorama(
                        mask_fov_3ch,
                        pano_w_med,
                        pano_h_med,
                        fov_deg=current_fov_deg,
                        background_pano=black_bg,
                        output_dir=str(output_dir)
                    )

                    # 3. 第二步还原：逆全局旋转 (旋转后的全景坐标 -> 原始全景坐标)
                    # remap maps 按 (h, w, grid_index) 缓存
                    final_global_mask = restore_panorama(
                        mask_on_rotated_pano,
                        grid_index=int(grid_boxes[0]),
                        sub_grid_index=current_sub_idx,
                        grid_cols=self.grid_cols,
                        grid_rows=self.grid_rows,
                        output_dir=str(output_dir)
                    )

                    # 4. Resize 到原图分辨率后保存（保证与 gt_mask 尺寸一致）
                    orig_w, orig_h = image.size
                    # import pdb; pdb.set_trace()
                    if final_global_mask.shape[:2] != (orig_h, orig_w):
                        final_global_mask = cv2.resize(final_global_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                    final_mask_path = output_dir / "mask.png"
                    cv2.imwrite(str(final_mask_path), final_global_mask)
                    print(f"  [MultiScale] Final global mask saved to: {final_mask_path}")
                    end_time = time.perf_counter()
                    print(f"  Reproject Mask Time taken: {end_time - start_time} seconds")



            if mask_path is None:
                return {
                    "status": "failed",
                    "stage": "segmentation",
                    "output_dir": str(output_dir)
                }
            


            # Save metadata
            metadata = {
                "scene_type": scene_type,
                "scene_id": scene_id,
                "object_name": object_name,
                "question": question,
                "description": description,
                "grid_boxes": grid_boxes,
                "refine": refine,
                "sub_grid_boxes": sub_grid_boxes,
                "image_path": image_path,
                "mask_path": mask_path
            }
            
            self.async_save_metadata_and_metrics(output_dir, metadata)
            
            return {
                "status": "success",
                "output_dir": str(output_dir),
                "mask_path": mask_path,
                "refined": use_subgrid
            }
            
        except Exception as e:
            print(f"  [Error] {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
                "output_dir": str(output_dir)
            }
    
    def _prepare_and_call_vlm(self, sample):
        """
        Phase 1: Load images and call VLM API (possibly two rounds for refinement).
        I/O-bound (image decode + HTTP round-trip), safe to run from many threads
        so that vLLM's continuous batching can process them in parallel.
        """
        scene_type = sample['scene_type']
        scene_id = sample['scene_id']
        object_name = sample['object_name']
        question = sample['question']
        image_path = sample['image_path']
        mask_path_src = sample['mask_path']

        output_dir = self.output_root / scene_type / scene_id / object_name / question
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(mask_path_src, output_dir / "gt_mask.png")

        image = Image.open(image_path).convert("RGB")
        image_small = image.resize((self.small_w, self.small_h), Image.LANCZOS)
        # image_medium = image.resize((self.small_w * self.grid_cols, self.small_h * self.grid_rows), Image.LANCZOS)
        image_medium = image.copy()

        description, grid_boxes, refine, sub_grid_boxes = self.generate_description(image_small, question, output_dir)

        return {
            'sample': sample,
            'image': image,
            'image_small': image_small,
            'image_medium': image_medium,
            'output_dir': output_dir,
            'description': description,
            'grid_boxes': grid_boxes,
            'refine': refine,
            'sub_grid_boxes': sub_grid_boxes,
        }

    def _process_after_vlm(self, prepared):
        """
        Phase 2: Rex detection + SAM2 segmentation + post-processing.
        GPU-bound — must run sequentially on a single thread.
        """
        sample = prepared['sample']
        image = prepared['image']
        image_small = prepared['image_small']
        image_medium = prepared['image_medium']
        output_dir = prepared['output_dir']
        description = prepared['description']
        grid_boxes = prepared['grid_boxes']
        refine = prepared['refine']
        sub_grid_boxes = prepared['sub_grid_boxes']

        scene_type = sample['scene_type']
        scene_id = sample['scene_id']
        object_name = sample['object_name']
        question = sample['question']
        image_path = sample['image_path']

        print(f"\n[GPU] {scene_type}/{scene_id}/{object_name}")
        print(f"  Question: {question}")
        print(f"  [GRID BOXES] {grid_boxes}" + (f", sub_grid: {sub_grid_boxes}" if sub_grid_boxes else ""))
        print(f"  [object name] {description}")

        use_subgrid = False

        try:
            mask_path = None

            # If VLM provides exactly one grid box, use the zoom-in (FOV) pipeline
            if len(grid_boxes) == 1:
                img_cv = cv2.cvtColor(np.array(image_medium), cv2.COLOR_RGB2BGR)

                use_subgrid = sub_grid_boxes and len(sub_grid_boxes) == 1
                current_fov_deg = self.fov_deg_subgrid if use_subgrid else self.fov_deg
                current_sub_idx = int(sub_grid_boxes[0]) if use_subgrid else None

                image_rotated = process_panorama(img_cv, int(grid_boxes[0]), sub_grid_index=current_sub_idx, grid_cols=self.grid_cols, grid_rows=self.grid_rows)
                if image_rotated is None:
                    print(f"  [Error] Failed to process panorama")
                    return {"status": "failed", "stage": "panorama", "output_dir": str(output_dir)}

                print(f"  [Success] Processed panorama")
                cv2.imwrite(output_dir/"image_rotated.jpg", image_rotated)

                extracted_fov = extract_fov(image_rotated, fov_deg=current_fov_deg, output_dir=output_dir)
                if extracted_fov is None:
                    print(f"  [Error] Failed to extract FoV")
                    return {"status": "failed", "stage": "fov", "output_dir": str(output_dir)}

                print(f"  [Success] Extracted FoV")
                cv2.imwrite(output_dir/"extracted_fov.jpg", extracted_fov)
                extracted_fov_rgb = cv2.cvtColor(extracted_fov, cv2.COLOR_BGR2RGB)
                extracted_fov_pil = Image.fromarray(extracted_fov_rgb)

                bboxes, points = self.detect_object(extracted_fov_pil, description, output_dir)
                if bboxes is None or points is None:
                    print(f"  [Error] Failed to detect object")
                    return {"status": "failed", "stage": "detection", "output_dir": str(output_dir)}

                cropped_image, adj_bboxes, adj_points, crop_box = \
                    self.crop_image_for_sam2(extracted_fov_pil, bboxes, points)

                mask_path, mask_image = self.segment_object(
                    cropped_image, extracted_fov_pil, adj_bboxes, adj_points, crop_box, output_dir)

                if mask_path:
                    start_time = time.perf_counter()
                    print(f"  [Success] Segmented object")
                    mask_fov = np.array(mask_image)
                    pano_w_med, pano_h_med = image_medium.size

                    black_bg = np.zeros((pano_h_med, pano_w_med, 3), dtype=np.uint8)
                    if len(mask_fov.shape) == 2:
                        mask_fov_3ch = cv2.cvtColor(mask_fov, cv2.COLOR_GRAY2BGR)
                    else:
                        mask_fov_3ch = mask_fov

                    mask_on_rotated_pano = restore_fov_to_panorama(
                        mask_fov_3ch, pano_w_med, pano_h_med,
                        fov_deg=current_fov_deg, background_pano=black_bg,
                        output_dir=str(output_dir))

                    final_global_mask = restore_panorama(
                        mask_on_rotated_pano, grid_index=int(grid_boxes[0]),
                        sub_grid_index=current_sub_idx,
                        grid_cols=self.grid_cols, grid_rows=self.grid_rows,
                        output_dir=str(output_dir))

                    orig_w, orig_h = image.size
                    if final_global_mask.shape[:2] != (orig_h, orig_w):
                        final_global_mask = cv2.resize(
                            final_global_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                    final_mask_path = output_dir / "mask.png"
                    cv2.imwrite(str(final_mask_path), final_global_mask)
                    print(f"  [MultiScale] Final global mask saved to: {final_mask_path}")
                    end_time = time.perf_counter()
                    print(f"  Reproject Mask Time taken: {end_time - start_time} seconds")

            # If VLM provides multiple grid boxes OR none, process the full panorama (resized)
            else:
                if not grid_boxes:
                    print(f"  [Warning] VLM returned empty grid_boxes, defaulting to full image processing")
                
                bboxes, points = self.detect_object(image_small, description, output_dir)
                if bboxes is None or points is None:
                    return {"status": "failed", "stage": "detection", "output_dir": str(output_dir)}

                cropped_image, adj_bboxes, adj_points, crop_box = \
                    self.crop_image_for_sam2(image_small, bboxes, points)

                mask_path, _ = self.segment_object(
                    cropped_image, image_small, adj_bboxes, adj_points, crop_box, output_dir)

            if mask_path is None:
                return {"status": "failed", "stage": "segmentation", "output_dir": str(output_dir)}

            metadata = {
                "scene_type": scene_type,
                "scene_id": scene_id,
                "object_name": object_name,
                "question": question,
                "description": description,
                "grid_boxes": grid_boxes,
                "refine": refine,
                "sub_grid_boxes": sub_grid_boxes,
                "image_path": image_path,
                "mask_path": mask_path
            }

            self.async_save_metadata_and_metrics(output_dir, metadata)

            return {"status": "success", "output_dir": str(output_dir), "mask_path": mask_path, "refined": use_subgrid}

        except Exception as e:
            print(f"  [Error] {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e), "output_dir": str(output_dir)}

    def run_batch(self, dataset_root, shuffle=False, skip_existing=False, limit=None, vlm_concurrency=32):
        """
        Process dataset with parallel VLM calls pipelined into sequential GPU processing.

        VLM API calls are I/O-bound and sent concurrently so vLLM's continuous
        batching can maximize GPU utilization on the server side.
        Rex + SAM2 run sequentially on the local GPU.

        Args:
            dataset_root: Root directory of the dataset
            shuffle: Whether to shuffle the dataset
            skip_existing: Whether to skip samples with existing masks
            limit: Maximum number of samples to process (None for all)
            vlm_concurrency: Number of concurrent VLM API requests
        """
        from queue import Queue, Empty

        print("=" * 80)
        print("PAP-12K Batch Processing (Parallel VLM Pipeline)")
        print("=" * 80)

        dataset = PAP_Dataset(dataset_root)
        stats = dataset.get_statistics()
        print(f"Total Images: {stats['total_images']}")
        print(f"Total QA Pairs: {stats['total_qa_pairs']}")

        all_samples = []
        skipped_count = 0
        for i, sample in enumerate(dataset.get_data(shuffle=shuffle)):
            if limit and i >= limit:
                break
            if skip_existing:
                output_dir = self.output_root / sample['scene_type'] / sample['scene_id'] / \
                             sample['object_name'] / sample['question']
                if (output_dir / "mask.png").exists():
                    skipped_count += 1
                    continue
            all_samples.append(sample)

        total = len(all_samples)
        results = {"success": 0, "failed": 0, "skipped": skipped_count, "error": 0, "refined": 0}

        if total == 0:
            print("No new samples to process.")
            return results

        effective_concurrency = vlm_concurrency if self.vlm_model is not None else 1
        print(f"\nSamples to process: {total}  (skipped: {skipped_count})")
        print(f"VLM concurrency: {effective_concurrency}")
        print("=" * 80)

        if effective_concurrency <= 1:
            with tqdm(total=total, desc="Processing") as pbar:
                for sample in all_samples:
                    result = self.process_sample(sample, skip_existing=False)
                    results[result["status"]] += 1
                    if result.get("refined"):
                        results["refined"] += 1
                    pbar.set_postfix({"✓": results["success"], "ref": results["refined"], "✗": results["failed"]})
                    pbar.update(1)
        else:
            gpu_queue = Queue(maxsize=16)
            vlm_done = threading.Event()

            def _vlm_producer():
                with concurrent.futures.ThreadPoolExecutor(max_workers=effective_concurrency) as pool:
                    future_to_sample = {
                        pool.submit(self._prepare_and_call_vlm, s): s
                        for s in all_samples
                    }
                    for future in concurrent.futures.as_completed(future_to_sample):
                        sample = future_to_sample[future]
                        try:
                            prepared = future.result()
                            gpu_queue.put(prepared)
                        except Exception as e:
                            print(f"\n  [VLM Error] {sample['scene_type']}/{sample['scene_id']}"
                                  f"/{sample['object_name']}: {e}")
                            gpu_queue.put({'_vlm_error': True, 'error': str(e)})
                vlm_done.set()

            producer = threading.Thread(target=_vlm_producer, daemon=True)
            producer.start()

            with tqdm(total=total, desc="Processing") as pbar:
                while True:
                    try:
                        item = gpu_queue.get(timeout=2.0)
                    except Empty:
                        if vlm_done.is_set() and gpu_queue.empty():
                            break
                        continue

                    if item.get('_vlm_error'):
                        results["error"] += 1
                    else:
                        result = self._process_after_vlm(item)
                        results[result["status"]] += 1
                        if result.get("refined"):
                            results["refined"] += 1

                    pbar.set_postfix({
                        "✓": results["success"],
                        "ref": results["refined"],
                        "✗": results["failed"],
                        "err": results["error"],
                        "Q": gpu_queue.qsize(),
                    })
                    pbar.update(1)

            producer.join()

        print("\nWaiting for background save tasks...")
        self.executor.shutdown(wait=True)

        print("\n" + "=" * 80)
        print("Processing Summary")
        print("=" * 80)
        print(f"Success:  {results['success']} (Refined: {results['refined']})")
        print(f"Failed:   {results['failed']}")
        print(f"Skipped:  {results['skipped']}")
        print(f"Errors:   {results['error']}")
        print("=" * 80)

        return results


def main():
    parser = argparse.ArgumentParser(description="Panoramic Affordance Prediction")
    
    parser.add_argument("--dataset_root", type=str, default="./PAP-12K", help="Root directory of the dataset")
    parser.add_argument("--output", type=str, default="output", help="Output directory for results")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset")
    parser.add_argument("--resume", action="store_true", help="Resume from last run: skip samples that already have a mask.png output")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to process")

    parser.add_argument("--vlm_api_url", type=str, default=None, help="API server base URL (e.g., http://localhost:8000 or https://api.openai.com)")
    parser.add_argument("--vlm_model_name", type=str, default="qwen3-vl-4b", help="Model name for the API")
    parser.add_argument("--vlm_api_key", type=str, default=None, help="API key for authentication (optional, not needed for local vLLM servers)")
    parser.add_argument("--vlm_concurrency", type=int, default=8, help="Number of concurrent VLM API requests (higher = better vLLM batching)")

    parser.add_argument("--rex_model", type=str, default="IDEA-Research/Rex-Omni", help="Rex-Omni model path")
    parser.add_argument("--sam2_model", type=str, default="facebook/sam2.1-hiera-large", help="SAM2 model path")
    parser.add_argument("--system_prompt_path", type=str, default="system_prompt/visual_grid_prompting.md", help="System prompt for VLM")

    args = parser.parse_args()
    
    # Initialize models
    print("Initializing models...")
    print("-" * 80)
    
    # Initialize VLM client if API URL is provided
    vlm_client = None
    if args.vlm_api_url:
        print(f"Connecting to API server at {args.vlm_api_url}...")
        try:
            vlm_client = VLMClient(
                api_url=args.vlm_api_url,
                model_name=args.vlm_model_name,
                api_key=args.vlm_api_key
            )
        except Exception as e:
            print(f"⚠ Failed to initialize VLM client: {e}")
            print("Continuing without VLM (will use questions as descriptions)")
    else:
        print("No VLM API URL provided, will use questions as descriptions")
    
    print("Loading Rex-Omni...")
    rex_model = RexOmniWrapper(
        model_path=args.rex_model,
        backend="transformers",
        max_tokens=4096,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )
    
    print("Loading SAM2...")
    sam2_model = SAM2ImagePredictor.from_pretrained(args.sam2_model)
    
    print("Models loaded successfully!")
    print("-" * 80)
    
    # Initialize pipeline
    system_prompt = Path(args.system_prompt_path).read_text()
    print(f"System prompt: {system_prompt}")

    pipeline = PanoAff(
        vlm_model=vlm_client,
        rex_model=rex_model,
        sam2_model=sam2_model,
        output_root=args.output,
        num_workers=8,
        system_prompt=system_prompt
    )
    
    # Get absolute path
    # dataset_root = Path("/hpc2hdd/home/zzhang300/zixin_workspace/ICML2026/PanoAff/PanoAff-1K")
    dataset_root = args.dataset_root
    
    # Run batch processing
    pipeline.run_batch(
        dataset_root=str(dataset_root),
        shuffle=args.shuffle,
        skip_existing=args.resume,
        limit=args.limit,
        vlm_concurrency=args.vlm_concurrency
    )


if __name__ == "__main__":
    main()
