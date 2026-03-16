import argparse
import os
import sys
import json
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# Add parent directories to path to ensure imports work from demo folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

from sam2.sam2_image_predictor import SAM2ImagePredictor
from rex_omni import RexOmniWrapper
from utils.pano_utils import process_panorama, restore_panorama, extract_fov, restore_fov_to_panorama

# Import classes from run.py
from run import PanoAff, VLMClient

class DemoPanoAff(PanoAff):
    def process_demo_image(self, image_path, question, output_dir):
        """
        Process a single image and question for demonstration.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing demo image: {image_path}")
        print(f"  Question: {question}")
        print(f"  Output directory: {output_dir}")
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_small = image.resize((self.small_w, self.small_h), Image.LANCZOS)
            image_medium = image.resize((self.small_w * self.grid_cols, self.small_h * self.grid_rows), Image.LANCZOS)
        
            # Step 1: Generate description and localization using VLM
            description, grid_boxes, refine, sub_grid_boxes = self.generate_description(image_small, question, output_dir)
            print(f"  [GRID BOXES] {grid_boxes}" + (f", sub_grid: {sub_grid_boxes}" if sub_grid_boxes else ""))
            print(f"  [object name] {description}")
            
            use_subgrid = False
            final_mask_path = None
            mask_path = None
            
            if not grid_boxes:
                print(f"  [Error] VLM returned empty grid_boxes, falling back to full image processing")
                bboxes, points = self.detect_object(image_small, description, output_dir)
                if bboxes is None or points is None:
                    return {"status": "failed", "stage": "detection", "output_dir": str(output_dir)}
                
                cropped_image, adj_bboxes, adj_points, crop_box = self.crop_image_for_sam2(image_small, bboxes, points)
                mask_path, _ = self.segment_object(cropped_image, image_small, adj_bboxes, adj_points, crop_box, output_dir)
                final_mask_path = mask_path
                
            elif len(grid_boxes) > 1:
                bboxes, points = self.detect_object(image_small, description, output_dir)
                if bboxes is None or points is None:
                    return {"status": "failed", "stage": "detection", "output_dir": str(output_dir)}
                
                cropped_image, adj_bboxes, adj_points, crop_box = self.crop_image_for_sam2(image_small, bboxes, points)
                mask_path, _ = self.segment_object(cropped_image, image_small, adj_bboxes, adj_points, crop_box, output_dir)
                final_mask_path = mask_path
                
            else:
                img_cv = cv2.cvtColor(np.array(image_medium), cv2.COLOR_RGB2BGR)
                
                # Sub-grid zoom refinement logic
                use_subgrid = sub_grid_boxes and len(sub_grid_boxes) == 1
                current_fov_deg = self.fov_deg_subgrid if use_subgrid else self.fov_deg
                current_sub_idx = int(sub_grid_boxes[0]) if use_subgrid else None

                image_rotated = process_panorama(img_cv, int(grid_boxes[0]), sub_grid_index=current_sub_idx, grid_cols=self.grid_cols, grid_rows=self.grid_rows)
                if image_rotated is None:
                    print(f"  [Error] Failed to process panorama")
                    return {"status": "failed", "stage": "panorama", "output_dir": str(output_dir)}
                else:
                    print(f"  [Success] Processed panorama")
                    cv2.imwrite(str(output_dir/"image_rotated.jpg"), image_rotated)

                extracted_fov = extract_fov(image_rotated, fov_deg=current_fov_deg, output_dir=output_dir)
                if extracted_fov is None:
                    print(f"  [Error] Failed to extract FoV")
                    return {"status": "failed", "stage": "fov", "output_dir": str(output_dir)}
                else:
                    print(f"  [Success] Extracted FoV")
                    cv2.imwrite(str(output_dir/"extracted_fov.jpg"), extracted_fov)
                    extracted_fov_rgb = cv2.cvtColor(extracted_fov, cv2.COLOR_BGR2RGB)
                    extracted_fov_pil = Image.fromarray(extracted_fov_rgb)
                
                # Step 2: Detect object using Rex-Omni
                bboxes, points = self.detect_object(extracted_fov_pil, description, output_dir)
                if bboxes is None or points is None:
                    print(f"  [Error] Failed to detect object")
                    return {"status": "failed", "stage": "detection", "output_dir": str(output_dir)}
                
                # Step 3: Crop image for SAM2
                cropped_image, adj_bboxes, adj_points, crop_box = self.crop_image_for_sam2(extracted_fov_pil, bboxes, points)
            
                # Step 4: Segment object using SAM2
                mask_path, mask_image = self.segment_object(cropped_image, extracted_fov_pil, adj_bboxes, adj_points, crop_box, output_dir)

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
                        fov_deg=current_fov_deg, background_pano=black_bg, output_dir=str(output_dir)
                    )

                    final_global_mask = restore_panorama(
                        mask_on_rotated_pano, grid_index=int(grid_boxes[0]),
                        sub_grid_index=current_sub_idx, grid_cols=self.grid_cols, grid_rows=self.grid_rows,
                        output_dir=str(output_dir)
                    )

                    orig_w, orig_h = image.size
                    if final_global_mask.shape[:2] != (orig_h, orig_w):
                        final_global_mask = cv2.resize(final_global_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                    final_mask_path = output_dir / "mask.png"
                    cv2.imwrite(str(final_mask_path), final_global_mask)
                    print(f"  [MultiScale] Final global mask saved to: {final_mask_path}")
                    end_time = time.perf_counter()
                    print(f"  Reproject Mask Time taken: {end_time - start_time:.4f} seconds")


            if not mask_path:
                return {"status": "failed", "stage": "segmentation", "output_dir": str(output_dir)}
            
            # Use original mask as final if not multiscale
            if final_mask_path is None:
                final_mask_path = mask_path

            # Save metadata
            metadata = {
                "question": question,
                "description": description,
                "grid_boxes": grid_boxes,
                "refine": refine,
                "sub_grid_boxes": sub_grid_boxes,
                "image_path": str(image_path),
                "mask_path": str(final_mask_path)
            }
            
            with open(output_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Wait for any background threads saving visualizations
            self.executor.shutdown(wait=True)
            
            print(f"\n[Success] Processing complete! Results saved in {output_dir}")
            return {
                "status": "success",
                "output_dir": str(output_dir),
                "mask_path": str(final_mask_path),
                "refined": use_subgrid
            }
            
        except Exception as e:
            print(f"  [Error] {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e), "output_dir": str(output_dir)}


def main():
    parser = argparse.ArgumentParser(description="PanoAff Demo: Affordance-based object detection and segmentation on a single image")
    
    # Core inputs
    parser.add_argument("--image_path", type=str, default="kitchen.jpg", help="Path to the input panorama image")
    parser.add_argument("--question", type=str, default=None, help="Affordance question (e.g. 'Where can I sit?')")
    parser.add_argument("--question_file", type=str, default="kitchen.txt", help="Path to the text file containing the question")
    parser.add_argument("--output", type=str, default="demo_output", help="Output directory for results")
    
    # Model parameters
    parser.add_argument("--vlm_api_url", type=str, default=None, help="API server base URL (e.g., http://localhost:8000)")
    parser.add_argument("--vlm_model_name", type=str, default="qwen3-vl-4b", help="Model name for the API")
    parser.add_argument("--vlm_api_key", type=str, default=None, help="API key for authentication")
    parser.add_argument("--rex_model", type=str, default="IDEA-Research/Rex-Omni", help="Rex-Omni model path")
    parser.add_argument("--sam2_model", type=str, default="facebook/sam2.1-hiera-large", help="SAM2 model path")
    parser.add_argument("--system_prompt_path", type=str, default=os.path.join(parent_dir, "system_prompt/combined_0225.md"), help="System prompt for VLM")
    
    # Grid and resolution parameters
    parser.add_argument("--grid_cols", type=int, default=4, help="Number of grid columns")
    parser.add_argument("--grid_rows", type=int, default=3, help="Number of grid rows")
    parser.add_argument("--fov_deg", type=float, default=90, help="Field of view angle in degrees")
    parser.add_argument("--fov_deg_subgrid", type=float, default=60, help="Field of view angle for subgrid")
    parser.add_argument("--grid_type", type=str, default="line", choices=["line", "color"], help="Grid drawing type")
    parser.add_argument("--grid_alpha", type=int, default=100, help="Alpha value for grid color")
    parser.add_argument("--line_thickness", type=int, default=5, help="Line thickness for grid")
    parser.add_argument("--font_size", type=int, default=50, help="Font size for grid labels")
    parser.add_argument("--small_w", type=int, default=2000, help="Width of image_small")
    parser.add_argument("--small_h", type=int, default=1000, help="Height of image_small")
    
    args = parser.parse_args()
    
    # Read question from file if not provided as argument
    question = args.question
    if question is None:
        if os.path.exists(args.question_file):
            with open(args.question_file, "r", encoding="utf-8") as f:
                question = f.read().strip()
            print(f"Loaded question from {args.question_file}: '{question}'")
        else:
            print(f"Error: Question not provided and file '{args.question_file}' not found.")
            return
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
        
    print("Initializing models...")
    print("-" * 80)
    
    vlm_client = None
    if args.vlm_api_url:
        print(f"Connecting to VLM API server at {args.vlm_api_url}...")
        try:
            vlm_client = VLMClient(
                api_url=args.vlm_api_url,
                model_name=args.vlm_model_name,
                api_key=args.vlm_api_key
            )
        except Exception as e:
            print(f"⚠ Failed to initialize VLM client: {e}")
            print("Continuing without VLM (will use question directly as description)")
    else:
        print("No VLM API URL provided, using question as object description")
    
    print(f"Loading Rex-Omni from {args.rex_model}...")
    rex_model = RexOmniWrapper(
        model_path=args.rex_model,
        backend="transformers",
        max_tokens=4096,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )
    
    print(f"Loading SAM2 from {args.sam2_model}...")
    sam2_model = SAM2ImagePredictor.from_pretrained(args.sam2_model)
    
    print("Models loaded successfully!")
    print("-" * 80)
    
    system_prompt = ""
    if os.path.exists(args.system_prompt_path):
        system_prompt = Path(args.system_prompt_path).read_text()
    else:
        print(f"Warning: System prompt file not found at {args.system_prompt_path}. Using empty system prompt.")

    pipeline = DemoPanoAff(
        vlm_model=vlm_client,
        rex_model=rex_model,
        sam2_model=sam2_model,
        output_root=args.output,
        num_workers=8,
        system_prompt=system_prompt,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
        fov_deg=args.fov_deg,
        fov_deg_subgrid=args.fov_deg_subgrid,
        grid_type=args.grid_type,
        grid_alpha=args.grid_alpha,
        line_thickness=args.line_thickness,
        font_size=args.font_size,
        small_w=args.small_w,
        small_h=args.small_h
    )
    
    # Generate timestamped output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_name = Path(args.image_path).stem
    out_dir = Path(args.output) / f"{image_name}_{timestamp}"
    
    pipeline.process_demo_image(args.image_path, question, out_dir)


if __name__ == "__main__":
    main()
