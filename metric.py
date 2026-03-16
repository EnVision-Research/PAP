import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate IoU metrics from mask images")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory containing the output results")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of worker processes (default: 16)")
    parser.add_argument("--force_recalc", action="store_true", help="Force recalculation and ignore existing JSON metrics")
    parser.add_argument("--difficult_json", type=str, default="difficult_cases.json", help="Path to difficult_cases.json for separate difficult metric calculation")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to CSV file to append metrics")
    return parser.parse_args()

def compute_iou(pred_mask, gt_mask):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    """
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    
    if union == 0:
        return 0.0, 0, 0
    
    iou = intersection / union
    return iou, int(intersection), int(union)

def process_single_case(case_info):
    """
    Process a single case: load images or JSON cache, compute IoU, and save cache.
    case_info is a tuple: (folder_path, gt_path, pred_path, output_dir_str, force_recalc)
    """
    cv2.setNumThreads(0)
    
    folder, gt_path, pred_path, output_dir_str, force_recalc = case_info
    json_path = Path(folder) / "metrics.json"
    # print(json_path)
    
    try:
        try:
            rel_path = str(Path(folder).relative_to(output_dir_str))
        except ValueError:
            rel_path = str(folder)

        # 1. 如果不强制重新计算，且 metrics.json 存在，则尝试直接读取缓存
        if not force_recalc and json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 校验 JSON 文件中是否有我们需要的键
                if all(k in data for k in ['iou', 'intersection', 'union']):
                    return {
                        'status': 'success',
                        'id': rel_path,
                        'iou': float(data['iou']),
                        'intersection': int(data['intersection']),
                        'union': int(data['union'])
                    }
            except Exception as e:
                # 如果 JSON 读取失败（比如文件损坏），忽略错误，继续往下执行重新计算
                pass

        # 2. 如果需要计算（无缓存、被强制或者缓存读取失败）
        gt_path_str = str(gt_path)
        pred_path_str = str(pred_path)
        
        gt_arr = cv2.imread(gt_path_str, cv2.IMREAD_GRAYSCALE)
        pred_arr = cv2.imread(pred_path_str, cv2.IMREAD_GRAYSCALE)
        
        if gt_arr is None:
            if not os.path.exists(gt_path_str):
                raise ValueError(f"GT file not found: {gt_path_str}")
            raise ValueError(f"Could not read GT image (None): {gt_path_str}")
            
        if pred_arr is None:
            if not os.path.exists(pred_path_str):
                raise ValueError(f"Pred file not found: {pred_path_str}")
            raise ValueError(f"Could not read Pred image (None): {pred_path_str}")
            
        if gt_arr.shape != pred_arr.shape:
            pred_arr = cv2.resize(pred_arr, (gt_arr.shape[1], gt_arr.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        gt_bool = gt_arr > 127
        pred_bool = pred_arr > 127
        
        iou, intersection, union = compute_iou(pred_bool, gt_bool)
        
        # 3. 将计算结果保存到 metrics.json 中
        metrics_data = {
            'iou': float(iou),
            'intersection': int(intersection),
            'union': int(union)
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=4)
        
        return {
            'status': 'success',
            'id': rel_path,
            'iou': iou,
            'intersection': intersection,
            'union': union
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'folder': str(folder),
            'error': str(e)
        }

def find_case_folders(root_dir):
    """
    Recursively find folders containing both gt_mask.png and mask.png (or pred_mask.png)
    """
    print("Searching for mask files... (this might take a moment)", flush=True)
    case_folders = []
    
    root_path = Path(root_dir)
    gt_masks = list(root_path.rglob("gt_mask.png"))
    
    for gt_path in gt_masks:
        folder = gt_path.parent
        pred_path = folder / "mask.png"
        if not pred_path.exists():
            pred_path = folder / "pred_mask.png"
            
        if pred_path.exists():
            case_folders.append((folder, gt_path, pred_path))
            
    return case_folders

def calculate_metrics(output_dir, num_workers=16, force_recalc=False, difficult_json="difficult_cases.json", csv_path=None):
    print(f"Scanning for cases in {output_dir}...")
    cases = find_case_folders(output_dir)
    
    difficult_cases_set = set()
    if os.path.exists(difficult_json):
        try:
            with open(difficult_json, 'r', encoding='utf-8') as f:
                difficult_data = json.load(f)
                for item in difficult_data:
                    case_path = f"{item['scene_type']}/{item['scene_id']}/{item['object_name']}/{item['question']}"
                    difficult_cases_set.add(case_path)
            print(f"Loaded {len(difficult_cases_set)} unique difficult cases from {difficult_json}.")
        except Exception as e:
            print(f"Warning: Could not read {difficult_json}: {e}")
    
    if not cases:
        print(f"No valid cases found in {output_dir}")
        return

    print(f"Found {len(cases)} cases. Calculating metrics with {num_workers} workers...")
    
    # 将 force_recalc 状态也传入 worker
    work_items = [(folder, gt, pred, str(output_dir), force_recalc) for folder, gt, pred in cases]
    
    all_ious = []
    total_intersection = 0
    total_union = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_case, item) for item in work_items]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
                
                if result['status'] == 'success':
                    all_ious.append(result)
                    total_intersection += result['intersection']
                    total_union += result['union']
                else:
                    pass
            except Exception as e:
                print(f"Worker exception: {e}")

    if not all_ious:
        print("No metrics calculated.")
        return

    print("Aggregating results...")
    
    def print_metrics(ious_list, name):
        if not ious_list:
            print(f"\nNo data for {name}.")
            return None
            
        gIoU = np.mean([item['iou'] for item in ious_list])
        t_intersection = sum([item['intersection'] for item in ious_list])
        t_union = sum([item['union'] for item in ious_list])
        cIoU = t_intersection / t_union if t_union > 0 else 0
        
        p_50 = np.mean([1 if item['iou'] > 0.5 else 0 for item in ious_list])
        
        iou_array = np.array([item['iou'] for item in ious_list])
        thresholds = np.arange(0.5, 0.96, 0.05)
        p_thresholds = [(iou_array > t).mean() for t in thresholds]
        p_50_95 = np.mean(p_thresholds)
        
        print(f"\nEvaluation Results: {name}")
        print("="*30)
        print(f"Total Samples: {len(ious_list)}")
        print(f"gIoU (Mean IoU): {gIoU:.4f}")
        print(f"cIoU (Cumulative IoU): {cIoU:.4f}")
        print(f"P@50: {p_50:.4f}")
        print(f"P@50:95: {p_50_95:.4f}")
        print("="*30)
        
        return {
            "Subset": name,
            "Total Samples": len(ious_list),
            "gIoU": round(gIoU, 4),
            "cIoU": round(cIoU, 4),
            "P@50": round(p_50, 4),
            "P@50:95": round(p_50_95, 4)
        }

    results = []
    
    res_all = print_metrics(all_ious, "All Cases")
    if res_all:
        results.append(res_all)
    
    if difficult_cases_set:
        # Match using rel_path format: scene_type/scene_id/object_name/question
        difficult_ious = [item for item in all_ious if str(item['id']).replace('\\', '/') in difficult_cases_set]
        res_diff = print_metrics(difficult_ious, "Difficult Cases Only")
        if res_diff:
            results.append(res_diff)
        
        # We can also compute for easy cases
        easy_ious = [item for item in all_ious if str(item['id']).replace('\\', '/') not in difficult_cases_set]
        res_easy = print_metrics(easy_ious, "Easy Cases Only")
        if res_easy:
            results.append(res_easy)
            
    if csv_path and results:
        import csv
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["Output Dir", "Subset", "Total Samples", "gIoU", "cIoU", "P@50", "P@50:95"])
            if not file_exists:
                writer.writeheader()
            for r in results:
                r["Output Dir"] = str(output_dir)
                writer.writerow(r)

if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True) 
    args = parse_args()
    calculate_metrics(args.output_dir, args.num_workers, args.force_recalc, args.difficult_json, args.csv_path)