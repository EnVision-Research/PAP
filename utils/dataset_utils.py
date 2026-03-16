import os
import random
from pathlib import Path

class PAP_Dataset:
    def __init__(self, root_dir):
        """
        Initialize the dataset reader.
        
        Args:
            root_dir (str): Path to the root directory containing the dataset 
        """
        self.root_dir = Path(root_dir)

    def _yield_samples(self):
        """
        Internal generator that yields data samples in sequential order.
        """
        # Iterate over scene types (e.g., bathroom, kitchen) - sorted for deterministic order
        for scene_type_path in sorted(self.root_dir.iterdir()):
            if not scene_type_path.is_dir():
                continue
            
            scene_type = scene_type_path.name
            
            # Iterate over scene IDs (e.g., 0003) - sorted
            for scene_id_path in sorted(scene_type_path.iterdir()):
                if not scene_id_path.is_dir():
                    continue
                
                scene_id = scene_id_path.name
                
                # Find the scene image (e.g., 0003.jpg)
                image_path = scene_id_path / f"{scene_id}.jpg"
                if not image_path.exists():
                    # Try other extensions if needed, or skip
                    # For now assuming .jpg based on observation
                    # print(f"No image found for {scene_id}")
                    continue
                
                # Iterate over object directories - sorted
                for object_path in sorted(scene_id_path.iterdir()):
                    if not object_path.is_dir():
                        continue
                    
                    object_name = object_path.name
                    
                    # Determine mask path
                    mask_refined = object_path / "mask_refined.png"
                    mask_normal = object_path / "mask.png"
                    
                    final_mask_path = None
                    if mask_refined.exists():
                        final_mask_path = mask_refined
                    elif mask_normal.exists():
                        final_mask_path = mask_normal
                    else:
                        # Skip if no mask found
                        # print(f"No mask found for {object_name} in {scene_id} {scene_type}")
                        continue
                    
                    # Read affordance questions
                    question_file = object_path / "affordance_question.txt"
                    if not question_file.exists():
                        # print(f"No question file found for {object_name}")
                        continue
                    
                    try:
                        with open(question_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            
                        for line in lines:
                            question = line.strip()
                            if not question:
                                continue
                                
                            # Each line is a separate case
                            yield {
                                'scene_type': scene_type,
                                'scene_id': scene_id,
                                'object_name': object_name,
                                'image_path': str(image_path.absolute()),
                                'mask_path': str(final_mask_path.absolute()),
                                'question': question
                            }
                            
                    except Exception as e:
                        print(f"Error reading {question_file}: {e}")

    def get_data(self, shuffle=False):
        """
        Generator that yields data samples.
        
        Args:
            shuffle (bool): If True, yields samples in random order. 
                            If False (default), yields in sequential (filesystem/alphabetical) order.
        
        Yields:
            dict: A dictionary containing details for each case.
        """
        if shuffle:
            # Collect all samples into a list first
            all_samples = list(self._yield_samples())
            random.shuffle(all_samples)
            yield from all_samples
        else:
            # Yield as we go
            yield from self._yield_samples()

    def get_statistics(self):
        """
        Calculate statistics for the dataset.
        
        Returns:
            dict: A dictionary containing 'total_images' and 'total_qa_pairs'.
        """
        unique_images = set()
        total_qa_pairs = 0
        
        # Iterate through all data to count (using sequential order is slightly faster/same)
        for item in self._yield_samples():
            unique_images.add(item['image_path'])
            total_qa_pairs += 1
            
        return {
            'total_images': len(unique_images),
            'total_qa_pairs': total_qa_pairs
        }

if __name__ == "__main__":
    # Automatic path detection
    dataset_root = Path('')
    
    if dataset_root.exists():
        reader = PAP_Dataset(dataset_root)
        print(f"Reading data from {dataset_root}...")
        
        # Calculate statistics
        stats = reader.get_statistics()
        print(f"Total Images: {stats['total_images']}")
        print(f"Total QA Pairs: {stats['total_qa_pairs']}")
        
        # Preview a few samples sequentially
        count = 0
        for sample in reader.get_data(shuffle=False):
            print(f"Sample {count}: {sample}")
            count += 1
            if count >= 3:
                break
                
    else:
        print(f"Dataset root not found at {dataset_root}")
