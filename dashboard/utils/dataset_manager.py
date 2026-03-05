import os
import gzip
import pickle
from pathlib import Path
from typing import List, Dict, Optional

class DatasetManager:
    def __init__(self, data_root="data"):
        self.data_root = Path(data_root)

    def list_datasets(self) -> List[Dict]:
        datasets = []
        if not self.data_root.exists():
            return datasets

        for item in self.data_root.iterdir():
            if not item.is_dir():
                continue
            
            # Check for poses directory (CSL style)
            poses_path = item / "poses"
            
            # Compatibility check for How2Sign and Phoenix
            # How2Sign usually has split/poses
            # Phoenix might have poses directly or be at root
            is_dataset = False
            if poses_path.exists():
                is_dataset = True
            elif (item / "train" / "poses").exists():
                is_dataset = True
            elif (item / "phoenix14t.train").exists(): # Phoenix indicator
                is_dataset = True
            
            if is_dataset:
                datasets.append({
                    "id": item.name,
                    "name": item.name,
                    "path": str(item)
                })
        return datasets

    def get_statistics(self, dataset_path: str) -> Dict:
        p = Path(dataset_path)
        stats = {"total_poses": 0, "splits": {}}
        
        # Look for annotation files
        # CSL / SOKE style
        for split in ["train", "val", "test"]:
            ann_file = p / f"csl_clean.{split}"
            if ann_file.exists():
                try:
                    with gzip.open(ann_file, "rb") as f:
                        data = pickle.load(f)
                        stats["splits"][split] = len(data)
                        stats["total_poses"] += len(data)
                except:
                    pass
        
        # Phoenix style
        if not stats["total_poses"]:
            split_map = {"train": "train", "val": "dev", "test": "test"}
            for s_name, f_suffix in split_map.items():
                ann_file = p / f"phoenix14t.{f_suffix}"
                if ann_file.exists():
                    try:
                        with gzip.open(ann_file, "rb") as f:
                            data = pickle.load(f)
                            stats["splits"][s_name] = len(data)
                            stats["total_poses"] += len(data)
                    except:
                        pass

        # If no annotations found, just count directories in poses/
        if not stats["total_poses"]:
            poses_dir = p / "poses"
            if poses_dir.exists():
                count = len([d for d in poses_dir.iterdir() if d.is_dir()])
                stats["total_poses"] = count
                stats["splits"]["unknown"] = count

        return stats

    def get_pose_list(self, dataset_path: str, split: str = "all") -> List[str]:
        p = Path(dataset_path)
        all_poses = {} # Use dict to deduplicate
        
        splits_to_check = ["train", "val", "test"] if split == "all" else [split]
        
        # Try to get names from CSL/Phoenix annotations
        for s in splits_to_check:
            # CSL style
            ann_file = p / f"csl_clean.{s}"
            if ann_file.exists():
                try:
                    with gzip.open(ann_file, "rb") as f:
                        data = pickle.load(f)
                        for x in data: all_poses[x["name"]] = True
                except: pass
            
            # Phoenix style
            split_map = {"train": "train", "val": "dev", "test": "test"}
            ann_file_ph = p / f"phoenix14t.{split_map.get(s, s)}"
            if ann_file_ph.exists():
                try:
                    with gzip.open(ann_file_ph, "rb") as f:
                        data = pickle.load(f)
                        for x in data: all_poses[x["name"]] = True
                except: pass
        
        if all_poses:
            return sorted(list(all_poses.keys()))
        
        # Fallback to listing directories in poses/
        poses_dir = p / "poses"
        if poses_dir.exists():
            return sorted([d.name for d in poses_dir.iterdir() if d.is_dir()])
        
        return []

    def resolve_pose_path(self, dataset_path: str, pose_name: str) -> Optional[str]:
        base = Path(dataset_path)
        # Standard CSL/SOKE
        p = base / "poses" / pose_name
        if p.exists(): return str(p)
        
        # How2Sign/Split style
        for split in ["train", "val", "test"]:
            p = base / split / "poses" / pose_name
            if p.exists(): return str(p)
            
        # Phoenix style (often flat or in subdirs)
        p = base / pose_name
        if p.exists() and p.is_dir(): return str(p)
        
        return None
