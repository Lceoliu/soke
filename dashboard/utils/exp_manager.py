import os
import subprocess
from pathlib import Path
import yaml

def is_valid_experiment(path):
    exp_path = Path(path)
    ckpt_path = exp_path / "checkpoints"
    if not ckpt_path.exists():
        return False
    # Check if there is at least one .ckpt file
    ckpts = list(ckpt_path.glob("*.ckpt"))
    return len(ckpts) > 0

def find_exp_config(exp_path):
    # Try common names first
    for name in ["config.yaml", "runtime_config.yaml"]:
        p = exp_path / name
        if p.exists():
            return p
    # Fallback to any yaml that looks like a config
    yamls = list(exp_path.glob("*.yaml"))
    if yamls:
        # Prioritize files with 'config' in name
        for y in yamls:
            if 'config' in y.name.lower():
                return y
        return yamls[0]
    return None

def list_experiments(base_dir="experiments/mgpt"):
    results = []
    base_path = Path(base_dir)
    if not base_path.exists():
        return results
    
    # Recursively find directories with checkpoints
    for root, dirs, files in os.walk(base_dir):
        p = Path(root)
        if is_valid_experiment(p):
            config_path = find_exp_config(p)
            config = {}
            if config_path:
                with open(config_path, 'r') as f:
                    try:
                        config = yaml.safe_load(f)
                    except:
                        pass
            
            results.append({
                "name": p.name,
                "path": str(p),
                "config": config,
                "config_path": str(config_path) if config_path else None,
                "created_at": os.path.getctime(p)
            })
    
    results.sort(key=lambda x: x["created_at"], reverse=True)
    return results

def run_generate_report(exp_dict, python_bin="python3"):
    exp_path = Path(exp_dict["path"])
    config_path = exp_dict.get("config_path")
    if not config_path:
        return False, "No config file found for this experiment."
    
    # Find log file
    logs = list(exp_path.glob("log_*_train.log"))
    if not logs:
        return False, "No training log (*_train.log) found."
    log_path = sorted(logs, key=lambda x: x.stat().st_mtime)[-1]
    
    # Find checkpoint
    ckpt_dir = exp_path / "checkpoints"
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return False, "No checkpoint found."
    ckpt_path = sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1]
    
    out_dir = exp_path / "auto_reports" / "rvq_stage1"
    
    cmd = [
        python_bin,
        "scripts/analysis/generate_rvq_stage1_report.py",
        "--cfg", str(config_path),
        "--log_path", str(log_path),
        "--ckpt_path", str(ckpt_path),
        "--output_dir", str(out_dir),
        "--batch_size", "32",
        "--max_samples", "1000" # Reduced for speed in dashboard
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, str(out_dir)
    except subprocess.CalledProcessError as e:
        return False, f"Report generation failed: {e.stderr}"
