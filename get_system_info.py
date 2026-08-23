#!/usr/bin/env python
"""
Collect machine configuration for paper reporting.

Usage:
    python get_system_info.py
"""

import platform
import sys
import json
from pathlib import Path
from datetime import datetime

def get_gpu_info():
    """Get GPU information."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"status": "No CUDA GPUs detected", "count": 0}

        gpu_info = {
            "status": "Available",
            "count": torch.cuda.device_count(),
            "devices": []
        }

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info["devices"].append({
                "id": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}"
            })

        gpu_info["driver_version"] = torch.version.cuda
        gpu_info["cudnn_version"] = torch.backends.cudnn.version()

        return gpu_info
    except Exception as e:
        return {"error": str(e)}

def get_cpu_info():
    """Get CPU information."""
    try:
        import psutil
        return {
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=False),
            "total_logical_cores": psutil.cpu_count(logical=True),
            "cpu_freq_ghz": psutil.cpu_freq().max / 1000,
            "total_ram_gb": psutil.virtual_memory().total / (1024**3)
        }
    except Exception as e:
        return {"error": str(e)}

def get_software_info():
    """Get software versions."""
    info = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
    }

    # PyTorch
    try:
        import torch
        info["pytorch_version"] = torch.__version__
    except:
        pass

    # TensorFlow (if available)
    try:
        import tensorflow as tf
        info["tensorflow_version"] = tf.__version__
    except:
        pass

    # NumPy
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except:
        pass

    # Scikit-learn
    try:
        import sklearn
        info["sklearn_version"] = sklearn.__version__
    except:
        pass

    return info

def main():
    print("\n" + "="*80)
    print("  MACHINE CONFIGURATION REPORT")
    print("="*80 + "\n")

    config = {
        "timestamp": datetime.now().isoformat(),
        "cpu": get_cpu_info(),
        "gpu": get_gpu_info(),
        "software": get_software_info()
    }

    # Display nicely
    print("CPU CONFIGURATION")
    print("─" * 80)
    for key, val in config["cpu"].items():
        if isinstance(val, float):
            print(f"  {key:<30}: {val:.2f}")
        else:
            print(f"  {key:<30}: {val}")

    print("\nGPU CONFIGURATION")
    print("─" * 80)
    gpu = config["gpu"]
    print(f"  Status: {gpu.get('status', 'Unknown')}")
    print(f"  Count: {gpu.get('count', 'N/A')}")
    if "devices" in gpu:
        for dev in gpu["devices"]:
            print(f"\n  Device {dev['id']}: {dev['name']}")
            print(f"    Total Memory: {dev['total_memory_gb']:.1f} GB")
            print(f"    Compute Capability: {dev['compute_capability']}")
    if "driver_version" in gpu:
        print(f"\n  CUDA Version: {gpu['driver_version']}")
    if "cudnn_version" in gpu:
        print(f"  cuDNN Version: {gpu['cudnn_version']}")

    print("\nSOFTWARE CONFIGURATION")
    print("─" * 80)
    for key, val in config["software"].items():
        print(f"  {key:<30}: {val}")

    # Save to JSON
    output_file = Path("machine_config.json")
    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ Configuration saved to {output_file}\n")

    # Generate paper text
    print("\nPAPER TEXT (Copy-Paste):")
    print("─" * 80)
    gpu_count = config["gpu"].get("count", 0)
    if gpu_count > 0:
        device_info = config["gpu"]["devices"][0]
        paper_text = f"""
Experiments were conducted on a machine with {config['cpu']['cpu_count']} physical CPU cores
(total {config['cpu']['total_logical_cores']} logical cores) and {config['cpu']['total_ram_gb']:.1f} GB of RAM.
For GPU acceleration, we used {gpu_count} × {device_info['name']} GPU(s) with {device_info['total_memory_gb']:.0f} GB memory each
(compute capability {device_info['compute_capability']}). We used CUDA {config['gpu']['driver_version']}
with cuDNN {config['gpu']['cudnn_version']}. All code was implemented in Python {config['software']['python_version']}
using PyTorch {config['software']['pytorch_version']}.
        """
    else:
        paper_text = f"""
Experiments were conducted on a machine with {config['cpu']['cpu_count']} physical CPU cores
(total {config['cpu']['total_logical_cores']} logical cores) and {config['cpu']['total_ram_gb']:.1f} GB of RAM.
All code was implemented in Python {config['software']['python_version']} using PyTorch {config['software']['pytorch_version']}.
        """
    print(paper_text.strip())
    print("\n" + "─" * 80 + "\n")

if __name__ == "__main__":
    main()
