#!/usr/bin/env python3
"""
Find Correct TaskConfig Import
Examines existing working configs to find the correct import path
"""

import os
from pathlib import Path

def find_correct_import():
    """Find the correct import path for TaskConfig"""
    
    print("🔍 Finding Correct TaskConfig Import")
    print("=" * 50)
    
    # Path to MONAI Label app
    app_path = Path("~/.local/monailabel/sample-apps/radiology").expanduser()
    configs_path = app_path / "lib" / "configs"
    
    # Look at existing working configs
    working_configs = ["segmentation.py", "deepgrow_2d.py", "deepgrow_3d.py"]
    
    for config_name in working_configs:
        config_path = configs_path / config_name
        if config_path.exists():
            print(f"📄 Examining {config_name}:")
            with open(config_path, 'r') as f:
                content = f.read()
                
                # Look for TaskConfig imports
                lines = content.split('\n')
                for i, line in enumerate(lines[:20]):  # Check first 20 lines
                    if "TaskConfig" in line:
                        print(f"   Line {i+1}: {line.strip()}")
                    if "from" in line and "import" in line:
                        print(f"   Line {i+1}: {line.strip()}")
                print()

if __name__ == "__main__":
    find_correct_import()





