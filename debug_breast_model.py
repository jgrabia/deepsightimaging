#!/usr/bin/env python3
"""
Debug Breast Model Implementation
Investigates and fixes the breast segmentation model
"""

import os
import sys
from pathlib import Path

def debug_breast_model():
    """Debug and fix the breast model implementation"""
    
    print("🔍 Debugging Breast Model Implementation")
    print("=" * 50)
    
    # Path to MONAI Label app
    app_path = Path("~/.local/monailabel/sample-apps/radiology").expanduser()
    
    # 1. Check existing working configs
    print("1. Analyzing existing working configs...")
    configs_path = app_path / "lib" / "configs"
    
    if configs_path.exists():
        for config_file in configs_path.glob("*.py"):
            if config_file.name != "__init__.py":
                print(f"   📁 Found config: {config_file.name}")
                
                # Read the first few lines to see the pattern
                with open(config_file, 'r') as f:
                    lines = f.readlines()[:10]
                    print(f"   📄 First few lines of {config_file.name}:")
                    for line in lines:
                        print(f"      {line.rstrip()}")
                    print()
    
    # 2. Check our breast config
    print("2. Checking our breast config...")
    breast_config_path = configs_path / "breast_segmentation.py"
    
    if breast_config_path.exists():
        with open(breast_config_path, 'r') as f:
            content = f.read()
            print(f"   📄 Our breast config content:")
            print(f"   {content}")
    else:
        print("   ❌ Breast config file not found!")
    
    # 3. Check main.py
    print("3. Checking main.py...")
    main_path = app_path / "main.py"
    
    if main_path.exists():
        with open(main_path, 'r') as f:
            content = f.read()
            
            # Check for breast import
            if "from lib.configs.breast_segmentation import BreastSegmentationConfig" in content:
                print("   ✅ Breast import found in main.py")
            else:
                print("   ❌ Breast import NOT found in main.py")
            
            # Check for breast model registration
            if "self.add_model(BreastSegmentationConfig())" in content:
                print("   ✅ Breast model registration found in main.py")
            else:
                print("   ❌ Breast model registration NOT found in main.py")
    
    # 4. Test import manually
    print("4. Testing manual import...")
    try:
        sys.path.insert(0, str(app_path))
        from lib.configs.breast_segmentation import BreastSegmentationConfig
        print("   ✅ Manual import successful")
        
        # Test instantiation
        config = BreastSegmentationConfig()
        print(f"   ✅ Config instantiation successful: {config.name}")
        
    except Exception as e:
        print(f"   ❌ Manual import failed: {e}")
    
    # 5. Check MONAI Label's class discovery
    print("5. Checking MONAI Label class discovery...")
    try:
        from monailabel.utils.others.class_utils import get_subclasses
        from monailabel.interfaces.tasks.config import TaskConfig
        
        # Get all TaskConfig subclasses
        task_configs = get_subclasses(TaskConfig, "lib.configs")
        print(f"   📋 Found TaskConfig subclasses: {[c.__name__ for c in task_configs]}")
        
        # Check if our breast config is in the list
        breast_found = any("breast" in c.__name__.lower() for c in task_configs)
        if breast_found:
            print("   ✅ Breast config found in MONAI Label discovery")
        else:
            print("   ❌ Breast config NOT found in MONAI Label discovery")
            
    except Exception as e:
        print(f"   ❌ Class discovery test failed: {e}")

if __name__ == "__main__":
    debug_breast_model()





