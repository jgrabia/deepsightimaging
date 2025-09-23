#!/usr/bin/env python3
"""
Final Fix for Breast Model Implementation
Properly adds breast model registration to main.py
"""

import os
from pathlib import Path

def fix_breast_model_final():
    """Fix the breast model registration in main.py"""
    
    print("🔧 Final Fix for Breast Model Registration")
    print("=" * 50)
    
    # Path to MONAI Label app
    app_path = Path("~/.local/monailabel/sample-apps/radiology").expanduser()
    main_path = app_path / "main.py"
    
    if not main_path.exists():
        print("❌ main.py not found!")
        return
    
    # Read current main.py
    with open(main_path, 'r') as f:
        content = f.read()
    
    print("📄 Current main.py content:")
    print(content)
    print("\n" + "="*50)
    
    # Check if breast model registration is already there
    if "self.add_model(BreastSegmentationConfig())" in content:
        print("✅ Breast model registration already exists!")
        return
    
    # Find the right place to add the registration
    # Look for existing model registrations
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Look for existing model registrations and add ours after them
        if "self.add_model(" in line and "Segmentation()" in line:
            print(f"📍 Found model registration at line {i+1}: {line}")
            # Add our breast model registration after this line
            new_lines.append("        self.add_model(BreastSegmentationConfig())")
            print("✅ Added breast model registration")
    
    # Write back the updated content
    new_content = '\n'.join(new_lines)
    
    with open(main_path, 'w') as f:
        f.write(new_content)
    
    print("\n📄 Updated main.py content:")
    print(new_content)
    
    print("\n🎉 Breast model registration added successfully!")
    print("\n🔄 Now restart the MONAI server with:")
    print("   monailabel start_server --app ~/.local/monailabel/sample-apps/radiology \\")
    print("   --studies ~/mri_app/dicom_download \\")
    print("   --host 0.0.0.0 --port 8000 \\")
    print("   --conf models breast_segmentation,segmentation,deepgrow_2d,deepgrow_3d")

if __name__ == "__main__":
    fix_breast_model_final()





