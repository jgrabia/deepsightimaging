#!/usr/bin/env python3
"""
Fix model name alias for real_breast_cancer
"""

import os
import shutil

# Create a symbolic link or copy the config file with the correct name
config_source = "~/.local/monailabel/sample-apps/radiology/lib/configs/advanced_breast_cancer_detection.py"
config_target = "~/.local/monailabel/sample-apps/radiology/lib/configs/real_breast_cancer.py"

# Copy the config file with the new name
shutil.copy2(os.path.expanduser(config_source), os.path.expanduser(config_target))

# Update the class name in the copied file
with open(os.path.expanduser(config_target), 'r') as f:
    content = f.read()

# Replace the class name
content = content.replace('AdvancedBreastCancerDetectionConfig', 'RealBreastCancerConfig')
content = content.replace('AdvancedBreastCancerDetection', 'AdvancedBreastCancerDetection')

with open(os.path.expanduser(config_target), 'w') as f:
    f.write(content)

print("✅ Created real_breast_cancer model alias")
print("✅ Updated class name to RealBreastCancerConfig")





