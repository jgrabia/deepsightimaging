#!/usr/bin/env python3
"""
Script to fix MONAI Label writer configuration issue
"""
import os
import sys

def main():
    print("🔧 MONAI Label Writer Configuration Issue")
    print("=" * 50)
    
    print("\n🎯 **Current Issue:**")
    print("✅ API request is working correctly")
    print("✅ 'output': 'json' parameter is being received")
    print("❌ Writer is still trying to write .dcm files")
    print("❌ GDCM error still occurs")
    
    print("\n📊 **Analysis:**")
    print("The logs show:")
    print("- Infer Request: {'output': 'json'} ✅")
    print("- Result ext: .dcm; write_to_file: True ❌")
    print("- The output parameter is not being handled by the writer")
    
    print("\n🔍 **Step 1: Check the Writer Configuration**")
    print("The issue is in the MONAI Label writer configuration:")
    print("\n# Check the writer configuration:")
    print("python3 -c \"import monailabel.transform.writer; print(monailabel.transform.writer.__file__)\"")
    
    print("\n# Look at the writer configuration:")
    print("cat /usr/local/lib/python3.10/dist-packages/monailabel/transform/writer.py")
    
    print("\n🔍 **Step 2: Check How Output Parameter Should Be Handled**")
    print("The writer needs to check for the 'output' parameter and handle it:")
    print("\n# Look for output parameter handling:")
    print("grep -n 'output' /usr/local/lib/python3.10/dist-packages/monailabel/transform/writer.py")
    
    print("\n🔍 **Step 3: Check the Radiology App Writer Configuration**")
    print("The radiology app might have its own writer configuration:")
    print("\ncd ~/.local/monailabel/sample-apps/radiology")
    print("grep -r 'writer' lib/")
    print("grep -r 'Result ext' lib/")
    
    print("\n🚀 **Step 4: Try Different Output Formats**")
    print("Let's try different output parameter values:")
    print("\n# Try 'nifti' instead of 'json':")
    print("params = {'output': 'nifti'}")
    
    print("\n# Try 'image' instead of 'json':")
    print("params = {'output': 'image'}")
    
    print("\n# Try 'array' instead of 'json':")
    print("params = {'output': 'array'}")
    
    print("\n🔍 **Step 5: Check MONAI Label Documentation**")
    print("Let's check what output formats are supported:")
    print("\n# Check the MONAI Label GitHub repository:")
    print("https://github.com/Project-MONAI/MONAILabel")
    
    print("\n# Check the writer documentation:")
    print("https://github.com/Project-MONAI/MONAILabel/tree/main/monailabel/transform")
    
    print("\n💡 **Expected Result:**")
    print("- Writer should respect the 'output' parameter")
    print("- No more GDCM errors")
    print("- JSON results returned to client")
    
    print("\n⚠️  **If Still Having Issues:**")
    print("1. Check MONAI Label GitHub issues for similar problems")
    print("2. Look for writer configuration examples")
    print("3. Consider modifying the writer configuration directly")

if __name__ == "__main__":
    main()
