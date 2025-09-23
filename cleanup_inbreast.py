#!/usr/bin/env python3
"""
Clean up INbreast training data directory
"""

import os
import shutil
import sys

def cleanup_directory(directory_path):
    """Remove all files and subdirectories from the specified directory"""
    
    if not os.path.exists(directory_path):
        print(f"❌ Directory does not exist: {directory_path}")
        return False
    
    try:
        # Count files before deletion
        file_count = 0
        dir_count = 0
        
        for root, dirs, files in os.walk(directory_path):
            file_count += len(files)
            dir_count += len(dirs)
        
        print(f"📁 Found {file_count} files and {dir_count} directories in {directory_path}")
        
        # Remove all contents
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"❌ Failed to delete {file_path}: {e}")
        
        print(f"✅ Successfully cleaned directory: {directory_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error cleaning directory {directory_path}: {e}")
        return False

def main():
    directory_path = "/home/ubuntu/extracted_training_data/INbreast"
    
    print(f"🧹 Cleaning directory: {directory_path}")
    
    # Confirm deletion
    response = input(f"Are you sure you want to delete all contents of {directory_path}? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Operation cancelled")
        return
    
    success = cleanup_directory(directory_path)
    
    if success:
        print("🎉 Cleanup completed successfully!")
    else:
        print("❌ Cleanup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()


