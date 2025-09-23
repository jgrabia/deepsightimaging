#!/usr/bin/env python3
"""
Test script to verify the Streamlit integration works
"""

import sys
import os

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        from balanced_breast_inferer import BalancedBreastInferer
        print("✅ BalancedBreastInferer imported successfully")
    except ImportError as e:
        print(f"❌ BalancedBreastInferer import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if the model can be loaded"""
    print("\nTesting model loading...")
    
    try:
        from balanced_breast_inferer import BalancedBreastInferer
        inferer = BalancedBreastInferer()
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_streamlit_app():
    """Test if the Streamlit app can be imported"""
    print("\nTesting Streamlit app import...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Try to import the main function
        from complete_dicom_app import main
        print("✅ Streamlit app imported successfully")
        return True
    except Exception as e:
        print(f"❌ Streamlit app import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Streamlit Integration")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_loading,
        test_streamlit_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Streamlit integration is ready.")
        print("\nTo run the Streamlit app:")
        print("streamlit run complete_dicom_app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()


