#!/usr/bin/env python3
"""
Script to find breast and lung cancer datasets from TCIA
"""

from tcia_utils import nbia
import json

def search_cancer_datasets():
    """Search for breast and lung cancer datasets in TCIA"""
    print("🔍 Searching for Breast and Lung Cancer Datasets in TCIA")
    print("=" * 60)
    
    # Search for breast cancer datasets
    print("\n🏥 **Breast Cancer Datasets:**")
    print("-" * 40)
    
    breast_collections = [
        "TCGA-BRCA",  # Breast Cancer
        "TCGA-BRCA-Pathology",  # Breast Cancer Pathology
        "TCGA-BRCA-Radiomics",  # Breast Cancer Radiomics
        "TCGA-BRCA-Clinical",  # Breast Cancer Clinical
        "TCGA-BRCA-Genomics",  # Breast Cancer Genomics
    ]
    
    for collection in breast_collections:
        try:
            print(f"\n📊 Collection: {collection}")
            studies = nbia.getPatientStudy(collection=collection, maxR=5)
            if studies:
                print(f"   ✅ Found {len(studies)} studies")
                for study in studies[:3]:  # Show first 3
                    print(f"      - Study: {study.get('StudyDescription', 'N/A')}")
                    print(f"        Patient: {study.get('PatientID', 'N/A')}")
            else:
                print(f"   ❌ No studies found")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Search for lung cancer datasets
    print("\n🫁 **Lung Cancer Datasets:**")
    print("-" * 40)
    
    lung_collections = [
        "TCGA-LUAD",  # Lung Adenocarcinoma
        "TCGA-LUSC",  # Lung Squamous Cell Carcinoma
        "TCGA-LUAD-Radiomics",  # Lung Adenocarcinoma Radiomics
        "TCGA-LUSC-Radiomics",  # Lung Squamous Cell Carcinoma Radiomics
        "NSCLC-Radiomics",  # Non-Small Cell Lung Cancer
        "LIDC-IDRI",  # Lung Image Database Consortium
    ]
    
    for collection in lung_collections:
        try:
            print(f"\n📊 Collection: {collection}")
            studies = nbia.getPatientStudy(collection=collection, maxR=5)
            if studies:
                print(f"   ✅ Found {len(studies)} studies")
                for study in studies[:3]:  # Show first 3
                    print(f"      - Study: {study.get('StudyDescription', 'N/A')}")
                    print(f"        Patient: {study.get('PatientID', 'N/A')}")
            else:
                print(f"   ❌ No studies found")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Search for general cancer datasets
    print("\n🎯 **General Cancer Datasets:**")
    print("-" * 40)
    
    general_collections = [
        "TCGA-OV",  # Ovarian Cancer
        "TCGA-GBM",  # Glioblastoma
        "TCGA-KIRC",  # Kidney Cancer
        "TCGA-HNSC",  # Head and Neck Cancer
    ]
    
    for collection in general_collections:
        try:
            print(f"\n📊 Collection: {collection}")
            studies = nbia.getPatientStudy(collection=collection, maxR=3)
            if studies:
                print(f"   ✅ Found {len(studies)} studies")
            else:
                print(f"   ❌ No studies found")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n📝 **Recommended Collections for Testing:**")
    print("1. LIDC-IDRI - Lung nodule dataset (most comprehensive)")
    print("2. TCGA-BRCA - Breast cancer dataset")
    print("3. TCGA-LUAD - Lung adenocarcinoma dataset")
    print("4. TCGA-LUSC - Lung squamous cell carcinoma dataset")
    
    print("\n🔧 **Next Steps:**")
    print("1. Use these collections in your Streamlit app")
    print("2. Download specific studies for testing")
    print("3. Test the tumor detection models on these datasets")

def get_series_info(collection, max_results=10):
    """Get detailed series information for a collection"""
    print(f"\n📋 Detailed Series Info for {collection}:")
    print("-" * 50)
    
    try:
        series = nbia.getSeries(collection=collection, maxR=max_results)
        
        if not series:
            print("❌ No series found")
            return
        
        for i, s in enumerate(series, 1):
            print(f"\n{i}. Series: {s.get('SeriesDescription', 'N/A')}")
            print(f"   Patient ID: {s.get('PatientID', 'N/A')}")
            print(f"   Modality: {s.get('Modality', 'N/A')}")
            print(f"   Body Part: {s.get('BodyPartExamined', 'N/A')}")
            print(f"   Image Count: {s.get('ImageCount', 'N/A')}")
            print(f"   Series UID: {s.get('SeriesInstanceUID', 'N/A')[:30]}...")
            
    except Exception as e:
        print(f"❌ Error getting series info: {e}")

if __name__ == "__main__":
    search_cancer_datasets()
    
    # Get detailed info for key collections
    print("\n" + "="*60)
    print("🔍 DETAILED ANALYSIS")
    print("="*60)
    
    key_collections = ["LIDC-IDRI", "TCGA-BRCA", "TCGA-LUAD"]
    
    for collection in key_collections:
        get_series_info(collection, max_results=5)





