# 🏥 Automated Training Data Collection Guide

## 🎯 **Overview**

This guide helps you automatically collect high-quality breast cancer training data with expert annotations from TCIA (The Cancer Imaging Archive).

## 🚀 **Quick Start**

### **Option 1: Command Line (Fast)**
```bash
# Download a specific dataset
python automated_data_collector.py --dataset CBIS-DDSM --max-series 50

# Download all datasets (recommended)
python automated_data_collector.py --full-collection --max-series 50
```

### **Option 2: Streamlit Interface (User-Friendly)**
```bash
# Launch the data collection app
streamlit run data_collection_app.py
```

## 📊 **Available Datasets**

### **1. CBIS-DDSM (Recommended First)**
- **Description**: Curated Breast Imaging Subset of DDSM
- **Images**: ~2,000 mammograms
- **Label Quality**: **Very High** - Expert radiologist annotations
- **Modality**: Mammography (MG)
- **Why Choose**: Best annotations, most reliable for training

### **2. INbreast**
- **Description**: INbreast Database for Mammographic Image Analysis
- **Images**: ~500 mammograms
- **Label Quality**: **Very High** - Expert annotations with BI-RADS
- **Modality**: Mammography (MG)
- **Why Choose**: High-quality annotations with clinical classifications

### **3. TCGA-BRCA**
- **Description**: The Cancer Genome Atlas Breast Cancer Collection
- **Images**: ~1,000 MRI scans
- **Label Quality**: **High** - Clinical annotations available
- **Modality**: MRI (MR)
- **Why Choose**: Large dataset with clinical metadata

### **4. BCDR**
- **Description**: Breast Cancer Digital Repository
- **Images**: ~800 images
- **Label Quality**: **High** - Clinical and pathological data
- **Modality**: Mixed
- **Why Choose**: Good clinical correlation data

## 🎯 **Recommended Collection Strategy**

### **Phase 1: Start Small (Validation)**
```bash
# Download 50 images from CBIS-DDSM to test
python automated_data_collector.py --dataset CBIS-DDSM --max-series 50
```

### **Phase 2: Expand (Training)**
```bash
# Download 200 images from each dataset
python automated_data_collector.py --dataset CBIS-DDSM --max-series 200
python automated_data_collector.py --dataset INbreast --max-series 200
```

### **Phase 3: Full Collection (Production)**
```bash
# Download all available data
python automated_data_collector.py --full-collection --max-series 500
```

## 📋 **Data Quality Assessment**

### **What You'll Get:**
- **DICOM Images**: High-quality medical images
- **Expert Annotations**: Radiologist-marked lesions
- **Clinical Metadata**: Patient demographics, diagnoses
- **BI-RADS Classifications**: Standardized breast imaging reporting
- **Pathological Data**: Confirmed diagnoses where available

### **Label Quality Comparison:**
| Dataset | Annotation Quality | Clinical Data | Size | Recommendation |
|---------|-------------------|---------------|------|----------------|
| CBIS-DDSM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Large | **Best for training** |
| INbreast | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Medium | **Best for validation** |
| TCGA-BRCA | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Large | **Good for clinical correlation** |
| BCDR | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | **Good for diversity** |

## 🔧 **Technical Details**

### **Storage Requirements:**
- **50 images**: ~2-5 GB
- **200 images**: ~8-20 GB
- **500 images**: ~20-50 GB
- **Full collection**: ~100-200 GB

### **Download Speed:**
- **Typical**: 10-50 MB/s
- **Estimated time for 200 images**: 30-60 minutes
- **Rate limiting**: 1 second between downloads

### **File Structure:**
```
training_data/
├── CBIS-DDSM/
│   ├── metadata.json
│   ├── series_uid_1.zip
│   ├── series_uid_2.zip
│   └── ...
├── INbreast/
│   ├── metadata.json
│   └── ...
├── training_manifest.json
└── download_log.txt
```

## 🎯 **Best Practices**

### **1. Start with CBIS-DDSM**
- Highest quality annotations
- Most reliable for initial training
- Good balance of normal and abnormal cases

### **2. Validate with INbreast**
- Use for model validation
- High-quality expert annotations
- Smaller, more manageable size

### **3. Expand with TCGA-BRCA**
- Large dataset for robust training
- Clinical correlation data
- Different imaging modality (MRI)

### **4. Monitor Quality**
- Check download logs for errors
- Verify metadata completeness
- Validate image quality

## 🚨 **Troubleshooting**

### **Common Issues:**

**1. Download Failures**
```bash
# Check network connection
ping services.cancerimagingarchive.net

# Verify TCIA API access
curl "https://services.cancerimagingarchive.net/nbia-api/services/v1/getCollectionValues"
```

**2. Storage Space**
```bash
# Check available space
df -h

# Clean up if needed
rm -rf training_data/partial_downloads
```

**3. Rate Limiting**
- TCIA has rate limits
- Script includes 1-second delays
- If blocked, wait 15-30 minutes

### **Error Recovery:**
```bash
# Resume interrupted downloads
python automated_data_collector.py --dataset CBIS-DDSM --max-series 50

# The script will skip already downloaded files
```

## 📈 **Next Steps After Collection**

### **1. Validate Data Quality**
```python
# Check downloaded data
import os
training_dir = "training_data"
for dataset in os.listdir(training_dir):
    if os.path.isdir(os.path.join(training_dir, dataset)):
        zip_files = [f for f in os.listdir(os.path.join(training_dir, dataset)) if f.endswith('.zip')]
        print(f"{dataset}: {len(zip_files)} images")
```

### **2. Update Training Script**
```python
# Modify advanced_breast_training.py to use new data
data_dir = "training_data"  # Point to new directory
```

### **3. Retrain Model**
```bash
# Retrain with new data
python advanced_breast_training.py
```

## 🎉 **Expected Results**

With proper training data, you should see:
- **More accurate predictions** - Real expert annotations
- **Better generalization** - Diverse patient population
- **Reduced false positives** - Higher quality labels
- **Clinical relevance** - Real-world performance

## 📞 **Support**

If you encounter issues:
1. Check the download logs in `training_data/download_log.txt`
2. Verify TCIA API access
3. Ensure sufficient storage space
4. Check network connectivity

---

**Happy Data Collection! 🚀**





