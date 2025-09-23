# MRI Breast Lesion Detection App - Complete Documentation

## 🎯 Project Overview
AI-powered breast lesion detection system using MONAI framework with DICOM image processing and Streamlit web interface.

## 🏗️ System Architecture
- **Backend**: Python with MONAI, PyTorch, Streamlit
- **Frontend**: Streamlit web interface
- **Data**: TCGA-BRCA MR images (15,399 files)
- **Model**: SegResNet for medical image segmentation
- **Infrastructure**: AWS EC2 with GPU support

## 📁 Key Files Structure
```
~/mri_app/
├── complete_dicom_app.py          # Main Streamlit application
├── advanced_breast_training.py    # Training script
├── consolidated_training_data/    # Training dataset (15,399 files)
│   ├── [173 subdirectories with DICOM files]
├── trained_models/               # Model checkpoints and reports
│   ├── best_breast_cancer_model.pth
│   ├── training_history.json
│   ├── training_report.json
│   └── training_plots.png
├── downloads/                    # TCIA download scripts
│   ├── download_tcia_normals.py
│   └── download_tcia_cancer.py
└── logs/                        # Training logs
```

## 🚀 Essential Commands

### 1. Start the DICOM Application
```bash
cd ~/mri_app
streamlit run complete_dicom_app.py
```

### 2. Run Classification Training (NEW - Recommended)
```bash
cd ~/mri_app
python breast_classification_trainer.py
```

### 3. Run Segmentation Training (Legacy)
```bash
cd ~/mri_app
python -c "from advanced_breast_training import AdvancedBreastCancerTrainer; AdvancedBreastCancerTrainer(data_dir='consolidated_training_data', config={'epochs': 40, 'learning_rate': 3e-5, 'batch_size': 2, 'patience': 25}).run_advanced_training()"
```

### 3. Test Data Loading
```bash
cd ~/mri_app
python -c "from advanced_breast_training import AdvancedBreastCancerTrainer; trainer = AdvancedBreastCancerTrainer(data_dir='consolidated_training_data'); training_data, validation_data, dataset_stats = trainer.prepare_advanced_training_data(); print(f'Training: {len(training_data)}, Validation: {len(validation_data)}')"
```

### 4. Monitor GPU Usage
```bash
nvidia-smi
watch -n 2 nvidia-smi
```

### 5. Check Training Progress
```bash
# Check if training is running
ps aux | grep python | grep advanced_breast_training

# View training logs
tail -f ~/mri_app/logs/training_*.log
```

## 📊 Current Dataset Status
- **Total Files**: 15,399 DICOM files
- **Training Samples**: 12,319
- **Validation Samples**: 3,080
- **Data Source**: TCGA-BRCA MR images
- **Image Size**: 512x512 pixels
- **Modality**: MR (Magnetic Resonance)
- **Quality**: Research-grade

## 🔧 Key Configuration Parameters

### Classification Model (NEW - Recommended)
```python
config = {
    'num_classes': 3,  # Normal, Benign, Malignant
    'batch_size': 16,
    'learning_rate': 1e-4,
    'epochs': 50,
    'patience': 10,
    'image_size': 224,
    'pretrained': True,
    'class_names': ['Normal', 'Benign', 'Malignant']
}
```

### Segmentation Model (Legacy)
```python
config = {
    'epochs': 40,
    'learning_rate': 3e-5,
    'batch_size': 2,
    'patience': 25,
    'model_type': 'SegResNet',
    'spatial_dims': 2,
    'in_channels': 1,
    'out_channels': 3,
    'image_size': [512, 512]
}
```

## 🎯 Training Results History
1. **First Attempt**: 0 valid files (validation bug)
2. **Second Attempt**: 5,815 files, dice 0.4473
3. **Current**: 15,399 files, expected much better results

## 🚨 Known Issues & Solutions

### Issue 1: Validation Bug (FIXED)
**Problem**: `'AdvancedBreastCancerTrainer' object has no attribute 'image_type_stats'`
**Solution**: The bug was in the validation method - now working correctly

### Issue 2: Memory Issues
**Problem**: CUDA out of memory
**Solution**: Use batch_size=2, cache_rate=0.2, num_workers=2

### Issue 3: False Positives
**Problem**: Model detecting too many lesions
**Solution**: Training on high-quality TCGA-BRCA data should improve this

## 📋 Setup Instructions for New Environment

### 1. Install Dependencies
```bash
pip install streamlit monai torch torchvision pydicom tcia-utils
```

### 2. Transfer Files
- Copy entire `~/mri_app/` directory
- Ensure `consolidated_training_data/` has all 15,399 files
- Verify `trained_models/` directory exists

### 3. Verify Setup
```bash
cd ~/mri_app
python -c "import streamlit, monai, torch; print('Dependencies OK')"
```

### 4. Test Data Loading
```bash
cd ~/mri_app
python -c "from advanced_breast_training import AdvancedBreastCancerTrainer; trainer = AdvancedBreastCancerTrainer(data_dir='consolidated_training_data'); print('Setup OK')"
```

## 🎯 Application Features
- **DICOM Upload**: Drag & drop DICOM files
- **AI Inference**: Real-time lesion detection
- **Clinical Analysis**: Detailed metrics and statistics
- **Image Overlay**: Visual lesion highlighting
- **Download**: Save processed images

## 📈 Performance Metrics
- **Dice Score**: Target >0.7 (currently improving)
- **Hausdorff Distance**: Target <10mm
- **Precision/Recall**: Target >0.8
- **Inference Time**: <5 seconds per image

## 🔄 Workflow
1. **Upload DICOM**: User uploads breast MR image
2. **Preprocessing**: MONAI transforms applied
3. **AI Inference**: SegResNet model processes image
4. **Post-processing**: Lesion detection and analysis
5. **Visualization**: Overlay results on original image
6. **Clinical Report**: Generate detailed analysis

## 🚀 Next Steps
1. **Monitor Current Training**: Let 40-epoch training complete
2. **Test New Model**: Run inference on test images
3. **Evaluate Performance**: Check for reduced false positives
4. **Deploy Updates**: Update DICOM app with new model
5. **Clinical Validation**: Test with real clinical cases

## 📞 Support Information
- **Model Type**: SegResNet (MONAI)
- **Framework**: PyTorch + MONAI
- **Data Source**: TCGA-BRCA via TCIA
- **Infrastructure**: AWS EC2 with Tesla T4 GPU
- **Last Updated**: September 13, 2025

## 🔐 Important Notes
- **Medical Device**: This is for research/development only
- **Data Privacy**: TCGA-BRCA data is de-identified
- **Model Validation**: Requires clinical validation before deployment
- **Backup**: Always backup trained models and data

---
*This documentation covers the complete setup and operation of the MRI Breast Lesion Detection system. All commands have been tested and verified to work with the current configuration.*
anto 