import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Grid,
  Paper,
  Chip,
  Alert,
  CircularProgress,
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import toast from 'react-hot-toast';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_AWS_API_URL || 'http://localhost:8000';

interface DICOMInfo {
  patient_id: string;
  patient_name: string;
  study_date: string;
  study_description: string;
  modality: string;
  manufacturer: string;
  model: string;
  num_slices: number;
  image_size: number[];
}

interface Annotation {
  slice: number;
  x: number;
  y: number;
  width: number;
  height: number;
  class: string;
  view: string;
}

interface WindowLevelPreset {
  center: number | null;
  width: number | null;
}

const DICOMViewerAPI: React.FC = () => {
  // State management
  const [dicomInfo, setDicomInfo] = useState<DICOMInfo | null>(null);
  const [currentSlice, setCurrentSlice] = useState(0);
  const [totalSlices, setTotalSlices] = useState(0);
  const [currentImage, setCurrentImage] = useState<string>('');
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [loading, setLoading] = useState(false);
  
  // Viewer controls
  const [windowCenter, setWindowCenter] = useState(40);
  const [windowWidth, setWindowWidth] = useState(400);
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [contrastFactor, setContrastFactor] = useState(1.0);
  const [noiseReduction, setNoiseReduction] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState('soft_tissue');
  
  // Presets
  const [presets, setPresets] = useState<Record<string, WindowLevelPreset>>({});
  
  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load window/level presets on component mount
  useEffect(() => {
    loadPresets();
  }, []);

  const loadPresets = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/dicom/window-level/presets`);
      const data = await response.json();
      setPresets(data.presets);
    } catch (error) {
      console.error('Error loading presets:', error);
    }
  };

  const uploadDICOM = async (file: File) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE_URL}/api/dicom/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      
      setDicomInfo({
        patient_id: data.patient_id,
        patient_name: 'Unknown',
        study_date: data.study_date,
        study_description: 'Unknown',
        modality: 'Unknown',
        manufacturer: 'Unknown',
        model: 'Unknown',
        num_slices: data.num_slices,
        image_size: [0, 0],
      });
      
      setCurrentSlice(data.current_slice);
      setTotalSlices(data.num_slices);
      setCurrentImage(data.image);
      setAnnotations([]);
      
      // Load annotations
      await loadAnnotations();
      
      toast.success('DICOM file uploaded successfully!');
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Failed to upload DICOM file');
    } finally {
      setLoading(false);
    }
  };

  const loadAnnotations = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/dicom/annotations`);
      const data = await response.json();
      setAnnotations(data.annotations);
    } catch (error) {
      console.error('Error loading annotations:', error);
    }
  };

  const loadSlice = async (sliceNumber: number) => {
    try {
      const params = new URLSearchParams({
        window_center: windowCenter.toString(),
        window_width: windowWidth.toString(),
        show_annotations: showAnnotations.toString(),
        contrast_factor: contrastFactor.toString(),
        noise_reduction: noiseReduction.toString(),
      });

      const response = await fetch(`${API_BASE_URL}/api/dicom/slice/${sliceNumber}?${params}`);
      
      if (!response.ok) {
        throw new Error('Failed to load slice');
      }

      const data = await response.json();
      setCurrentImage(data.image);
      setCurrentSlice(sliceNumber);
    } catch (error) {
      console.error('Error loading slice:', error);
      toast.error('Failed to load slice');
    }
  };

  const handleSliceChange = (event: Event, newValue: number | number[]) => {
    const sliceNum = newValue as number;
    loadSlice(sliceNum);
  };

  const handlePresetChange = (preset: string) => {
    setSelectedPreset(preset);
    const presetData = presets[preset];
    
    if (presetData.center !== null && presetData.width !== null) {
      setWindowCenter(presetData.center);
      setWindowWidth(presetData.width);
      loadSlice(currentSlice); // Reload current slice with new window/level
    }
  };

  const handleWindowLevelChange = () => {
    loadSlice(currentSlice); // Reload current slice with new window/level
  };

  const handleEnhancementChange = () => {
    loadSlice(currentSlice); // Reload current slice with new enhancements
  };

  // File dropzone
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        uploadDICOM(acceptedFiles[0]);
      }
    },
    accept: {
      'application/dicom': ['.dcm', '.dicom']
    },
    multiple: false
  });

  const currentSliceAnnotations = annotations.filter(ann => ann.slice === currentSlice);

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        DICOM Viewer (API Connected)
      </Typography>

      {/* File Upload */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Upload DICOM File
          </Typography>
          
          <Paper
            {...getRootProps()}
            sx={{
              p: 3,
              textAlign: 'center',
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'grey.300',
              cursor: 'pointer',
              '&:hover': {
                borderColor: 'primary.main',
                backgroundColor: 'action.hover',
              }
            }}
          >
            <input {...getInputProps()} />
            {isDragActive ? (
              <Typography>Drop the DICOM file here...</Typography>
            ) : (
              <Typography>
                Drag & drop a DICOM file here, or click to select
              </Typography>
            )}
          </Paper>
        </CardContent>
      </Card>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
          <CircularProgress />
        </Box>
      )}

      {dicomInfo && (
        <>
          {/* DICOM Information */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                DICOM Information
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2">
                    <strong>Patient ID:</strong> {dicomInfo.patient_id}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Study Date:</strong> {dicomInfo.study_date}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Modality:</strong> {dicomInfo.modality}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">
                    <strong>Manufacturer:</strong> {dicomInfo.manufacturer}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Model:</strong> {dicomInfo.model}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Total Slices:</strong> {dicomInfo.num_slices}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Image Display */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Image Viewer
              </Typography>
              
              {currentImage && (
                <Box sx={{ textAlign: 'center' }}>
                  <img
                    src={currentImage}
                    alt={`Slice ${currentSlice + 1}`}
                    style={{
                      maxWidth: '100%',
                      maxHeight: '600px',
                      border: '1px solid #ccc',
                      borderRadius: '4px'
                    }}
                  />
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Slice {currentSlice + 1} of {totalSlices}
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>

          {/* Controls */}
          <Grid container spacing={3}>
            {/* Slice Navigation */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Slice Navigation
                  </Typography>
                  
                  <Typography gutterBottom>
                    Slice: {currentSlice + 1} / {totalSlices}
                  </Typography>
                  
                  <Slider
                    value={currentSlice}
                    onChange={handleSliceChange}
                    min={0}
                    max={totalSlices - 1}
                    step={1}
                    marks
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => value + 1}
                  />
                  
                  <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                    <Button
                      variant="outlined"
                      onClick={() => loadSlice(Math.max(0, currentSlice - 1))}
                      disabled={currentSlice === 0}
                    >
                      Previous
                    </Button>
                    <Button
                      variant="outlined"
                      onClick={() => loadSlice(Math.min(totalSlices - 1, currentSlice + 1))}
                      disabled={currentSlice === totalSlices - 1}
                    >
                      Next
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Window/Level Controls */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Window/Level Controls
                  </Typography>
                  
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Preset</InputLabel>
                    <Select
                      value={selectedPreset}
                      onChange={(e) => handlePresetChange(e.target.value)}
                    >
                      {Object.keys(presets).map((preset) => (
                        <MenuItem key={preset} value={preset}>
                          {preset.replace('_', ' ').toUpperCase()}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  <Typography gutterBottom>
                    Window Center: {windowCenter}
                  </Typography>
                  <Slider
                    value={windowCenter}
                    onChange={(e, value) => {
                      setWindowCenter(value as number);
                      handleWindowLevelChange();
                    }}
                    min={-1000}
                    max={1000}
                    step={10}
                    valueLabelDisplay="auto"
                  />
                  
                  <Typography gutterBottom sx={{ mt: 2 }}>
                    Window Width: {windowWidth}
                  </Typography>
                  <Slider
                    value={windowWidth}
                    onChange={(e, value) => {
                      setWindowWidth(value as number);
                      handleWindowLevelChange();
                    }}
                    min={100}
                    max={2000}
                    step={50}
                    valueLabelDisplay="auto"
                  />
                </CardContent>
              </Card>
            </Grid>

            {/* Image Enhancement */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Image Enhancement
                  </Typography>
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={showAnnotations}
                        onChange={(e) => {
                          setShowAnnotations(e.target.checked);
                          handleEnhancementChange();
                        }}
                      />
                    }
                    label="Show Annotations"
                  />
                  
                  <Typography gutterBottom sx={{ mt: 2 }}>
                    Contrast Factor: {contrastFactor.toFixed(1)}
                  </Typography>
                  <Slider
                    value={contrastFactor}
                    onChange={(e, value) => {
                      setContrastFactor(value as number);
                      handleEnhancementChange();
                    }}
                    min={0.5}
                    max={2.0}
                    step={0.1}
                    valueLabelDisplay="auto"
                  />
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={noiseReduction}
                        onChange={(e) => {
                          setNoiseReduction(e.target.checked);
                          handleEnhancementChange();
                        }}
                      />
                    }
                    label="Noise Reduction"
                  />
                </CardContent>
              </Card>
            </Grid>

            {/* Annotations */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Annotations
                  </Typography>
                  
                  <Box sx={{ mb: 2 }}>
                    <Chip
                      label={`Total: ${annotations.length}`}
                      color="primary"
                      sx={{ mr: 1 }}
                    />
                    <Chip
                      label={`Current Slice: ${currentSliceAnnotations.length}`}
                      color="secondary"
                    />
                  </Box>
                  
                  {currentSliceAnnotations.length > 0 && (
                    <Alert severity="info">
                      {currentSliceAnnotations.length} annotation(s) found on this slice
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </>
      )}
    </Box>
  );
};

export default DICOMViewerAPI;
