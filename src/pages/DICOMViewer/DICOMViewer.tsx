import React, { useState, useRef, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Button,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  Switch,
  FormControlLabel,
  TextField,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  RotateLeft as RotateIcon,
  Adjust as AdjustIcon,
  Info as InfoIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import { useQuery, useMutation } from 'react-query';
import { useDropzone } from 'react-dropzone';
import toast from 'react-hot-toast';

interface DICOMFile {
  id: string;
  filename: string;
  patient_id: string;
  study_date: string;
  modality: string;
  size: number;
  uploaded_at: string;
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

interface DICOMMetadata {
  patient_id: string;
  patient_name: string;
  study_description: string;
  series_description: string;
  modality: string;
  study_date: string;
  manufacturer: string;
  model: string;
  series_number: string;
  instance_number: string;
}

const DICOMViewer: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<DICOMFile | null>(null);
  const [currentSlice, setCurrentSlice] = useState(0);
  const [totalSlices, setTotalSlices] = useState(65);
  const [windowCenter, setWindowCenter] = useState(40);
  const [windowWidth, setWindowWidth] = useState(400);
  const [zoom, setZoom] = useState(1);
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [annotationFolder, setAnnotationFolder] = useState('');
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1000);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const playbackRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch DICOM files list
  const { data: dicomFiles, refetch: refetchFiles } = useQuery<{ files: DICOMFile[] }>(
    'dicom-files',
    () => fetch('/api/dicom/list').then(res => res.json()),
    { refetchInterval: 30000 }
  );

  // Fetch DICOM metadata
  const { data: metadata } = useQuery<{ metadata: DICOMMetadata }>(
    ['dicom-metadata', selectedFile?.id],
    () => fetch(`/api/dicom/${selectedFile?.id}`).then(res => res.json()),
    { enabled: !!selectedFile }
  );

  // Handle file upload
  const onDrop = useMutation(
    async (files: File[]) => {
      const file = files[0];
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('/api/v1/upload', {
        method: 'POST',
        body: formData,
        headers: {
          'Authorization': 'Bearer mock_token_123',
          'X-Customer-ID': 'hospital_001'
        }
      });
      
      if (!response.ok) {
        throw new Error('Upload failed');
      }
      
      return response.json();
    },
    {
      onSuccess: () => {
        toast.success('DICOM file uploaded successfully!');
        refetchFiles();
      },
      onError: () => {
        toast.error('Upload failed. Please try again.');
      }
    }
  );

  // File dropzone
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: onDrop.mutate,
    accept: {
      'application/dicom': ['.dcm', '.dicom']
    },
    multiple: false
  });

  // Load annotations for selected file
  useEffect(() => {
    if (selectedFile && showAnnotations) {
      // Mock annotation loading - replace with actual API call
      const mockAnnotations: Annotation[] = [
        {
          slice: 21,
          x: 1276,
          y: 672,
          width: 228,
          height: 219,
          class: 'benign',
          view: 'rcc'
        }
      ];
      setAnnotations(mockAnnotations);
    } else {
      setAnnotations([]);
    }
  }, [selectedFile, showAnnotations]);

  // Cine mode playback
  useEffect(() => {
    if (isPlaying) {
      playbackRef.current = setInterval(() => {
        setCurrentSlice(prev => (prev + 1) % totalSlices);
      }, playbackSpeed);
    } else {
      if (playbackRef.current) {
        clearInterval(playbackRef.current);
        playbackRef.current = null;
      }
    }

    return () => {
      if (playbackRef.current) {
        clearInterval(playbackRef.current);
      }
    };
  }, [isPlaying, playbackSpeed, totalSlices]);

  // Mock image rendering - replace with actual DICOM rendering
  const renderImage = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw mock DICOM image
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw mock breast image
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.ellipse(canvas.width / 2, canvas.height / 2, 150, 200, 0, 0, 2 * Math.PI);
    ctx.fill();
    
    // Draw annotations
    if (showAnnotations && annotations.length > 0) {
      annotations.forEach((annotation, index) => {
        if (annotation.slice === currentSlice) {
          // Draw annotation box
          ctx.strokeStyle = '#ffeb3b';
          ctx.lineWidth = 3;
          ctx.strokeRect(
            annotation.x * zoom,
            annotation.y * zoom,
            annotation.width * zoom,
            annotation.height * zoom
          );
          
          // Draw corner markers
          const markerSize = 10;
          ctx.fillStyle = '#ffeb3b';
          ctx.fillRect(annotation.x * zoom - markerSize/2, annotation.y * zoom - markerSize/2, markerSize, markerSize);
          ctx.fillRect((annotation.x + annotation.width) * zoom - markerSize/2, annotation.y * zoom - markerSize/2, markerSize, markerSize);
          ctx.fillRect(annotation.x * zoom - markerSize/2, (annotation.y + annotation.height) * zoom - markerSize/2, markerSize, markerSize);
          ctx.fillRect((annotation.x + annotation.width) * zoom - markerSize/2, (annotation.y + annotation.height) * zoom - markerSize/2, markerSize, markerSize);
          
          // Draw center crosshair
          ctx.strokeStyle = '#ffeb3b';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo((annotation.x + annotation.width/2) * zoom, annotation.y * zoom);
          ctx.lineTo((annotation.x + annotation.width/2) * zoom, (annotation.y + annotation.height) * zoom);
          ctx.moveTo(annotation.x * zoom, (annotation.y + annotation.height/2) * zoom);
          ctx.lineTo((annotation.x + annotation.width) * zoom, (annotation.y + annotation.height/2) * zoom);
          ctx.stroke();
        }
      });
    }
    
    // Draw slice indicator
    ctx.fillStyle = '#000000';
    ctx.font = '16px Arial';
    ctx.fillText(`Slice ${currentSlice + 1}/${totalSlices}`, 10, 30);
  };

  useEffect(() => {
    renderImage();
  }, [currentSlice, zoom, showAnnotations, annotations]);

  const handleFileSelect = (file: DICOMFile) => {
    setSelectedFile(file);
    setCurrentSlice(0);
    setIsPlaying(false);
  };

  const handleSliceChange = (_: Event, value: number | number[]) => {
    setCurrentSlice(value as number);
  };

  const handleWindowLevelChange = (center: number, width: number) => {
    setWindowCenter(center);
    setWindowWidth(width);
  };

  const togglePlayback = () => {
    setIsPlaying(!isPlaying);
  };

  const handleZoomChange = (factor: number) => {
    setZoom(prev => Math.max(0.5, Math.min(5, prev * factor)));
  };

  const windowPresets = [
    { name: 'Breast', center: 40, width: 400 },
    { name: 'Soft Tissue', center: 50, width: 400 },
    { name: 'Bone', center: 400, width: 1800 },
    { name: 'Lung', center: -600, width: 1500 },
  ];

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 3, fontWeight: 'bold' }}>
        DICOM Viewer
      </Typography>

      <Grid container spacing={3}>
        {/* File Selection Panel */}
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Files
            </Typography>
            
            {/* Upload Area */}
            <Box
              {...getRootProps()}
              sx={{
                border: '2px dashed',
                borderColor: isDragActive ? 'primary.main' : 'grey.300',
                borderRadius: 2,
                p: 3,
                textAlign: 'center',
                cursor: 'pointer',
                mb: 2,
                backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
                '&:hover': {
                  backgroundColor: 'action.hover',
                }
              }}
            >
              <input {...getInputProps()} />
              <Typography variant="body2" color="textSecondary">
                {isDragActive
                  ? 'Drop DICOM file here...'
                  : 'Drag & drop DICOM file or click to browse'}
              </Typography>
              {onDrop.isLoading && (
                <Typography variant="body2" color="primary" sx={{ mt: 1 }}>
                  Uploading...
                </Typography>
              )}
            </Box>

            {/* File List */}
            <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
              {dicomFiles?.files.map((file) => (
                <Card
                  key={file.id}
                  sx={{
                    mb: 1,
                    cursor: 'pointer',
                    border: selectedFile?.id === file.id ? 2 : 1,
                    borderColor: selectedFile?.id === file.id ? 'primary.main' : 'divider',
                  }}
                  onClick={() => handleFileSelect(file)}
                >
                  <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                    <Typography variant="subtitle2" noWrap>
                      {file.filename}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      {file.patient_id} • {file.modality}
                    </Typography>
                  </CardContent>
                </Card>
              ))}
            </Box>
          </Paper>

          {/* Annotation Controls */}
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Annotations
            </Typography>
            
            <FormControlLabel
              control={
                <Switch
                  checked={showAnnotations}
                  onChange={(e) => setShowAnnotations(e.target.checked)}
                />
              }
              label="Show Annotations"
            />
            
            <TextField
              fullWidth
              label="Annotation Folder Path"
              value={annotationFolder}
              onChange={(e) => setAnnotationFolder(e.target.value)}
              size="small"
              sx={{ mt: 2 }}
              placeholder="C:\MRIAPP\annotations"
            />
            
            {annotations.length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Loaded Annotations:
                </Typography>
                {annotations.map((annotation, index) => (
                  <Chip
                    key={index}
                    label={`Slice ${annotation.slice}: ${annotation.class}`}
                    size="small"
                    sx={{ mr: 1, mb: 1 }}
                  />
                ))}
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Main Viewer */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">
                {selectedFile ? selectedFile.filename : 'No file selected'}
              </Typography>
              <Box>
                <Tooltip title="Zoom In">
                  <IconButton onClick={() => handleZoomChange(1.2)}>
                    <ZoomInIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Zoom Out">
                  <IconButton onClick={() => handleZoomChange(0.8)}>
                    <ZoomOutIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Reset Zoom">
                  <IconButton onClick={() => setZoom(1)}>
                    <ZoomOutIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>

            {/* Image Canvas */}
            <Box
              sx={{
                border: 1,
                borderColor: 'divider',
                borderRadius: 1,
                overflow: 'hidden',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                minHeight: 400,
                backgroundColor: '#000',
              }}
            >
              <canvas
                ref={canvasRef}
                width={400}
                height={500}
                style={{
                  maxWidth: '100%',
                  maxHeight: '100%',
                  objectFit: 'contain',
                }}
              />
            </Box>

            {/* Slice Navigation */}
            <Box sx={{ mt: 2 }}>
              <Box display="flex" alignItems="center" gap={2}>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => setCurrentSlice(Math.max(0, currentSlice - 1))}
                  disabled={currentSlice === 0}
                >
                  Previous
                </Button>
                
                <Box sx={{ flexGrow: 1, px: 2 }}>
                  <Slider
                    value={currentSlice}
                    onChange={handleSliceChange}
                    min={0}
                    max={totalSlices - 1}
                    step={1}
                    marks={[
                      { value: 0, label: '1' },
                      { value: Math.floor(totalSlices / 2), label: Math.floor(totalSlices / 2) + 1 },
                      { value: totalSlices - 1, label: totalSlices },
                    ]}
                  />
                </Box>
                
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => setCurrentSlice(Math.min(totalSlices - 1, currentSlice + 1))}
                  disabled={currentSlice === totalSlices - 1}
                >
                  Next
                </Button>
              </Box>

              {/* Cine Controls */}
              <Box display="flex" alignItems="center" gap={1} mt={2}>
                <IconButton onClick={togglePlayback} color="primary">
                  {isPlaying ? <PauseIcon /> : <PlayIcon />}
                </IconButton>
                <Typography variant="body2">
                  {isPlaying ? 'Playing' : 'Paused'}
                </Typography>
                <FormControl size="small" sx={{ minWidth: 120, ml: 2 }}>
                  <InputLabel>Speed</InputLabel>
                  <Select
                    value={playbackSpeed}
                    onChange={(e) => setPlaybackSpeed(e.target.value as number)}
                    label="Speed"
                  >
                    <MenuItem value={2000}>0.5x</MenuItem>
                    <MenuItem value={1000}>1x</MenuItem>
                    <MenuItem value={500}>2x</MenuItem>
                    <MenuItem value={250}>4x</MenuItem>
                  </Select>
                </FormControl>
              </Box>
            </Box>
          </Paper>
        </Grid>

        {/* Controls Panel */}
        <Grid item xs={12} md={3}>
          {/* Window/Level Controls */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Window/Level
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Presets
              </Typography>
              {windowPresets.map((preset) => (
                <Button
                  key={preset.name}
                  size="small"
                  variant="outlined"
                  sx={{ mr: 1, mb: 1 }}
                  onClick={() => handleWindowLevelChange(preset.center, preset.width)}
                >
                  {preset.name}
                </Button>
              ))}
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Window Center: {windowCenter}
              </Typography>
              <Slider
                value={windowCenter}
                onChange={(_, value) => handleWindowLevelChange(value as number, windowWidth)}
                min={-1000}
                max={1000}
                step={10}
              />
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Window Width: {windowWidth}
              </Typography>
              <Slider
                value={windowWidth}
                onChange={(_, value) => handleWindowLevelChange(windowCenter, value as number)}
                min={1}
                max={2000}
                step={10}
              />
            </Box>
          </Paper>

          {/* Image Information */}
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Image Info
            </Typography>
            
            {metadata?.metadata && (
              <Box>
                <Typography variant="body2" color="textSecondary">
                  <strong>Patient:</strong> {metadata.metadata.patient_name}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  <strong>ID:</strong> {metadata.metadata.patient_id}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  <strong>Study:</strong> {metadata.metadata.study_description}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  <strong>Modality:</strong> {metadata.metadata.modality}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  <strong>Date:</strong> {metadata.metadata.study_date}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  <strong>Scanner:</strong> {metadata.metadata.manufacturer} {metadata.metadata.model}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  <strong>Series:</strong> {metadata.metadata.series_number}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  <strong>Instance:</strong> {metadata.metadata.instance_number}
                </Typography>
              </Box>
            )}
            
            {!selectedFile && (
              <Typography variant="body2" color="textSecondary">
                Select a DICOM file to view metadata
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DICOMViewer;
