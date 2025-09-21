import React, { useState } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Button,
  Card,
  CardContent,
  CardActions,
  Chip,
  LinearProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Psychology as AIIcon,
  PlayArrow as RunIcon,
  Download as DownloadIcon,
  Visibility as ViewIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Schedule as PendingIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { useQuery, useMutation } from 'react-query';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';

interface AnalysisResult {
  analysis_id: string;
  file_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  results?: {
    lesion_detection?: {
      lesions_found: number;
      confidence: number;
      locations: Array<{
        slice: number;
        x: number;
        y: number;
        width: number;
        height: number;
        confidence: number;
      }>;
    };
    quality_assessment?: {
      overall_quality: 'excellent' | 'good' | 'fair' | 'poor';
      artifacts_detected: number;
      noise_level: 'low' | 'medium' | 'high';
      sharpness_score: number;
    };
    segmentation?: {
      structures_segmented: string[];
      volume_measurements: Record<string, number>;
    };
  };
  created_at: string;
  completed_at?: string;
}

interface DICOMFile {
  id: string;
  filename: string;
  patient_id: string;
  modality: string;
  uploaded_at: string;
}

const AIAnalysis: React.FC = () => {
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>(['lesion_detection']);
  const [showResultsDialog, setShowResultsDialog] = useState(false);
  const [selectedResult, setSelectedResult] = useState<AnalysisResult | null>(null);
  const navigate = useNavigate();

  // Available AI models
  const availableModels = [
    {
      id: 'lesion_detection',
      name: 'Lesion Detection',
      description: 'Detect and localize lesions in medical images',
      status: 'enabled',
      version: 'v1.2.3',
      accuracy: 94.2
    },
    {
      id: 'quality_assessment',
      name: 'Quality Assessment',
      description: 'Assess image quality and detect artifacts',
      status: 'enabled',
      version: 'v1.1.0',
      accuracy: 91.8
    },
    {
      id: 'segmentation',
      name: 'Tissue Segmentation',
      description: 'Segment anatomical structures',
      status: 'beta',
      version: 'v0.9.1',
      accuracy: 88.5
    }
  ];

  // Fetch available DICOM files
  const { data: dicomFiles } = useQuery<{ files: DICOMFile[] }>(
    'dicom-files-for-analysis',
    () => fetch('/api/dicom/list').then(res => res.json())
  );

  // Fetch analysis results
  const { data: analysisResults, refetch: refetchResults } = useQuery<AnalysisResult[]>(
    'analysis-results',
    () => fetch('/api/ai/results').then(res => res.json()),
    { refetchInterval: 5000 }
  );

  // Run AI analysis
  const runAnalysis = useMutation(
    async ({ fileId, models }: { fileId: string; models: string[] }) => {
      const response = await fetch(`/api/ai/analyze/${fileId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer mock_token_123'
        },
        body: JSON.stringify({ models })
      });
      
      if (!response.ok) {
        throw new Error('Analysis failed');
      }
      
      return response.json();
    },
    {
      onSuccess: () => {
        toast.success('AI analysis started successfully!');
        refetchResults();
      },
      onError: () => {
        toast.error('Failed to start AI analysis');
      }
    }
  );

  const handleRunAnalysis = () => {
    if (selectedFiles.length === 0) {
      toast.error('Please select at least one file');
      return;
    }
    
    if (selectedModels.length === 0) {
      toast.error('Please select at least one AI model');
      return;
    }

    selectedFiles.forEach(fileId => {
      runAnalysis.mutate({ fileId, models: selectedModels });
    });
  };

  const handleViewResults = (result: AnalysisResult) => {
    setSelectedResult(result);
    setShowResultsDialog(true);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <SuccessIcon color="success" />;
      case 'processing':
        return <PendingIcon color="warning" />;
      case 'failed':
        return <ErrorIcon color="error" />;
      default:
        return <PendingIcon color="disabled" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'processing':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const ModelCard: React.FC<{ model: typeof availableModels[0] }> = ({ model }) => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
          <AIIcon sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" component="h2">
            {model.name}
          </Typography>
          <Chip
            label={model.status}
            size="small"
            color={model.status === 'enabled' ? 'success' : 'warning'}
            sx={{ ml: 'auto' }}
          />
        </Box>
        <Typography variant="body2" color="textSecondary" paragraph>
          {model.description}
        </Typography>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="body2" color="textSecondary">
            Version: {model.version}
          </Typography>
          <Typography variant="body2" color="primary">
            Accuracy: {model.accuracy}%
          </Typography>
        </Box>
      </CardContent>
      <CardActions>
        <Button
          size="small"
          disabled={model.status !== 'enabled'}
          onClick={() => {
            if (selectedModels.includes(model.id)) {
              setSelectedModels(prev => prev.filter(id => id !== model.id));
            } else {
              setSelectedModels(prev => [...prev, model.id]);
            }
          }}
          color={selectedModels.includes(model.id) ? 'primary' : 'default'}
          variant={selectedModels.includes(model.id) ? 'contained' : 'outlined'}
        >
          {selectedModels.includes(model.id) ? 'Selected' : 'Select'}
        </Button>
      </CardActions>
    </Card>
  );

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 3, fontWeight: 'bold' }}>
        AI Analysis
      </Typography>

      <Grid container spacing={3}>
        {/* AI Models Selection */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Available AI Models
            </Typography>
            <Grid container spacing={2}>
              {availableModels.map((model) => (
                <Grid item xs={12} key={model.id}>
                  <ModelCard model={model} />
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>

        {/* File Selection & Controls */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Select Files for Analysis
            </Typography>
            
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>DICOM Files</InputLabel>
              <Select
                multiple
                value={selectedFiles}
                onChange={(e) => setSelectedFiles(e.target.value as string[])}
                label="DICOM Files"
              >
                {dicomFiles?.files.map((file) => (
                  <MenuItem key={file.id} value={file.id}>
                    {file.filename} ({file.modality})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Box display="flex" gap={2} mb={3}>
              <Button
                variant="contained"
                startIcon={<RunIcon />}
                onClick={handleRunAnalysis}
                disabled={runAnalysis.isLoading || selectedFiles.length === 0}
                fullWidth
              >
                {runAnalysis.isLoading ? 'Running Analysis...' : 'Run AI Analysis'}
              </Button>
              <Button
                variant="outlined"
                onClick={() => navigate('/viewer')}
                startIcon={<ViewIcon />}
              >
                View Images
              </Button>
            </Box>

            {runAnalysis.isLoading && (
              <Box>
                <Typography variant="body2" gutterBottom>
                  Running AI analysis on {selectedFiles.length} file(s)...
                </Typography>
                <LinearProgress />
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Analysis Results */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Analysis Results
            </Typography>
            
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>File</TableCell>
                    <TableCell>Models Used</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Started</TableCell>
                    <TableCell>Completed</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {analysisResults?.map((result) => {
                    const file = dicomFiles?.files.find(f => f.id === result.file_id);
                    return (
                      <TableRow key={result.analysis_id}>
                        <TableCell>
                          <Box>
                            <Typography variant="body2">
                              {file?.filename || 'Unknown File'}
                            </Typography>
                            <Typography variant="caption" color="textSecondary">
                              {file?.patient_id}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Box display="flex" gap={0.5}>
                            {selectedModels.map(modelId => {
                              const model = availableModels.find(m => m.id === modelId);
                              return (
                                <Chip
                                  key={modelId}
                                  label={model?.name || modelId}
                                  size="small"
                                  variant="outlined"
                                />
                              );
                            })}
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Box display="flex" alignItems="center" gap={1}>
                            {getStatusIcon(result.status)}
                            <Chip
                              label={result.status}
                              size="small"
                              color={getStatusColor(result.status)}
                              variant="outlined"
                            />
                          </Box>
                        </TableCell>
                        <TableCell>
                          {new Date(result.created_at).toLocaleString()}
                        </TableCell>
                        <TableCell>
                          {result.completed_at
                            ? new Date(result.completed_at).toLocaleString()
                            : '-'
                          }
                        </TableCell>
                        <TableCell>
                          <Box display="flex" gap={1}>
                            <Tooltip title="View Results">
                              <IconButton
                                size="small"
                                onClick={() => handleViewResults(result)}
                                disabled={result.status !== 'completed'}
                              >
                                <ViewIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Download Report">
                              <IconButton
                                size="small"
                                disabled={result.status !== 'completed'}
                              >
                                <DownloadIcon />
                              </IconButton>
                            </Tooltip>
                          </Box>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Results Dialog */}
      <Dialog
        open={showResultsDialog}
        onClose={() => setShowResultsDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" alignItems="center" gap={1}>
            <AIIcon />
            AI Analysis Results
          </Box>
        </DialogTitle>
        <DialogContent>
          {selectedResult && selectedResult.results && (
            <Box>
              {/* Lesion Detection Results */}
              {selectedResult.results.lesion_detection && (
                <Paper sx={{ p: 2, mb: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Lesion Detection Results
                  </Typography>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Lesions Found: {selectedResult.results.lesion_detection.lesions_found}
                  </Typography>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Confidence: {(selectedResult.results.lesion_detection.confidence * 100).toFixed(1)}%
                  </Typography>
                  {selectedResult.results.lesion_detection.locations.map((location, index) => (
                    <Box key={index} sx={{ mt: 1, p: 1, bgcolor: 'grey.100', borderRadius: 1 }}>
                      <Typography variant="body2">
                        Location {index + 1}: Slice {location.slice}, Position ({location.x}, {location.y}), 
                        Size {location.width}×{location.height}, Confidence {(location.confidence * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  ))}
                </Paper>
              )}

              {/* Quality Assessment Results */}
              {selectedResult.results.quality_assessment && (
                <Paper sx={{ p: 2, mb: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Quality Assessment Results
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">
                        Overall Quality
                      </Typography>
                      <Chip
                        label={selectedResult.results.quality_assessment.overall_quality}
                        color={
                          selectedResult.results.quality_assessment.overall_quality === 'excellent' ? 'success' :
                          selectedResult.results.quality_assessment.overall_quality === 'good' ? 'primary' :
                          selectedResult.results.quality_assessment.overall_quality === 'fair' ? 'warning' : 'error'
                        }
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">
                        Sharpness Score
                      </Typography>
                      <Typography variant="h6">
                        {selectedResult.results.quality_assessment.sharpness_score.toFixed(2)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">
                        Artifacts Detected
                      </Typography>
                      <Typography variant="h6">
                        {selectedResult.results.quality_assessment.artifacts_detected}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">
                        Noise Level
                      </Typography>
                      <Chip
                        label={selectedResult.results.quality_assessment.noise_level}
                        color={
                          selectedResult.results.quality_assessment.noise_level === 'low' ? 'success' :
                          selectedResult.results.quality_assessment.noise_level === 'medium' ? 'warning' : 'error'
                        }
                      />
                    </Grid>
                  </Grid>
                </Paper>
              )}

              {/* Segmentation Results */}
              {selectedResult.results.segmentation && (
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Segmentation Results
                  </Typography>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Structures Segmented: {selectedResult.results.segmentation.structures_segmented.join(', ')}
                  </Typography>
                  {Object.entries(selectedResult.results.segmentation.volume_measurements).map(([structure, volume]) => (
                    <Box key={structure} display="flex" justifyContent="space-between" alignItems="center">
                      <Typography variant="body2">
                        {structure}:
                      </Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {volume.toFixed(2)} cm³
                      </Typography>
                    </Box>
                  ))}
                </Paper>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowResultsDialog(false)}>
            Close
          </Button>
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            disabled={!selectedResult || selectedResult.status !== 'completed'}
          >
            Download Report
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AIAnalysis;
