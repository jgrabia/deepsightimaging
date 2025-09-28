import React from 'react';
import { Box, Typography, Paper, LinearProgress, Chip, Alert, Button, Card, CardContent, Grid } from '@mui/material';
import { PlayArrow as PlayArrowIcon, Stop as StopIcon, Refresh as RefreshIcon, Psychology as PsychologyIcon } from '@mui/icons-material';
import { useQuery, useMutation } from 'react-query';
import axios from 'axios';
import toast from 'react-hot-toast';

const API_BASE_URL = process.env.REACT_APP_AWS_API_URL || 'http://3.88.157.239:8000';

function RealTrainingMonitor() {
  // Fetch real training data from AWS server
  const { data: trainingData, isLoading, error, refetch } = useQuery(
    'training-status',
    async () => {
      const response = await axios.get(`${API_BASE_URL}/api/training/status`);
      return response.data;
    },
    {
      refetchInterval: 5000, // Refresh every 5 seconds
      refetchIntervalInBackground: true,
    }
  );

  // Start training mutation
  const startTrainingMutation = useMutation(
    async (config: any) => {
      const response = await axios.post(`${API_BASE_URL}/api/training/start`, config);
      return response.data;
    },
    {
      onSuccess: () => {
        toast.success('Training started successfully');
        refetch();
      },
      onError: (error: any) => {
        toast.error('Failed to start training: ' + (error?.message || error));
      }
    }
  );

  // Stop training mutation
  const stopTrainingMutation = useMutation(
    async () => {
      const response = await axios.post(`${API_BASE_URL}/api/training/stop`);
      return response.data;
    },
    {
      onSuccess: () => {
        toast.success('Training stopped successfully');
        refetch();
      },
      onError: (error: any) => {
        toast.error('Failed to stop training: ' + (error?.message || error));
      }
    }
  );

  const trainingProgress = trainingData || {
    status: 'Disconnected',
    currentPhase: 'Not Connected',
    filesProcessed: 0,
    totalFiles: 13421,
    epoch: 0,
    totalEpochs: 30,
    accuracy: 0,
    loss: 0,
    lastUpdate: new Date().toISOString()
  };

  const handleStartTraining = () => {
    startTrainingMutation.mutate({
      model_type: "UNet",
      learning_rate: 1e-3,
      batch_size: 8,
      epochs: 30
    });
  };

  const handleStopTraining = () => {
    stopTrainingMutation.mutate();
  };

  const progress = trainingProgress.epoch > 0 ? (trainingProgress.epoch / trainingProgress.totalEpochs) * 100 : 0;
  const dataProgress = (trainingProgress.filesProcessed / trainingProgress.totalFiles) * 100;

  if (isLoading) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h4" gutterBottom>Loading Training Status...</Typography>
        <LinearProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">
          <Typography variant="h6" gutterBottom>
            Connection Error
          </Typography>
          <Typography variant="body2">
            Could not connect to AWS training server at {API_BASE_URL}. 
            Make sure the enhanced API server is running.
          </Typography>
          <Button 
            variant="outlined" 
            onClick={() => refetch()} 
            sx={{ mt: 2 }}
          >
            Retry Connection
          </Button>
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        🧠 AI Training Monitor - AWS Server
      </Typography>
      
      {/* Connection Status */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Training Status
        </Typography>
        
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Connection Status
                </Typography>
                <Chip 
                  label="Connected to AWS Server" 
                  color="success"
                  sx={{ mb: 2 }}
                />
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    variant="contained"
                    startIcon={<PlayArrowIcon />}
                    onClick={handleStartTraining}
                    disabled={startTrainingMutation.isLoading || trainingProgress.status === 'Training'}
                    size="small"
                  >
                    {startTrainingMutation.isLoading ? 'Starting...' : 'Start Training'}
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<StopIcon />}
                    onClick={handleStopTraining}
                    disabled={stopTrainingMutation.isLoading || trainingProgress.status !== 'Training'}
                    size="small"
                    color="error"
                  >
                    {stopTrainingMutation.isLoading ? 'Stopping...' : 'Stop'}
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<RefreshIcon />}
                    onClick={() => refetch()}
                    size="small"
                  >
                    Refresh
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Server Info
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Connected to AWS server: {API_BASE_URL}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Real-time training data from your 13,421 DICOM files with annotations.
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Chip label="Model: UNet" variant="outlined" size="small" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="Learning Rate: 1e-3" variant="outlined" size="small" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="Batch Size: 8" variant="outlined" size="small" sx={{ mr: 1, mb: 1 }} />
                  <Chip label="Real Annotations: 13,421 files" variant="outlined" size="small" />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>

      {/* Training Progress */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          DBT Lesion Detection Training with Real Annotations
        </Typography>
        
        {/* Data Processing Progress */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2">Data Processing</Typography>
            <Typography variant="body2">
              {trainingProgress.filesProcessed.toLocaleString()} / {trainingProgress.totalFiles.toLocaleString()} files
            </Typography>
          </Box>
          <LinearProgress 
            variant="determinate" 
            value={dataProgress} 
            sx={{ height: 8, borderRadius: 4 }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            {dataProgress.toFixed(1)}% Complete - {trainingProgress.currentPhase}
          </Typography>
        </Box>

        {/* Training Progress */}
        {trainingProgress.epoch > 0 && (
          <Box sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="body2">Training Progress</Typography>
              <Typography variant="body2">
                Epoch {trainingProgress.epoch} / {trainingProgress.totalEpochs}
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={progress} 
              sx={{ height: 8, borderRadius: 4 }}
            />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              {progress.toFixed(1)}% Complete
            </Typography>
          </Box>
        )}

        <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
          <Chip 
            label={`Status: ${trainingProgress.status}`} 
            color={trainingProgress.status === 'Training' ? 'primary' : 'default'}
            variant="outlined" 
          />
          {trainingProgress.accuracy > 0 && (
            <Chip 
              label={`Accuracy: ${(trainingProgress.accuracy * 100).toFixed(2)}%`} 
              color="primary" 
              variant="outlined" 
            />
          )}
          {trainingProgress.loss > 0 && (
            <Chip 
              label={`Loss: ${trainingProgress.loss.toFixed(4)}`} 
              color="secondary" 
              variant="outlined" 
            />
          )}
        </Box>

        <Typography variant="body2" color="text.secondary">
          {trainingProgress.epoch > 0 
            ? "Training is in progress. The model is learning to detect lesions in DBT images using real annotations."
            : "Processing DICOM files and creating annotation labels. Training will begin once data preparation is complete."
          }
        </Typography>
      </Paper>

      {/* Training Configuration */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Training Configuration
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" gutterBottom>
              Model Architecture
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Chip label="UNet" variant="outlined" />
              <Chip label="64 Filters" variant="outlined" />
              <Chip label="Dropout: 0.1" variant="outlined" />
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" gutterBottom>
              Training Parameters
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Chip label="Learning Rate: 1e-3" variant="outlined" />
              <Chip label="Batch Size: 8" variant="outlined" />
              <Chip label="Epochs: 30" variant="outlined" />
            </Box>
          </Grid>
          
          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom>
              Data & Annotations
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Chip label="13,421 DICOM Files" variant="outlined" />
              <Chip label="3,052 JSON Annotations" variant="outlined" />
              <Chip label="6 CSV Metadata Files" variant="outlined" />
              <Chip label="Real Lesion Coordinates" variant="outlined" color="primary" />
            </Box>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
}

export default RealTrainingMonitor;
