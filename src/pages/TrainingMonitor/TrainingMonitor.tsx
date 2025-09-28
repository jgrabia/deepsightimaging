import React from 'react';
import { Box, Typography, Paper, LinearProgress, Chip, Alert, Button } from '@mui/material';
import { useQuery } from 'react-query';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_AWS_API_URL || 'http://44.213.58.5:8000';

function TrainingMonitor() {
  // Fetch real training data from server
  const { data: trainingData, isLoading, error } = useQuery(
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

  // Fallback data if API not available
  const trainingProgress = trainingData || {
    epoch: 0,
    totalEpochs: 30,
    accuracy: 0,
    loss: 0,
    status: 'Initializing',
    filesProcessed: 0,
    totalFiles: 13421,
    currentPhase: 'Data Processing'
  };

  const progress = trainingProgress.epoch > 0 ? (trainingProgress.epoch / trainingProgress.totalEpochs) * 100 : 0;
  const dataProgress = (trainingProgress.filesProcessed / trainingProgress.totalFiles) * 100;

  if (isLoading) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h4" gutterBottom>Loading Training Status...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="warning">
          Could not connect to training server. Make sure the backend API is running.
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        AI Training Monitor
      </Typography>
      
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          DBT Lesion Detection Training with Real Annotations
        </Typography>
        
        {/* Data Processing Progress */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2">Data Processing</Typography>
            <Typography variant="body2">
              {trainingProgress.filesProcessed} / {trainingProgress.totalFiles} files
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
              label={`Accuracy: ${trainingProgress.accuracy.toFixed(2)}%`} 
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

      {/* Additional Info */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Training Configuration
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Chip label="Model: UNet" variant="outlined" />
          <Chip label="Learning Rate: 1e-3" variant="outlined" />
          <Chip label="Batch Size: 8" variant="outlined" />
          <Chip label="Real Annotations: 13,421 files" variant="outlined" />
        </Box>
      </Paper>
    </Box>
  );
}

export default TrainingMonitor;
