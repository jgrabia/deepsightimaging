import React from 'react';
import { Box, Typography, Paper, LinearProgress, Chip, Alert, Button, Card, CardContent, Grid } from '@mui/material';
import { PlayArrow as PlayArrowIcon, Stop as StopIcon, Refresh as RefreshIcon, Psychology as PsychologyIcon } from '@mui/icons-material';
import toast from 'react-hot-toast';

function StandaloneTrainingMonitor() {
  const [isConnected, setIsConnected] = React.useState(false);
  const [isTraining, setIsTraining] = React.useState(false);
  const [trainingProgress, setTrainingProgress] = React.useState({
    status: 'Disconnected',
    currentPhase: 'Not Connected',
    filesProcessed: 0,
    totalFiles: 13421,
    epoch: 0,
    totalEpochs: 30,
    accuracy: 0,
    loss: 0,
    lastUpdate: new Date().toISOString()
  });

  // Simulate training data for demo purposes
  const simulateTraining = () => {
    setIsConnected(true);
    setIsTraining(true);
    setTrainingProgress({
      status: 'Processing',
      currentPhase: 'Data Processing',
      filesProcessed: Math.floor(Math.random() * 1000),
      totalFiles: 13421,
      epoch: Math.floor(Math.random() * 5),
      totalEpochs: 30,
      accuracy: Math.random() * 0.6 + 0.4,
      loss: Math.random() * 0.1 + 0.02,
      lastUpdate: new Date().toISOString()
    });
    toast.success('Training simulation started');

    // Simulate progress updates
    const interval = setInterval(() => {
      setTrainingProgress(prev => {
        const newFilesProcessed = Math.min(prev.filesProcessed + Math.floor(Math.random() * 50), prev.totalFiles);
        const newEpoch = prev.filesProcessed >= prev.totalFiles && prev.epoch < prev.totalEpochs 
          ? prev.epoch + 1 
          : prev.epoch;
        const newAccuracy = Math.min(prev.accuracy + (Math.random() - 0.5) * 0.05, 0.95);
        const newLoss = Math.max(prev.loss + (Math.random() - 0.5) * 0.01, 0.001);

        return {
          ...prev,
          filesProcessed: newFilesProcessed,
          epoch: newEpoch,
          accuracy: newAccuracy,
          loss: newLoss,
          status: newFilesProcessed >= prev.totalFiles ? 'Training' : 'Processing',
          currentPhase: newFilesProcessed >= prev.totalFiles ? 'Training Epochs' : 'Data Processing',
          lastUpdate: new Date().toISOString()
        };
      });
    }, 2000);

    // Store interval ID for cleanup
    (window as any).trainingInterval = interval;
  };

  const stopTraining = () => {
    setIsTraining(false);
    setTrainingProgress({
      status: 'Stopped',
      currentPhase: 'Training Stopped',
      filesProcessed: 0,
      totalFiles: 13421,
      epoch: 0,
      totalEpochs: 30,
      accuracy: 0,
      loss: 0,
      lastUpdate: new Date().toISOString()
    });
    toast.success('Training stopped');
    
    // Clear interval
    if ((window as any).trainingInterval) {
      clearInterval((window as any).trainingInterval);
    }
  };

  const refreshStatus = () => {
    toast('Refreshing training status...', { icon: '🔄' });
    // In a real app, this would fetch from the API
  };

  const progress = trainingProgress.epoch > 0 ? (trainingProgress.epoch / trainingProgress.totalEpochs) * 100 : 0;
  const dataProgress = (trainingProgress.filesProcessed / trainingProgress.totalFiles) * 100;

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        🧠 AI Training Monitor - Standalone Demo
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
                  label={isConnected ? "Connected" : "Disconnected"} 
                  color={isConnected ? "success" : "error"}
                  sx={{ mb: 2 }}
                />
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    variant="contained"
                    startIcon={<PlayArrowIcon />}
                    onClick={simulateTraining}
                    disabled={isTraining}
                    size="small"
                  >
                    Start Demo
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<StopIcon />}
                    onClick={stopTraining}
                    disabled={!isTraining}
                    size="small"
                    color="error"
                  >
                    Stop
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<RefreshIcon />}
                    onClick={refreshStatus}
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
                  Training Info
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  This is a standalone demo that simulates the training process.
                  In production, this would connect to your AWS training server.
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
      {isConnected && (
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
      )}

      {/* Demo Information */}
      {!isConnected && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            🎯 Demo Mode
          </Typography>
          <Typography variant="body2">
            This is a standalone demo that simulates the training process. Click "Start Demo" to see how the training monitor works.
            In production, this would connect to your AWS training server running the real breast cancer detection model.
          </Typography>
        </Alert>
      )}

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

export default StandaloneTrainingMonitor;
