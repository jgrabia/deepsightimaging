import React from 'react';
import { Box, Typography, Paper, LinearProgress, Chip } from '@mui/material';

function TrainingMonitor() {
  const [trainingProgress] = React.useState({
    epoch: 7,
    totalEpochs: 50,
    accuracy: 53.78,
    loss: 0.0424,
    status: 'Training'
  });

  const progress = (trainingProgress.epoch / trainingProgress.totalEpochs) * 100;

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        AI Training Monitor
      </Typography>
      
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          DBT Lesion Detection Training
        </Typography>
        
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2">Progress</Typography>
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

        <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
          <Chip 
            label={`Accuracy: ${trainingProgress.accuracy}%`} 
            color="primary" 
            variant="outlined" 
          />
          <Chip 
            label={`Loss: ${trainingProgress.loss}`} 
            color="secondary" 
            variant="outlined" 
          />
          <Chip 
            label={trainingProgress.status} 
            color="success" 
            variant="filled" 
          />
        </Box>

        <Typography variant="body2" color="text.secondary">
          Training is in progress. The model is learning to detect lesions in DBT images.
          This process typically takes several hours to complete.
        </Typography>
      </Paper>
    </Box>
  );
}

export default TrainingMonitor;
