import React from 'react';
import { Box, Typography, Paper, TextField, Button, Alert, Grid } from '@mui/material';

function CloudAPI() {
  const [config, setConfig] = React.useState({
    apiUrl: 'https://api.deepsightimaging.ai',
    apiToken: '',
    uploadEndpoint: '/api/v1/upload'
  });

  const [isValidating, setIsValidating] = React.useState(false);
  const [validationResult, setValidationResult] = React.useState<'success' | 'error' | null>(null);

  const handleConfigChange = (field: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setConfig({
      ...config,
      [field]: event.target.value
    });
  };

  const handleValidate = async () => {
    setIsValidating(true);
    // Simulate API validation
    setTimeout(() => {
      setValidationResult('success');
      setIsValidating(false);
    }, 2000);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Cloud API Configuration
      </Typography>
      
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          DeepSight Imaging AI API Settings
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="API Base URL"
              value={config.apiUrl}
              onChange={handleConfigChange('apiUrl')}
              variant="outlined"
              margin="normal"
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Upload Endpoint"
              value={config.uploadEndpoint}
              onChange={handleConfigChange('uploadEndpoint')}
              variant="outlined"
              margin="normal"
            />
          </Grid>
          
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="API Token"
              value={config.apiToken}
              onChange={handleConfigChange('apiToken')}
              variant="outlined"
              margin="normal"
              type="password"
              helperText="Your secure API token for authentication"
            />
          </Grid>
          
          <Grid item xs={12}>
            <Button
              variant="contained"
              onClick={handleValidate}
              disabled={isValidating || !config.apiToken}
              sx={{ mt: 2 }}
            >
              {isValidating ? 'Validating...' : 'Validate Connection'}
            </Button>
          </Grid>
        </Grid>

        {validationResult === 'success' && (
          <Alert severity="success" sx={{ mt: 2 }}>
            API connection validated successfully!
          </Alert>
        )}
        
        {validationResult === 'error' && (
          <Alert severity="error" sx={{ mt: 2 }}>
            Failed to connect to API. Please check your configuration.
          </Alert>
        )}
      </Paper>
    </Box>
  );
}

export default CloudAPI;
