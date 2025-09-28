import React from 'react';
import { Box, Typography, Paper, Button, Alert, CircularProgress } from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';

const API_BASE_URL = process.env.REACT_APP_AWS_API_URL || 'http://3.88.157.239:8000';

function TestConnection() {
  const [connectionStatus, setConnectionStatus] = React.useState<'testing' | 'success' | 'error' | 'idle'>('idle');
  const [response, setResponse] = React.useState<any>(null);
  const [error, setError] = React.useState<string>('');

  const testConnection = async () => {
    setConnectionStatus('testing');
    setError('');
    
    try {
      const res = await fetch(`${API_BASE_URL}/`);
      if (res.ok) {
        const data = await res.json();
        setResponse(data);
        setConnectionStatus('success');
      } else {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setConnectionStatus('error');
    }
  };

  const testTrainingStatus = async () => {
    setConnectionStatus('testing');
    setError('');
    
    try {
      const res = await fetch(`${API_BASE_URL}/api/training/status`);
      if (res.ok) {
        const data = await res.json();
        setResponse(data);
        setConnectionStatus('success');
      } else {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setConnectionStatus('error');
    }
  };

  const testSystemStatus = async () => {
    setConnectionStatus('testing');
    setError('');
    
    try {
      const res = await fetch(`${API_BASE_URL}/api/system/status`);
      if (res.ok) {
        const data = await res.json();
        setResponse(data);
        setConnectionStatus('success');
      } else {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setConnectionStatus('error');
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        🔌 AWS Server Connection Test
      </Typography>
      
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Connection Configuration
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          API Base URL: <code>{API_BASE_URL}</code>
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
          <Button
            variant="contained"
            startIcon={connectionStatus === 'testing' ? <CircularProgress size={20} /> : <RefreshIcon />}
            onClick={testConnection}
            disabled={connectionStatus === 'testing'}
          >
            Test Basic Connection
          </Button>
          
          <Button
            variant="outlined"
            onClick={testTrainingStatus}
            disabled={connectionStatus === 'testing'}
          >
            Test Training Status
          </Button>
          
          <Button
            variant="outlined"
            onClick={testSystemStatus}
            disabled={connectionStatus === 'testing'}
          >
            Test System Status
          </Button>
        </Box>

        {connectionStatus === 'success' && (
          <Alert severity="success" sx={{ mb: 2 }}>
            ✅ Connection successful!
          </Alert>
        )}

        {connectionStatus === 'error' && (
          <Alert severity="error" sx={{ mb: 2 }}>
            ❌ Connection failed: {error}
          </Alert>
        )}

        {response && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Response:
            </Typography>
            <Paper sx={{ p: 2, backgroundColor: 'grey.100' }}>
              <pre style={{ margin: 0, fontSize: '12px', overflow: 'auto' }}>
                {JSON.stringify(response, null, 2)}
              </pre>
            </Paper>
          </Box>
        )}
      </Paper>

      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Expected Endpoints
        </Typography>
        <Typography variant="body2" component="div">
          <ul>
            <li><code>GET /</code> - Basic server info</li>
            <li><code>GET /api/training/status</code> - Training progress</li>
            <li><code>GET /api/system/status</code> - System resources</li>
            <li><code>GET /api/tcia/collections</code> - TCIA collections</li>
            <li><code>POST /api/files/upload</code> - File upload</li>
            <li><code>POST /api/ai/analyze</code> - AI analysis</li>
          </ul>
        </Typography>
      </Paper>
    </Box>
  );
}

export default TestConnection;
