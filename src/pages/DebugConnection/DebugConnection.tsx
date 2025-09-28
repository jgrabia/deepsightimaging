import React from 'react';
import { Box, Typography, Paper, Button, Alert, CircularProgress, Accordion, AccordionSummary, AccordionDetails } from '@mui/material';
import { ExpandMore as ExpandMoreIcon, Refresh as RefreshIcon } from '@mui/icons-material';

const API_BASE_URL = process.env.REACT_APP_AWS_API_URL || 'http://3.88.157.239:8000';

function DebugConnection() {
  const [results, setResults] = React.useState<any[]>([]);
  const [isRunning, setIsRunning] = React.useState(false);

  const runDiagnostics = async () => {
    setIsRunning(true);
    const newResults: any[] = [];

    // Test 1: Basic connectivity
    try {
      const response = await fetch(`${API_BASE_URL}/`, {
        method: 'GET',
        mode: 'cors',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        newResults.push({
          test: 'Basic Connectivity',
          status: 'success',
          message: 'Connected successfully',
          data: data
        });
      } else {
        newResults.push({
          test: 'Basic Connectivity',
          status: 'error',
          message: `HTTP ${response.status}: ${response.statusText}`,
          data: null
        });
      }
    } catch (error) {
      newResults.push({
        test: 'Basic Connectivity',
        status: 'error',
        message: `Network error: ${error}`,
        data: null
      });
    }

    // Test 2: Training status
    try {
      const response = await fetch(`${API_BASE_URL}/api/training/status`, {
        method: 'GET',
        mode: 'cors'
      });
      
      if (response.ok) {
        const data = await response.json();
        newResults.push({
          test: 'Training Status',
          status: 'success',
          message: 'Training status retrieved',
          data: data
        });
      } else {
        newResults.push({
          test: 'Training Status',
          status: 'error',
          message: `HTTP ${response.status}: ${response.statusText}`,
          data: null
        });
      }
    } catch (error) {
      newResults.push({
        test: 'Training Status',
        status: 'error',
        message: `Network error: ${error}`,
        data: null
      });
    }

    // Test 3: TCIA collections
    try {
      const response = await fetch(`${API_BASE_URL}/api/tcia/collections`, {
        method: 'GET',
        mode: 'cors'
      });
      
      if (response.ok) {
        const data = await response.json();
        newResults.push({
          test: 'TCIA Collections',
          status: 'success',
          message: 'TCIA collections retrieved',
          data: data
        });
      } else {
        newResults.push({
          test: 'TCIA Collections',
          status: 'error',
          message: `HTTP ${response.status}: ${response.statusText}`,
          data: null
        });
      }
    } catch (error) {
      newResults.push({
        test: 'TCIA Collections',
        status: 'error',
        message: `Network error: ${error}`,
        data: null
      });
    }

    // Test 4: File upload (simulation)
    try {
      const formData = new FormData();
      formData.append('files', new Blob(['test content'], { type: 'application/dicom' }), 'test.dcm');
      
      const response = await fetch(`${API_BASE_URL}/api/files/upload`, {
        method: 'POST',
        mode: 'cors',
        body: formData
      });
      
      if (response.ok) {
        const data = await response.json();
        newResults.push({
          test: 'File Upload',
          status: 'success',
          message: 'File upload endpoint working',
          data: data
        });
      } else {
        newResults.push({
          test: 'File Upload',
          status: 'error',
          message: `HTTP ${response.status}: ${response.statusText}`,
          data: null
        });
      }
    } catch (error) {
      newResults.push({
        test: 'File Upload',
        status: 'error',
        message: `Network error: ${error}`,
        data: null
      });
    }

    // Test 5: CORS preflight
    try {
      const response = await fetch(`${API_BASE_URL}/api/files/upload`, {
        method: 'OPTIONS',
        mode: 'cors',
        headers: {
          'Access-Control-Request-Method': 'POST',
          'Access-Control-Request-Headers': 'Content-Type'
        }
      });
      
      newResults.push({
        test: 'CORS Preflight',
        status: response.ok ? 'success' : 'warning',
        message: response.ok ? 'CORS preflight working' : `CORS preflight failed: ${response.status}`,
        data: { status: response.status, headers: Object.fromEntries(response.headers.entries()) }
      });
    } catch (error) {
      newResults.push({
        test: 'CORS Preflight',
        status: 'error',
        message: `CORS error: ${error}`,
        data: null
      });
    }

    setResults(newResults);
    setIsRunning(false);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        🔍 AWS Server Connection Diagnostics
      </Typography>
      
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Connection Configuration
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          API Base URL: <code>{API_BASE_URL}</code>
        </Typography>
        
        <Button
          variant="contained"
          startIcon={isRunning ? <CircularProgress size={20} /> : <RefreshIcon />}
          onClick={runDiagnostics}
          disabled={isRunning}
          size="large"
        >
          {isRunning ? 'Running Diagnostics...' : 'Run Full Diagnostics'}
        </Button>
      </Paper>

      {results.length > 0 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Diagnostic Results
          </Typography>
          
          {results.map((result, index) => (
            <Accordion key={index} sx={{ mb: 1 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Typography variant="h6">
                    {result.test}
                  </Typography>
                  <Alert 
                    severity={result.status === 'success' ? 'success' : result.status === 'warning' ? 'warning' : 'error'}
                    sx={{ minWidth: 0, py: 0 }}
                  >
                    {result.status === 'success' ? '✅' : result.status === 'warning' ? '⚠️' : '❌'}
                  </Alert>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {result.message}
                </Typography>
                {result.data && (
                  <Paper sx={{ p: 2, backgroundColor: 'grey.100' }}>
                    <pre style={{ margin: 0, fontSize: '12px', overflow: 'auto' }}>
                      {JSON.stringify(result.data, null, 2)}
                    </pre>
                  </Paper>
                )}
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      )}

      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Common Issues & Solutions
        </Typography>
        <Box component="div">
          <Typography variant="body2" component="div">
            <strong>❌ Network Error:</strong>
            <ul>
              <li>AWS Security Group not allowing port 8000</li>
              <li>API server not running</li>
              <li>Firewall blocking connections</li>
            </ul>
            
            <strong>❌ CORS Error:</strong>
            <ul>
              <li>Server not configured for cross-origin requests</li>
              <li>Missing preflight handling</li>
            </ul>
            
            <strong>✅ Solutions:</strong>
            <ul>
              <li>Fix AWS Security Group: Allow inbound port 8000 from 0.0.0.0/0</li>
              <li>Restart API server on AWS</li>
              <li>Check server logs for errors</li>
            </ul>
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
}

export default DebugConnection;
