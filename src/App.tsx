import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';

// Components
import Layout from './components/Layout/Layout';
import Dashboard from './pages/Dashboard/Dashboard';
import DICOMViewer from './pages/DICOMViewer/DICOMViewerAPI';
import AIAnalysis from './pages/AIAnalysis/AIAnalysis';
import Workflow from './pages/Workflow/CompleteWorkflow';
import TrainingMonitor from './pages/TrainingMonitor/StandaloneTrainingMonitor';
import Settings from './pages/Settings/Settings';
import CloudAPI from './pages/CloudAPI/CloudAPI';
import Login from './pages/Login/Login';

// Create theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 8,
  },
});

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  const [isAuthenticated, setIsAuthenticated] = React.useState(false);

  // Check authentication on app load
  React.useEffect(() => {
    const token = localStorage.getItem('auth_token');
    setIsAuthenticated(!!token);
  }, []);

  if (!isAuthenticated) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <QueryClientProvider client={queryClient}>
          <Login onLogin={() => setIsAuthenticated(true)} />
          <Toaster position="top-right" />
        </QueryClientProvider>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <QueryClientProvider client={queryClient}>
        <Router>
          <Layout onLogout={() => setIsAuthenticated(false)}>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/viewer" element={<DICOMViewer />} />
              <Route path="/ai-analysis" element={<AIAnalysis />} />
              <Route path="/workflow" element={<Workflow />} />
              <Route path="/training" element={<TrainingMonitor />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/cloud-api" element={<CloudAPI />} />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </Layout>
          <Toaster position="top-right" />
        </Router>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

export default App;
