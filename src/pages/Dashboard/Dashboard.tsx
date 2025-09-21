import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  Avatar,
  LinearProgress,
} from '@mui/material';
import {
  Visibility as ViewerIcon,
  Psychology as AIIcon,
  CloudUpload as CloudIcon,
  TrendingUp as TrendingIcon,
  Assignment as ReportIcon,
  Storage as StorageIcon,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import { useNavigate } from 'react-router-dom';

// Mock data - replace with actual API calls
const mockStats = {
  totalImages: 1247,
  processedToday: 23,
  aiAnalysisComplete: 89,
  cloudUploads: 156,
};

const mockRecentActivity = [
  { id: 1, type: 'upload', patient: 'P001234', timestamp: '2 min ago', status: 'completed' },
  { id: 2, type: 'analysis', patient: 'P001235', timestamp: '5 min ago', status: 'processing' },
  { id: 3, type: 'report', patient: 'P001236', timestamp: '12 min ago', status: 'completed' },
  { id: 4, type: 'upload', patient: 'P001237', timestamp: '18 min ago', status: 'completed' },
];

const mockTrainingProgress = {
  epoch: 15,
  totalEpochs: 50,
  accuracy: 0.8734,
  loss: 0.1245,
};

const Dashboard: React.FC = () => {
  const navigate = useNavigate();

  // Mock API calls
  const { data: stats } = useQuery('dashboard-stats', () => Promise.resolve(mockStats));
  const { data: recentActivity } = useQuery('recent-activity', () => Promise.resolve(mockRecentActivity));
  const { data: trainingProgress } = useQuery('training-progress', () => Promise.resolve(mockTrainingProgress));

  const StatCard: React.FC<{
    title: string;
    value: string | number;
    icon: React.ReactNode;
    color: string;
    onClick?: () => void;
  }> = ({ title, value, icon, color, onClick }) => (
    <Card 
      sx={{ 
        height: '100%', 
        cursor: onClick ? 'pointer' : 'default',
        '&:hover': onClick ? { boxShadow: 3 } : {}
      }}
      onClick={onClick}
    >
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography color="textSecondary" gutterBottom variant="h6">
              {title}
            </Typography>
            <Typography variant="h4" component="h2" sx={{ fontWeight: 'bold' }}>
              {value}
            </Typography>
          </Box>
          <Avatar sx={{ backgroundColor: color, width: 56, height: 56 }}>
            {icon}
          </Avatar>
        </Box>
      </CardContent>
    </Card>
  );

  const QuickActionCard: React.FC<{
    title: string;
    description: string;
    icon: React.ReactNode;
    onClick: () => void;
  }> = ({ title, description, icon, onClick }) => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
          <Avatar sx={{ backgroundColor: 'primary.main', mr: 2 }}>
            {icon}
          </Avatar>
          <Typography variant="h6" component="h2">
            {title}
          </Typography>
        </Box>
        <Typography color="textSecondary" variant="body2">
          {description}
        </Typography>
      </CardContent>
      <CardActions>
        <Button size="small" onClick={onClick}>
          Launch
        </Button>
      </CardActions>
    </Card>
  );

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 4, fontWeight: 'bold' }}>
        Dashboard
      </Typography>

      {/* Statistics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Images"
            value={stats?.totalImages || 0}
            icon={<StorageIcon />}
            color="#1976d2"
            onClick={() => navigate('/viewer')}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Processed Today"
            value={stats?.processedToday || 0}
            icon={<ViewerIcon />}
            color="#388e3c"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="AI Analysis"
            value={stats?.aiAnalysisComplete || 0}
            icon={<AIIcon />}
            color="#f57c00"
            onClick={() => navigate('/ai-analysis')}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Cloud Uploads"
            value={stats?.cloudUploads || 0}
            icon={<CloudIcon />}
            color="#7b1fa2"
            onClick={() => navigate('/cloud-api')}
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Quick Actions */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
              Quick Actions
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <QuickActionCard
                  title="DICOM Viewer"
                  description="View and analyze medical images"
                  icon={<ViewerIcon />}
                  onClick={() => navigate('/viewer')}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <QuickActionCard
                  title="AI Analysis"
                  description="Run AI-powered image analysis"
                  icon={<AIIcon />}
                  onClick={() => navigate('/ai-analysis')}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <QuickActionCard
                  title="Workflow"
                  description="Manage imaging workflow"
                  icon={<ReportIcon />}
                  onClick={() => navigate('/workflow')}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <QuickActionCard
                  title="Training Monitor"
                  description="Monitor AI model training"
                  icon={<TrendingIcon />}
                  onClick={() => navigate('/training')}
                />
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
              Recent Activity
            </Typography>
            <Box>
              {recentActivity?.map((activity) => (
                <Box
                  key={activity.id}
                  display="flex"
                  alignItems="center"
                  justifyContent="space-between"
                  py={1}
                  borderBottom="1px solid #f0f0f0"
                >
                  <Box display="flex" alignItems="center">
                    <Avatar sx={{ width: 32, height: 32, mr: 2, fontSize: '0.875rem' }}>
                      {activity.type === 'upload' && <CloudIcon />}
                      {activity.type === 'analysis' && <AIIcon />}
                      {activity.type === 'report' && <ReportIcon />}
                    </Avatar>
                    <Box>
                      <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                        Patient {activity.patient}
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        {activity.timestamp}
                      </Typography>
                    </Box>
                  </Box>
                  <Chip
                    label={activity.status}
                    size="small"
                    color={activity.status === 'completed' ? 'success' : 'warning'}
                    variant="outlined"
                  />
                </Box>
              ))}
            </Box>
          </Paper>
        </Grid>

        {/* Training Progress */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
              AI Model Training Progress
            </Typography>
            <Grid container spacing={3} alignItems="center">
              <Grid item xs={12} md={8}>
                <Box mb={2}>
                  <Typography variant="body2" color="textSecondary">
                    Epoch {trainingProgress?.epoch || 0} of {trainingProgress?.totalEpochs || 50}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={((trainingProgress?.epoch || 0) / (trainingProgress?.totalEpochs || 50)) * 100}
                    sx={{ height: 8, borderRadius: 4, mt: 1 }}
                  />
                </Box>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">
                      Accuracy
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                      {((trainingProgress?.accuracy || 0) * 100).toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">
                      Loss
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                      {(trainingProgress?.loss || 0).toFixed(4)}
                    </Typography>
                  </Grid>
                </Grid>
              </Grid>
              <Grid item xs={12} md={4}>
                <Button
                  variant="contained"
                  fullWidth
                  onClick={() => navigate('/training')}
                  sx={{ height: 48 }}
                >
                  View Details
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
