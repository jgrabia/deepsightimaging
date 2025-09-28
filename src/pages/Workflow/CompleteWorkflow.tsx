import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  Card,
  CardContent,
  CardActions,
  Chip,
  IconButton,
  Tooltip,
  Alert,
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
  Checkbox,
  Slider,
  Switch,
  TextField
} from '@mui/material';
import {
  Search as SearchIcon,
  Download as DownloadIcon,
  Visibility as VisibilityIcon,
  Psychology as PsychologyIcon,
  CloudUpload as CloudUploadIcon,
  Analytics as AnalyticsIcon,
  MedicalServices as MedicalServicesIcon,
  Dataset as DatasetIcon,
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import toast from 'react-hot-toast';

// Simulated data and functions (replace with real implementations)
const mockCollections = [
  'Breast-Cancer-Screening-DBT',
  'CBIS-DDSM',
  'INbreast',
  'MIAS'
];

const mockBodyParts = ['BREAST', 'CHEST', 'ABDOMEN'];
const mockManufacturers = ['HOLOGIC', 'SIEMENS', 'GE', 'PHILIPS'];

interface TCIASeries {
  SeriesInstanceUID: string;
  PatientID: string;
  StudyInstanceUID: string;
  SeriesDescription: string;
  Modality: string;
  BodyPartExamined: string;
  Manufacturer: string;
  NumberOfImages: number;
  Collection: string;
}

function CompleteWorkflow() {
  const [activeTab, setActiveTab] = React.useState(0);
  const [searchFilters, setSearchFilters] = React.useState({
    collection: '',
    bodyPart: '',
    modality: 'MG',
    manufacturer: ''
  });
  const [searchResults, setSearchResults] = React.useState<TCIASeries[]>([]);
  const [selectedSeries, setSelectedSeries] = React.useState<string[]>([]);
  const [isSearching, setIsSearching] = React.useState(false);
  const [uploadedFiles, setUploadedFiles] = React.useState<File[]>([]);
  const [currentImage, setCurrentImage] = React.useState<any>(null);
  const [aiAnalysis, setAiAnalysis] = React.useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = React.useState(false);

  // TCIA Search
  const handleSearch = async () => {
    setIsSearching(true);
    try {
      const response = await fetch('http://3.88.157.239:8000/api/tcia/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(searchFilters),
      });
      
      if (response.ok) {
        const results = await response.json();
        setSearchResults(results);
        toast.success(`Found ${results.length} series`);
      } else {
        throw new Error('Search failed');
      }
    } catch (error) {
      console.error('Search error:', error);
      toast.error('Failed to search TCIA database');
    } finally {
      setIsSearching(false);
    }
  };

  // File upload
  const onDrop = async (acceptedFiles: File[]) => {
    try {
      const formData = new FormData();
      acceptedFiles.forEach(file => {
        formData.append('files', file);
      });
      
      const response = await fetch('http://3.88.157.239:8000/api/files/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const result = await response.json();
        setUploadedFiles(prev => [...prev, ...acceptedFiles]);
        toast.success(`Uploaded ${acceptedFiles.length} files to AWS server`);
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Failed to upload files to AWS server');
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/dicom': ['.dcm', '.dicom'],
      'image/*': ['.jpg', '.jpeg', '.png']
    }
  });

  // AI Analysis
  const runAIAnalysis = async () => {
    if (!currentImage) {
      toast.error('Please select an image first');
      return;
    }
    
    setIsAnalyzing(true);
    try {
      // First upload the file
      const formData = new FormData();
      formData.append('files', currentImage);
      
      const uploadResponse = await fetch('http://3.88.157.239:8000/api/files/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (uploadResponse.ok) {
        const uploadResult = await uploadResponse.json();
        const uploadedFile = uploadResult.files[0];
        
        // Then run AI analysis
        const analysisResponse = await fetch('http://3.88.157.239:8000/api/ai/analyze', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            image_path: uploadedFile.path,
            model_name: 'deepedit',
            confidence_threshold: 0.5
          }),
        });
        
        if (analysisResponse.ok) {
          const analysisResult = await analysisResponse.json();
          setAiAnalysis(analysisResult);
          toast.success('AI analysis completed');
        } else {
          throw new Error('Analysis failed');
        }
      } else {
        throw new Error('File upload failed');
      }
    } catch (error) {
      console.error('AI analysis error:', error);
      toast.error('Failed to run AI analysis');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const tabs = [
    { label: 'TCIA Search', icon: <SearchIcon /> },
    { label: 'Upload Files', icon: <CloudUploadIcon /> },
    { label: 'DICOM Viewer', icon: <VisibilityIcon /> },
    { label: 'AI Analysis', icon: <PsychologyIcon /> },
    { label: 'Results', icon: <AnalyticsIcon /> }
  ];

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        🏥 DeepSight Imaging AI - Complete Workflow
      </Typography>
      
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Medical Imaging AI Workflow
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
          Complete end-to-end workflow for medical imaging analysis with AI-powered insights.
        </Typography>
        
        {/* Tab Navigation */}
        <Box sx={{ display: 'flex', gap: 1, mb: 3, flexWrap: 'wrap' }}>
          {tabs.map((tab, index) => (
            <Button
              key={index}
              variant={activeTab === index ? 'contained' : 'outlined'}
              startIcon={tab.icon}
              onClick={() => setActiveTab(index)}
              sx={{ mb: 1 }}
            >
              {tab.label}
            </Button>
          ))}
        </Box>

        {/* TCIA Search Tab */}
        {activeTab === 0 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              🔍 Search TCIA Database
            </Typography>
            
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Collection</InputLabel>
                  <Select
                    value={searchFilters.collection}
                    onChange={(e) => setSearchFilters({...searchFilters, collection: e.target.value})}
                    label="Collection"
                  >
                    <MenuItem value="">All Collections</MenuItem>
                    {mockCollections.map((collection) => (
                      <MenuItem key={collection} value={collection}>
                        {collection}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Body Part</InputLabel>
                  <Select
                    value={searchFilters.bodyPart}
                    onChange={(e) => setSearchFilters({...searchFilters, bodyPart: e.target.value})}
                    label="Body Part"
                  >
                    <MenuItem value="">All Body Parts</MenuItem>
                    {mockBodyParts.map((bodyPart) => (
                      <MenuItem key={bodyPart} value={bodyPart}>
                        {bodyPart}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Modality</InputLabel>
                  <Select
                    value={searchFilters.modality}
                    onChange={(e) => setSearchFilters({...searchFilters, modality: e.target.value})}
                    label="Modality"
                  >
                    <MenuItem value="MG">MG (Mammography)</MenuItem>
                    <MenuItem value="CT">CT</MenuItem>
                    <MenuItem value="MR">MR</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Manufacturer</InputLabel>
                  <Select
                    value={searchFilters.manufacturer}
                    onChange={(e) => setSearchFilters({...searchFilters, manufacturer: e.target.value})}
                    label="Manufacturer"
                  >
                    <MenuItem value="">All Manufacturers</MenuItem>
                    {mockManufacturers.map((manufacturer) => (
                      <MenuItem key={manufacturer} value={manufacturer}>
                        {manufacturer}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
            
            <Button
              variant="contained"
              startIcon={<SearchIcon />}
              onClick={handleSearch}
              disabled={isSearching}
              size="large"
            >
              {isSearching ? 'Searching...' : 'Search TCIA'}
            </Button>
            
            {isSearching && <LinearProgress sx={{ mt: 2 }} />}
            
            {/* Search Results */}
            {searchResults.length > 0 && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Search Results ({searchResults.length} series found)
                </Typography>
                
                <TableContainer component={Paper}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell padding="checkbox">Select</TableCell>
                        <TableCell>Patient ID</TableCell>
                        <TableCell>Collection</TableCell>
                        <TableCell>Series Description</TableCell>
                        <TableCell>Modality</TableCell>
                        <TableCell>Manufacturer</TableCell>
                        <TableCell>Images</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {searchResults.map((series) => (
                        <TableRow key={series.SeriesInstanceUID}>
                          <TableCell padding="checkbox">
                            <Checkbox
                              checked={selectedSeries.includes(series.SeriesInstanceUID)}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setSelectedSeries([...selectedSeries, series.SeriesInstanceUID]);
                                } else {
                                  setSelectedSeries(selectedSeries.filter(id => id !== series.SeriesInstanceUID));
                                }
                              }}
                            />
                          </TableCell>
                          <TableCell>{series.PatientID}</TableCell>
                          <TableCell>
                            <Chip label={series.Collection} size="small" />
                          </TableCell>
                          <TableCell>{series.SeriesDescription}</TableCell>
                          <TableCell>
                            <Chip label={series.Modality} size="small" color="primary" />
                          </TableCell>
                          <TableCell>{series.Manufacturer}</TableCell>
                          <TableCell>{series.NumberOfImages}</TableCell>
                          <TableCell>
                            <Tooltip title="View series">
                              <IconButton size="small">
                                <VisibilityIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Download series">
                              <IconButton size="small">
                                <DownloadIcon />
                              </IconButton>
                            </Tooltip>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
                
                {selectedSeries.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Button
                      variant="contained"
                      startIcon={<DownloadIcon />}
                      color="secondary"
                    >
                      Download Selected ({selectedSeries.length})
                    </Button>
                  </Box>
                )}
              </Box>
            )}
          </Box>
        )}

        {/* Upload Files Tab */}
        {activeTab === 1 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              📁 Upload DICOM Files
            </Typography>
            
            <Box
              {...getRootProps()}
              sx={{
                border: '2px dashed',
                borderColor: isDragActive ? 'primary.main' : 'grey.300',
                borderRadius: 2,
                p: 4,
                textAlign: 'center',
                cursor: 'pointer',
                backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
                mb: 3
              }}
            >
              <input {...getInputProps()} />
              <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                {isDragActive ? 'Drop files here' : 'Drag & drop DICOM files here'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                or click to select files (.dcm, .dicom, .jpg, .png)
              </Typography>
            </Box>
            
            {uploadedFiles.length > 0 && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Uploaded Files ({uploadedFiles.length})
                </Typography>
                {uploadedFiles.map((file, index) => (
                  <Card key={index} sx={{ mb: 1 }}>
                    <CardContent>
                      <Typography variant="body1">{file.name}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <Button size="small" onClick={() => setCurrentImage(file)}>
                        View
                      </Button>
                      <Button size="small" color="error">
                        Remove
                      </Button>
                    </CardActions>
                  </Card>
                ))}
              </Box>
            )}
          </Box>
        )}

        {/* DICOM Viewer Tab */}
        {activeTab === 2 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              🖼️ DICOM Viewer & Annotation Tools
            </Typography>
            
            {currentImage ? (
              <Box>
                <Paper sx={{ p: 2, mb: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Current Image: {currentImage.name}
                  </Typography>
                  
                  {/* Image Display Area */}
                  <Box
                    sx={{
                      width: '100%',
                      height: 400,
                      border: '1px solid',
                      borderColor: 'grey.300',
                      borderRadius: 1,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      backgroundColor: 'grey.100',
                      mb: 2
                    }}
                  >
                    <Typography variant="h6" color="text.secondary">
                      DICOM Image Viewer
                      <br />
                      <small>Annotation tools: Rectangle, Ellipse, Pencil, Text</small>
                    </Typography>
                  </Box>
                  
                  {/* Annotation Controls */}
                  <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                    <Button variant="outlined" startIcon={<MedicalServicesIcon />}>
                      Rectangle
                    </Button>
                    <Button variant="outlined">
                      Ellipse
                    </Button>
                    <Button variant="outlined">
                      Pencil
                    </Button>
                    <Button variant="outlined">
                      Text
                    </Button>
                  </Box>
                  
                  {/* Window/Level Controls */}
                  <Box sx={{ display: 'flex', gap: 3, alignItems: 'center' }}>
                    <Box sx={{ width: 200 }}>
                      <Typography gutterBottom>Window Level</Typography>
                      <Slider defaultValue={50} />
                    </Box>
                    <Box sx={{ width: 200 }}>
                      <Typography gutterBottom>Window Width</Typography>
                      <Slider defaultValue={400} />
                    </Box>
                    <Switch defaultChecked />
                    <Typography>Show Annotations</Typography>
                  </Box>
                </Paper>
              </Box>
            ) : (
              <Alert severity="info">
                No image selected. Upload files or search TCIA database to view images.
              </Alert>
            )}
          </Box>
        )}

        {/* AI Analysis Tab */}
        {activeTab === 3 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              🧠 AI Analysis & Inference
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      MONAI Label Server
                    </Typography>
                    <TextField
                      fullWidth
                      label="Server URL"
                      defaultValue="http://localhost:8000"
                      margin="normal"
                    />
                    <TextField
                      fullWidth
                      label="Model"
                      defaultValue="deepedit"
                      margin="normal"
                    />
                    <Button
                      variant="contained"
                      fullWidth
                      startIcon={isAnalyzing ? <StopIcon /> : <PlayArrowIcon />}
                      onClick={runAIAnalysis}
                      disabled={!currentImage}
                      sx={{ mt: 2 }}
                    >
                      {isAnalyzing ? 'Analyzing...' : 'Run AI Analysis'}
                    </Button>
                    {isAnalyzing && <LinearProgress sx={{ mt: 1 }} />}
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Analysis Results
                    </Typography>
                    {aiAnalysis ? (
                      <Box>
                        <Typography variant="body1">
                          <strong>Overall Confidence:</strong> {(aiAnalysis.confidence * 100).toFixed(1)}%
                        </Typography>
                        <Typography variant="body1" sx={{ mt: 2 }}>
                          <strong>Findings:</strong>
                        </Typography>
                        {aiAnalysis.findings.map((finding: any, index: number) => (
                          <Box key={index} sx={{ ml: 2, mt: 1 }}>
                            <Chip
                              label={`${finding.type}: ${(finding.confidence * 100).toFixed(1)}%`}
                              color={finding.confidence > 0.8 ? 'error' : 'warning'}
                              size="small"
                            />
                            <Typography variant="body2" color="text.secondary">
                              {finding.location}
                            </Typography>
                          </Box>
                        ))}
                        <Alert severity="info" sx={{ mt: 2 }}>
                          {aiAnalysis.recommendations}
                        </Alert>
                      </Box>
                    ) : (
                      <Typography color="text.secondary">
                        Run analysis to see results
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        )}

        {/* Results Tab */}
        {activeTab === 4 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              📊 Analysis Results & Reports
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <DatasetIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                      Dataset Summary
                    </Typography>
                    <Typography variant="body2">
                      • Total Series: {searchResults.length}
                    </Typography>
                    <Typography variant="body2">
                      • Uploaded Files: {uploadedFiles.length}
                    </Typography>
                    <Typography variant="body2">
                      • Analyzed Images: {aiAnalysis ? '1' : '0'}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <AnalyticsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                      AI Performance
                    </Typography>
                    <Typography variant="body2">
                      • Average Confidence: {aiAnalysis ? `${(aiAnalysis.confidence * 100).toFixed(1)}%` : 'N/A'}
                    </Typography>
                    <Typography variant="body2">
                      • Processing Time: {aiAnalysis ? '2.3s' : 'N/A'}
                    </Typography>
                    <Typography variant="body2">
                      • Model Version: deepedit v1.0
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <MedicalServicesIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                      Clinical Notes
                    </Typography>
                    <Typography variant="body2">
                      • Findings: {aiAnalysis ? aiAnalysis.findings.length : 0}
                    </Typography>
                    <Typography variant="body2">
                      • Recommendations: {aiAnalysis ? '1' : '0'}
                    </Typography>
                    <Typography variant="body2">
                      • Follow-up: Required
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
            
            <Box sx={{ mt: 3 }}>
              <Button variant="contained" startIcon={<DownloadIcon />}>
                Export Report (PDF)
              </Button>
              <Button variant="outlined" sx={{ ml: 2 }}>
                Save Session
              </Button>
            </Box>
          </Box>
        )}
      </Paper>
    </Box>
  );
}

export default CompleteWorkflow;
