import React from 'react';
import {
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  Alert,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Search as SearchIcon,
  Download as DownloadIcon,
  CloudDownload as CloudDownloadIcon,
  Info as InfoIcon,
  FilterList as FilterIcon
} from '@mui/icons-material';
import { useQuery, useMutation } from 'react-query';
import axios from 'axios';
import toast from 'react-hot-toast';

// Use AWS server endpoints
const USE_MOCK_DATA = false;
const API_BASE_URL = process.env.REACT_APP_AWS_API_URL || 'http://3.88.157.239:8000';

interface TCIAConfig {
  nbiaUrl: string;
  apiToken: string;
}

interface TCIAFilter {
  collection: string;
  bodyPart: string;
  modality: string;
  manufacturer: string;
  patientId: string;
  studyUid: string;
}

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

function CloudAPI() {
  const [config, setConfig] = React.useState<TCIAConfig>({
    nbiaUrl: 'https://services.cancerimagingarchive.net/services/v4/NBIA',
    apiToken: ''
  });

  const [filters, setFilters] = React.useState<TCIAFilter>({
    collection: '',
    bodyPart: '',
    modality: 'MG',
    manufacturer: '',
    patientId: '',
    studyUid: ''
  });

  const [selectedSeries, setSelectedSeries] = React.useState<string[]>([]);
  const [isSearching, setIsSearching] = React.useState(false);
  const [searchResults, setSearchResults] = React.useState<TCIASeries[]>([]);

  // Fetch available collections, body parts, etc.
  const { data: collections } = useQuery(
    'tcia-collections',
    async () => {
      if (USE_MOCK_DATA) {
        return ['Breast-Cancer-Screening-DBT', 'CBIS-DDSM', 'INbreast', 'MIAS'];
      }
      const response = await axios.get(`${API_BASE_URL}/api/tcia/collections`);
      return response.data;
    },
    {
      refetchOnWindowFocus: false,
      staleTime: 10 * 60 * 1000, // 10 minutes
    }
  );

  const { data: bodyParts } = useQuery(
    'tcia-body-parts',
    async () => {
      if (USE_MOCK_DATA) {
        return ['BREAST', 'CHEST', 'ABDOMEN', 'HEAD'];
      }
      const response = await axios.get(`${API_BASE_URL}/api/tcia/body-parts`);
      return response.data;
    },
    {
      refetchOnWindowFocus: false,
      staleTime: 10 * 60 * 1000,
    }
  );

  // Search TCIA series
  const searchMutation = useMutation(
    async (searchFilters: TCIAFilter) => {
      if (USE_MOCK_DATA) {
        // Return mock search results
        return [
          {
            SeriesInstanceUID: '1.2.3.4.5.6.7.8.9.10',
            PatientID: 'DBT-P00013',
            StudyInstanceUID: '1.2.3.4.5.6.7.8.9.11',
            SeriesDescription: 'DBT Series',
            Modality: 'MG',
            BodyPartExamined: 'BREAST',
            Manufacturer: 'HOLOGIC',
            NumberOfImages: 75,
            Collection: 'Breast-Cancer-Screening-DBT'
          },
          {
            SeriesInstanceUID: '1.2.3.4.5.6.7.8.9.12',
            PatientID: 'DBT-P00024',
            StudyInstanceUID: '1.2.3.4.5.6.7.8.9.13',
            SeriesDescription: 'DBT Series',
            Modality: 'MG',
            BodyPartExamined: 'BREAST',
            Manufacturer: 'HOLOGIC',
            NumberOfImages: 73,
            Collection: 'Breast-Cancer-Screening-DBT'
          }
        ];
      }
      const response = await axios.post(`${API_BASE_URL}/api/tcia/search`, searchFilters);
      return response.data;
    },
    {
      onSuccess: (data) => {
        setSearchResults(data);
        toast.success(`Found ${data.length} series`);
      },
      onError: (error: any) => {
        toast.error('Search failed: ' + (error?.message || error));
      }
    }
  );

  // Download selected series
  const downloadMutation = useMutation(
    async (seriesUids: string[]) => {
      const response = await axios.post(`${API_BASE_URL}/api/tcia/download`, {
        seriesUids,
        target: 'cloud' // or 'local'
      });
      return response.data;
    },
    {
      onSuccess: (data) => {
        toast.success(`Downloaded ${data.downloadedCount} series to cloud storage`);
        setSelectedSeries([]);
      },
      onError: (error: any) => {
        toast.error('Download failed: ' + (error?.message || error));
      }
    }
  );

  const handleConfigChange = (field: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setConfig({
      ...config,
      [field]: event.target.value
    });
  };

  const handleFilterChange = (field: string) => (event: any) => {
    setFilters({
      ...filters,
      [field]: event.target.value
    });
  };

  const handleSearch = () => {
    setIsSearching(true);
    searchMutation.mutate(filters, {
      onSettled: () => setIsSearching(false)
    });
  };

  const handleSeriesSelect = (seriesUid: string, checked: boolean) => {
    if (checked) {
      setSelectedSeries([...selectedSeries, seriesUid]);
    } else {
      setSelectedSeries(selectedSeries.filter(uid => uid !== seriesUid));
    }
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedSeries(searchResults.map(series => series.SeriesInstanceUID));
    } else {
      setSelectedSeries([]);
    }
  };

  const handleDownload = () => {
    if (selectedSeries.length === 0) {
      toast.error('Please select at least one series to download');
      return;
    }
    downloadMutation.mutate(selectedSeries);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        TCIA Search & Cloud Download
      </Typography>
      
      {/* Configuration Section */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">TCIA Configuration</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <TextField
                fullWidth
                label="NBIA API URL"
                value={config.nbiaUrl}
                onChange={handleConfigChange('nbiaUrl')}
                variant="outlined"
                margin="normal"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="API Token (Optional)"
                value={config.apiToken}
                onChange={handleConfigChange('apiToken')}
                variant="outlined"
                margin="normal"
                type="password"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Search Filters */}
      <Paper sx={{ p: 3, mt: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <FilterIcon sx={{ mr: 1 }} />
          <Typography variant="h6">Search Filters</Typography>
        </Box>
        
        <Grid container spacing={2}>
          <Grid item xs={12} md={6} lg={3}>
            <FormControl fullWidth>
              <InputLabel>Collection</InputLabel>
              <Select
                value={filters.collection}
                onChange={handleFilterChange('collection')}
                label="Collection"
              >
                <MenuItem value="">All Collections</MenuItem>
                {collections?.map((collection: any) => (
                  <MenuItem key={collection.name} value={collection.name}>
                    {collection.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={6} lg={3}>
            <FormControl fullWidth>
              <InputLabel>Body Part</InputLabel>
              <Select
                value={filters.bodyPart}
                onChange={handleFilterChange('bodyPart')}
                label="Body Part"
              >
                <MenuItem value="">All Body Parts</MenuItem>
                {bodyParts?.map((bodyPart: string) => (
                  <MenuItem key={bodyPart} value={bodyPart}>
                    {bodyPart}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={6} lg={3}>
            <FormControl fullWidth>
              <InputLabel>Modality</InputLabel>
              <Select
                value={filters.modality}
                onChange={handleFilterChange('modality')}
                label="Modality"
              >
                <MenuItem value="">All Modalities</MenuItem>
                <MenuItem value="MG">MG (Mammography)</MenuItem>
                <MenuItem value="CT">CT</MenuItem>
                <MenuItem value="MR">MR</MenuItem>
                <MenuItem value="DX">DX (Digital X-Ray)</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={6} lg={3}>
            <FormControl fullWidth>
              <InputLabel>Manufacturer</InputLabel>
              <Select
                value={filters.manufacturer}
                onChange={handleFilterChange('manufacturer')}
                label="Manufacturer"
              >
                <MenuItem value="">All Manufacturers</MenuItem>
                <MenuItem value="HOLOGIC">HOLOGIC</MenuItem>
                <MenuItem value="SIEMENS">SIEMENS</MenuItem>
                <MenuItem value="GE">GE</MenuItem>
                <MenuItem value="PHILIPS">PHILIPS</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Patient ID"
              value={filters.patientId}
              onChange={handleFilterChange('patientId')}
              variant="outlined"
              margin="normal"
              placeholder="e.g., DBT-P00013"
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Study UID"
              value={filters.studyUid}
              onChange={handleFilterChange('studyUid')}
              variant="outlined"
              margin="normal"
              placeholder="e.g., 1.2.826.0.1.3680043.8.498..."
            />
          </Grid>
        </Grid>
        
        <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
          <Button
            variant="contained"
            startIcon={<SearchIcon />}
            onClick={handleSearch}
            disabled={isSearching || searchMutation.isLoading}
            size="large"
          >
            {isSearching ? 'Searching...' : 'Search TCIA'}
          </Button>
          
          <Button
            variant="outlined"
            onClick={() => {
              setFilters({
                collection: '',
                bodyPart: '',
                modality: 'MG',
                manufacturer: '',
                patientId: '',
                studyUid: ''
              });
              setSearchResults([]);
              setSelectedSeries([]);
            }}
          >
            Clear Filters
          </Button>
        </Box>
        
        {searchMutation.isLoading && (
          <LinearProgress sx={{ mt: 2 }} />
        )}
      </Paper>

      {/* Search Results */}
      {searchResults.length > 0 && (
        <Paper sx={{ p: 3, mt: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              Search Results ({searchResults.length} series found)
            </Typography>
            
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={selectedSeries.length === searchResults.length}
                    indeterminate={selectedSeries.length > 0 && selectedSeries.length < searchResults.length}
                    onChange={(e) => handleSelectAll(e.target.checked)}
                  />
                }
                label="Select All"
              />
              
              <Button
                variant="contained"
                startIcon={<CloudDownloadIcon />}
                onClick={handleDownload}
                disabled={selectedSeries.length === 0 || downloadMutation.isLoading}
                color="secondary"
              >
                Download Selected ({selectedSeries.length})
              </Button>
            </Box>
          </Box>
          
          <TableContainer sx={{ maxHeight: 600 }}>
            <Table stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell padding="checkbox">Select</TableCell>
                  <TableCell>Patient ID</TableCell>
                  <TableCell>Collection</TableCell>
                  <TableCell>Series Description</TableCell>
                  <TableCell>Modality</TableCell>
                  <TableCell>Body Part</TableCell>
                  <TableCell>Manufacturer</TableCell>
                  <TableCell>Images</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {searchResults.map((series) => (
                  <TableRow key={series.SeriesInstanceUID} hover>
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={selectedSeries.includes(series.SeriesInstanceUID)}
                        onChange={(e) => handleSeriesSelect(series.SeriesInstanceUID, e.target.checked)}
                      />
                    </TableCell>
                    <TableCell>{series.PatientID}</TableCell>
                    <TableCell>
                      <Chip label={series.Collection} size="small" />
                    </TableCell>
                    <TableCell>{series.SeriesDescription || 'N/A'}</TableCell>
                    <TableCell>
                      <Chip label={series.Modality} size="small" color="primary" />
                    </TableCell>
                    <TableCell>{series.BodyPartExamined || 'N/A'}</TableCell>
                    <TableCell>{series.Manufacturer || 'N/A'}</TableCell>
                    <TableCell>{series.NumberOfImages}</TableCell>
                    <TableCell>
                      <Tooltip title="Download this series">
                        <IconButton
                          size="small"
                          onClick={() => {
                            setSelectedSeries([series.SeriesInstanceUID]);
                            downloadMutation.mutate([series.SeriesInstanceUID]);
                          }}
                        >
                          <DownloadIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="View series details">
                        <IconButton size="small">
                          <InfoIcon />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}

      {/* Download Progress */}
      {downloadMutation.isLoading && (
        <Paper sx={{ p: 3, mt: 2 }}>
          <Typography variant="h6" gutterBottom>
            Downloading to Cloud Storage...
          </Typography>
          <LinearProgress />
          <Typography variant="body2" sx={{ mt: 1 }}>
            Downloading {selectedSeries.length} series to cloud storage
          </Typography>
        </Paper>
      )}
    </Box>
  );
}

export default CloudAPI;
