import React from 'react';
import { Box, Typography, Paper, Stepper, Step, StepLabel, Button } from '@mui/material';

const steps = [
  'Referral & Order',
  'Scheduling & Preparation',
  'Image Acquisition',
  'Processing & Archiving',
  'Image Interpretation',
  'Reporting',
  'Communication'
];

function Workflow() {
  const [activeStep, setActiveStep] = React.useState(0);

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        DeepSight Imaging Workflow
      </Typography>
      
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Medical Imaging Workflow
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
          Complete end-to-end workflow for medical imaging from referral to communication.
        </Typography>
        
        <Stepper activeStep={activeStep} alternativeLabel>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        
        <Box sx={{ display: 'flex', flexDirection: 'row', pt: 2 }}>
          <Button
            color="inherit"
            disabled={activeStep === 0}
            onClick={handleBack}
            sx={{ mr: 1 }}
          >
            Back
          </Button>
          <Box sx={{ flex: '1 1 auto' }} />
          <Button onClick={handleNext}>
            {activeStep === steps.length - 1 ? 'Finish' : 'Next'}
          </Button>
        </Box>
      </Paper>
    </Box>
  );
}

export default Workflow;
