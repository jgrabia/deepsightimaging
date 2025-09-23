import React from 'react';
import { Box, Typography, Paper, Switch, FormControlLabel, Divider } from '@mui/material';

function Settings() {
  const [settings, setSettings] = React.useState({
    notifications: true,
    autoSave: true,
    darkMode: false,
    showAnnotations: true
  });

  const handleSettingChange = (setting: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setSettings({
      ...settings,
      [setting]: event.target.checked
    });
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Application Settings
        </Typography>
        
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={settings.notifications}
                onChange={handleSettingChange('notifications')}
              />
            }
            label="Enable Notifications"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={settings.autoSave}
                onChange={handleSettingChange('autoSave')}
              />
            }
            label="Auto-save Changes"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={settings.showAnnotations}
                onChange={handleSettingChange('showAnnotations')}
              />
            }
            label="Show Annotations by Default"
          />
          
          <Divider sx={{ my: 2 }} />
          
          <FormControlLabel
            control={
              <Switch
                checked={settings.darkMode}
                onChange={handleSettingChange('darkMode')}
              />
            }
            label="Dark Mode"
          />
        </Box>
      </Paper>
    </Box>
  );
}

export default Settings;
