# DeepSight Imaging AI - DICOM Bridge Setup Guide

## Overview
This guide explains how to set up the DICOM-to-API bridge for traditional MRI scanners that don't support REST API calls directly.

## Problem Solved
- **Traditional MRI scanners** (Siemens Skyra 3T, GE Discovery MR750, etc.) only support DICOM networking
- **Modern cloud APIs** use REST with bearer tokens
- **DICOM Bridge** converts between the two protocols seamlessly

## Architecture
```
MRI Scanner → DICOM Bridge → Cloud API
(Siemens/GE) → (Your Server) → (deepsightimaging.ai)
```

## Installation

### 1. System Requirements
- **Operating System**: Windows 10/11, Linux (Ubuntu 18.04+), macOS
- **Python**: 3.8 or higher
- **Network**: Internet connection for cloud API access
- **Ports**: 104 (DICOM) must be available

### 2. Install Dependencies
```bash
pip install pynetdicom requests
```

### 3. Download Bridge Software
```bash
# Download the bridge files
wget https://api.deepsightimaging.ai/downloads/dicom_bridge.py
wget https://api.deepsightimaging.ai/downloads/bridge_config.json
```

## Configuration

### 1. Get Your API Credentials
1. Access DeepSight Imaging AI dashboard
2. Go to "☁️ Cloud DICOM API" → "👥 Customer Management"
3. Select your customer account
4. Go to "🔑 API Configuration" tab
5. Copy your API token and Customer ID

### 2. Configure Bridge
Edit `bridge_config.json`:
```json
{
  "dicom_ae_title": "DEEPSIGHT_BRIDGE",
  "dicom_port": 104,
  "dicom_host": "0.0.0.0",
  "api_base_url": "https://api.deepsightimaging.ai",
  "api_token": "eyJjdXN0b21lcl9pZCI6Imhvc3BpdGFsXzAwMSIs...",
  "customer_id": "hospital_001",
  "max_retries": 3,
  "timeout": 300
}
```

### 3. Test Configuration
```bash
python dicom_bridge.py
```

Expected output:
```
DeepSight Imaging AI - DICOM to API Bridge
==================================================
Testing API connection...
✅ API connection test successful
Starting DICOM server on port 104...
AE Title: DEEPSIGHT_BRIDGE
Customer ID: hospital_001

MRI Configuration:
  AE Title: DEEPSIGHT_BRIDGE
  IP Address: [This server's IP address]
  Port: 104

Press Ctrl+C to stop the server
```

## MRI Scanner Configuration

### Siemens Skyra 3T
1. **Access Service Menu**:
   - Log in with service/administrator credentials
   - Navigate to "Service" → "Network" → "DICOM"

2. **Create DICOM Destination**:
   - Click "Add New Destination"
   - Configure:
     ```
     Destination Name: DeepSight Imaging AI
     AE Title: DEEPSIGHT_BRIDGE
     IP Address: [Bridge server IP]
     Port: 104
     Protocol: TCP
     Transfer Syntax: JPEG Lossless
     ```

3. **Enable Auto-Forward**:
   - Go to "Study Forwarding"
   - Enable forwarding to "DeepSight Imaging AI"
   - Set to forward all studies

### GE Discovery MR750
1. **Access Network Settings**:
   - Log in with administrator privileges
   - Navigate to "System" → "Network" → "DICOM"

2. **Add Remote Node**:
   - Click "Add Node"
   - Configure:
     ```
     Node Name: DeepSight Imaging AI
     AE Title: DEEPSIGHT_BRIDGE
     IP Address: [Bridge server IP]
     Port: 104
     Compression: Lossless JPEG
     ```

3. **Configure Routing**:
   - Go to "Routing Rules"
   - Add rule to send all studies to "DeepSight Imaging AI"

### Philips Ingenia
1. **Access DICOM Configuration**:
   - Navigate to "Configuration" → "DICOM"

2. **Create Remote AE**:
   - Click "Add Remote AE"
   - Configure:
     ```
     AE Title: DEEPSIGHT_BRIDGE
     IP Address: [Bridge server IP]
     Port: 104
     Transfer Syntax: JPEG Lossless
     ```

3. **Set Export Rules**:
   - Go to "Export Rules"
   - Create rule to export to "DeepSight Imaging AI"

## Network Configuration

### Firewall Rules
```bash
# Allow DICOM traffic (port 104)
iptables -A INPUT -p tcp --dport 104 -j ACCEPT

# Allow HTTPS outbound (port 443)
iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT
```

### Network Requirements
- **Bridge server** must be accessible from MRI scanner
- **Bridge server** must have internet access
- **Port 104** must be open for DICOM traffic
- **Port 443** must be open for HTTPS API calls

## Testing

### 1. Test DICOM Connection
```bash
# From MRI scanner, send test image
# Check bridge logs for receipt
tail -f dicom_bridge.log
```

### 2. Test API Upload
```bash
# Check cloud dashboard for received images
# Go to "☁️ Cloud DICOM API" → "📡 Image Reception"
```

### 3. Verify End-to-End Flow
1. **Send test image** from MRI scanner
2. **Check bridge logs** for DICOM receipt
3. **Check bridge logs** for API upload
4. **Check cloud dashboard** for image appearance
5. **Verify image quality** and metadata

## Troubleshooting

### Common Issues

#### 1. "Connection Refused" Error
**Symptoms**: MRI cannot connect to bridge
**Solutions**:
- Verify bridge server is running
- Check firewall allows port 104
- Confirm IP address is correct
- Test with telnet: `telnet [bridge_ip] 104`

#### 2. "API Upload Failed" Error
**Symptoms**: DICOM received but API upload fails
**Solutions**:
- Verify API token is correct
- Check internet connectivity
- Confirm API endpoint is accessible
- Test with curl: `curl -H "Authorization: Bearer [token]" https://api.deepsightimaging.ai/api/v1/status`

#### 3. "Invalid DICOM Format" Error
**Symptoms**: Bridge receives but cannot process DICOM
**Solutions**:
- Verify MRI is sending valid DICOM
- Check transfer syntax compatibility
- Update bridge to latest version
- Contact support with DICOM sample

### Log Analysis
```bash
# Monitor bridge logs
tail -f dicom_bridge.log

# Check for errors
grep ERROR dicom_bridge.log

# Monitor API calls
grep "API upload" dicom_bridge.log
```

## Security Considerations

### Network Security
- **Firewall**: Only allow necessary ports
- **VPN**: Use VPN for remote bridge servers
- **Updates**: Keep bridge software updated
- **Monitoring**: Monitor for unauthorized access

### Data Security
- **Encryption**: All API calls use HTTPS/TLS
- **Tokens**: API tokens are encrypted in config
- **Logs**: Logs don't contain patient data
- **Cleanup**: Temporary files are automatically deleted

## Support

### Documentation
- Bridge documentation: https://docs.deepsightimaging.ai/bridge
- API documentation: https://docs.deepsightimaging.ai/api
- Troubleshooting guide: https://docs.deepsightimaging.ai/support

### Support Contact
- Email: support@deepsightimaging.ai
- Phone: Available for enterprise customers
- Response time: 24 hours for standard requests

### Emergency Support
- Critical issues: Available 24/7 for enterprise customers
- Escalation: Contact your account manager

---

**Note**: This bridge solution works with any DICOM-compatible MRI scanner, regardless of manufacturer or model. It provides a seamless connection between traditional DICOM systems and modern cloud APIs.

