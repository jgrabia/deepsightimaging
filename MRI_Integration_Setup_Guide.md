# DeepSight AI - MRI Machine Integration Setup Guide

## Overview
This guide explains how to configure your MRI machine to send images directly to DeepSight AI for real-time analysis, annotation, and reporting.

## Prerequisites
- MRI machine with DICOM capabilities
- Network connectivity between MRI and DeepSight AI server
- Administrative access to MRI scanner configuration
- DeepSight AI server running and accessible

## Step 1: Get Your API Token

### 1.1 Access DeepSight Imaging AI Dashboard
1. Launch DeepSight Imaging AI application
2. Navigate to "☁️ Cloud DICOM API" in the sidebar
3. Go to "👥 Customer Management" tab
4. Create a new customer account or select existing customer
5. Go to "🔑 API Configuration" tab

### 1.2 Generate API Token
1. Copy your unique API token from the dashboard
2. Note the API endpoints:
   - **Base URL**: `https://api.deepsightimaging.ai`
   - **Upload Endpoint**: `/api/v1/upload`
   - **Status Endpoint**: `/api/v1/status`
3. Test your token using the provided cURL example

### 1.3 Verify API Access
- API token should be active and valid
- Test connection using the "Test Connection" button
- Ensure your network can reach `https://api.deepsightimaging.ai`

## Step 2: MRI Machine Configuration

### 2.1 Access MRI Scanner Settings
1. Log into MRI scanner with administrative privileges
2. Navigate to **Network** or **DICOM** settings
3. Look for **DICOM Configuration** or **Network Services**

### 2.2 Configure API Upload
1. Find **DICOM Export** or **Network Services**
2. Configure API-based upload settings
3. Set the following parameters:

#### For All MRI Manufacturers (Cloud API):
```
API Base URL: https://api.deepsightimaging.ai
API Endpoint: /api/v1/upload
Authentication: Bearer Token
API Token: [Your unique API token]
Content Type: application/dicom
Max File Size: 500MB
Timeout: 300 seconds
```

#### Example Configuration:
```bash
# Environment Variables
export API_BASE_URL="https://api.deepsightimaging.ai"
export API_TOKEN="eyJjdXN0b21lcl9pZCI6Imhvc3BpdGFsXzAwMSIs..."
export UPLOAD_ENDPOINT="/api/v1/upload"

# cURL Example
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/dicom" \
  -F "file=@image.dcm" \
  $API_BASE_URL$UPLOAD_ENDPOINT
```

### 2.3 Configure Auto-Forwarding (Optional)
1. Navigate to **Study Forwarding** or **Auto-Route**
2. Enable automatic forwarding to DeepSight Imaging AI
3. Configure forwarding rules:
   - Forward all studies
   - Forward specific sequences only
   - Forward based on study description
4. Set forwarding destination to API endpoint with your token

### 2.4 Test Configuration
1. Send a test image using your API token
2. Verify image appears in DeepSight Imaging AI dashboard
3. Check image quality and metadata in the "📡 Image Reception" tab

## Step 3: Network Configuration

### 3.1 Firewall Settings
Configure firewall to allow HTTPS traffic to DeepSight Imaging AI:

#### Windows Firewall:
```cmd
netsh advfirewall firewall add rule name="DeepSight Imaging AI HTTPS" dir=out action=allow protocol=TCP remoteport=443
```

#### Linux iptables:
```bash
iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT
```

#### Network Equipment:
- Configure router/switch to allow HTTPS (port 443) outbound
- Ensure no network segmentation blocking HTTPS traffic
- Allow DNS resolution for `api.deepsightimaging.ai`

### 3.2 Network Security
- Use dedicated VLAN for medical imaging traffic
- Enable VPN for remote access
- Monitor network traffic for anomalies
- Regular security audits

## Step 4: Verification and Testing

### 4.1 Connection Test
1. In DeepSight AI, go to "Image Reception" tab
2. Verify server shows "🟢 DICOM Server Running"
3. Send test image from MRI scanner
4. Confirm image appears in "Recent Images" list

### 4.2 Image Quality Check
1. Open received image in DeepSight AI
2. Verify image displays correctly
3. Check metadata is complete
4. Test annotation and AI analysis features

### 4.3 Workflow Test
1. Complete end-to-end workflow:
   - Image received from MRI
   - Radiologist annotates image
   - AI analysis runs automatically
   - Report generated and reviewed

## Troubleshooting

### Common Issues:

#### 1. "Connection Failed" Error
**Symptoms**: MRI cannot connect to DeepSight AI
**Solutions**:
- Verify DeepSight AI server is running
- Check IP address and port configuration
- Test network connectivity (ping, telnet)
- Verify firewall settings

#### 2. "Images Not Received" Error
**Symptoms**: MRI shows successful send, but images don't appear in DeepSight AI
**Solutions**:
- Check DICOM server logs
- Verify AE Title matches exactly
- Test with different image types
- Check for network timeouts

#### 3. "Image Display Issues" Error
**Symptoms**: Images received but display incorrectly
**Solutions**:
- Verify transfer syntax compatibility
- Check image compression settings
- Update DICOM server configuration
- Test with uncompressed images

#### 4. "Slow Image Transfer" Error
**Symptoms**: Images take long time to transfer
**Solutions**:
- Check network bandwidth
- Optimize transfer syntax settings
- Consider compression options
- Monitor network latency

### Diagnostic Commands:

#### Test Network Connectivity:
```bash
# Test if port is open
telnet [DeepSight AI IP] 104

# Test with netcat
nc -zv [DeepSight AI IP] 104
```

#### Monitor DICOM Traffic:
```bash
# Monitor network traffic
tcpdump -i any port 104

# Check DICOM server logs
tail -f /var/log/deepsight/dicom.log
```

## Security Considerations

### HIPAA Compliance:
- Ensure all data transmission is encrypted
- Use secure network protocols
- Implement access controls
- Regular security audits
- Data backup and recovery procedures

### Network Security:
- Use dedicated medical imaging network
- Implement network segmentation
- Enable intrusion detection
- Regular security updates
- Monitor for unauthorized access

## Support and Maintenance

### Regular Maintenance:
- Monitor server performance
- Update software regularly
- Backup configurations
- Test disaster recovery procedures
- Review security logs

### Support Contacts:
- DeepSight AI Technical Support: [support@deepsightimaging.ai]
- MRI Manufacturer Support: [Check your equipment manual]
- Network Administrator: [Your IT department]

## Configuration Examples

### Complete Configuration for Siemens Skyra 3T:
```
MRI Scanner Settings:
- AE Title: SKYRA3T
- Port: 104
- Transfer Syntax: JPEG Lossless

DeepSight AI Settings:
- AE Title: DEEPSIGHT_AI
- IP: 192.168.1.100
- Port: 104
- Protocol: TCP
- Storage Commitment: Yes
```

### Network Configuration:
```
Firewall Rules:
- Allow TCP port 104 inbound
- Allow TCP port 104 outbound
- Block all other DICOM ports unless needed

Router Configuration:
- Static route to DeepSight AI server
- QoS settings for DICOM traffic
- Monitoring and logging enabled
```

## Success Criteria

Your integration is successful when:
- ✅ MRI can send images to DeepSight AI
- ✅ Images display correctly in DeepSight AI
- ✅ AI analysis runs automatically
- ✅ Reports generate successfully
- ✅ Network performance is acceptable
- ✅ Security requirements are met

## Next Steps

After successful integration:
1. Train staff on DeepSight AI workflow
2. Configure AI models for your specific needs
3. Set up automated reporting
4. Implement quality assurance procedures
5. Plan for system scaling and updates

---

**Need Help?** Contact DeepSight AI support at support@deepsightimaging.ai or refer to the online documentation at docs.deepsightimaging.ai
e not w
ty