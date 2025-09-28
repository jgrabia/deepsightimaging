# DeepSight Imaging AI - API Documentation

## Overview
DeepSight Imaging AI provides a secure, HIPAA-compliant cloud API for medical imaging analysis. All communication is encrypted and authenticated using bearer tokens.

## Base URL
```
https://api.deepsightimaging.ai
```

## Authentication
All API requests require authentication using a bearer token in the Authorization header:

```http
Authorization: Bearer YOUR_API_TOKEN
```

### Getting Your API Token
1. Access the DeepSight Imaging AI dashboard
2. Navigate to "☁️ Cloud DICOM API" → "👥 Customer Management"
3. Create or select your customer account
4. Go to "🔑 API Configuration" tab
5. Copy your unique API token

## API Endpoints

### 1. Upload DICOM Image

**Endpoint:** `POST /api/v1/upload`

**Description:** Upload a DICOM image for analysis

**Headers:**
```http
Authorization: Bearer YOUR_API_TOKEN
Content-Type: multipart/form-data
```

**Request Body:**
- `file`: DICOM image file (multipart form data)

**Example Request:**
```bash
curl -X POST \
  -H "Authorization: Bearer eyJjdXN0b21lcl9pZCI6Imhvc3BpdGFsXzAwMSIs..." \
  -F "file=@brain_mri.dcm" \
  https://api.deepsightimaging.ai/api/v1/upload
```

**Success Response (200 OK):**
```json
{
  "success": true,
  "image_id": "img_1234567890abcdef",
  "message": "DICOM image uploaded successfully",
  "processing_status": "pending",
  "uploaded_at": "2024-01-15T10:30:00Z"
}
```

**Error Response (400/401/500):**
```json
{
  "success": false,
  "error": "Invalid file format",
  "message": "Please upload a valid DICOM file"
}
```

### 2. Get Upload Status

**Endpoint:** `GET /api/v1/status`

**Description:** Get status of uploaded images and processing

**Headers:**
```http
Authorization: Bearer YOUR_API_TOKEN
```

**Query Parameters:**
- `image_id` (optional): Specific image ID to check status

**Example Request:**
```bash
curl -H "Authorization: Bearer YOUR_API_TOKEN" \
  https://api.deepsightimaging.ai/api/v1/status
```

**Success Response (200 OK):**
```json
{
  "success": true,
  "images": [
    {
      "image_id": "img_1234567890abcdef",
      "uploaded_at": "2024-01-15T10:30:00Z",
      "processing_status": "completed",
      "analysis_results": {
        "lesions_detected": 2,
        "confidence_score": 0.87
      }
    }
  ]
}
```

### 3. Get Analysis Results

**Endpoint:** `GET /api/v1/analysis/{image_id}`

**Description:** Get detailed analysis results for a specific image

**Headers:**
```http
Authorization: Bearer YOUR_API_TOKEN
```

**Example Request:**
```bash
curl -H "Authorization: Bearer YOUR_API_TOKEN" \
  https://api.deepsightimaging.ai/api/v1/analysis/img_1234567890abcdef
```

**Success Response (200 OK):**
```json
{
  "success": true,
  "image_id": "img_1234567890abcdef",
  "analysis_results": {
    "lesions": {
      "count": 2,
      "locations": [
        {
          "region": "Right frontal lobe",
          "coordinates": [100, 150, 200, 250],
          "confidence": 0.87,
          "size": 8.5
        }
      ]
    },
    "quality_assessment": {
      "overall_quality": 8.5,
      "artifacts": "Minimal motion artifact",
      "recommendation": "Images suitable for diagnosis"
    },
    "recommendations": [
      "Consider correlation with clinical history",
      "Follow-up imaging recommended in 3 months"
    ]
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid file or parameters |
| 401 | Unauthorized - Invalid or missing API token |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Image or resource not found |
| 413 | Payload Too Large - File exceeds 500MB limit |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |

## Rate Limits

- **Upload requests**: 100 requests per minute per customer
- **Status requests**: 1000 requests per minute per customer
- **Analysis requests**: 100 requests per minute per customer

## File Requirements

### Supported Formats
- DICOM (.dcm, .dicom)
- Maximum file size: 500MB
- Supported modalities: MR, CT, X-Ray, Ultrasound

### DICOM Requirements
- Valid DICOM header
- Pixel data present
- Patient ID (for tracking)
- Study and Series information

## Security Features

### Encryption
- **In Transit**: TLS 1.3 encryption for all API communications
- **At Rest**: AES-256 encryption for stored data
- **Token Security**: HMAC-signed tokens with expiration

### HIPAA Compliance
- End-to-end encryption
- Audit logging for all access
- Customer data isolation
- Business Associate Agreement (BAA) available

### Data Privacy
- Images are encrypted before storage
- Customer data is completely isolated
- Automatic data retention (7 years)
- Secure deletion upon request

## Integration Examples

### Python
```python
import requests

# Upload DICOM image
with open('brain_mri.dcm', 'rb') as f:
    response = requests.post(
        'https://api.deepsightimaging.ai/api/v1/upload',
        headers={'Authorization': f'Bearer {api_token}'},
        files={'file': f}
    )

result = response.json()
print(f"Image ID: {result['image_id']}")
```

### JavaScript/Node.js
```javascript
const FormData = require('form-data');
const fs = require('fs');

const form = new FormData();
form.append('file', fs.createReadStream('brain_mri.dcm'));

fetch('https://api.deepsightimaging.ai/api/v1/upload', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${apiToken}`
  },
  body: form
})
.then(response => response.json())
.then(data => console.log('Image ID:', data.image_id));
```

### cURL
```bash
# Upload image
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "file=@image.dcm" \
  https://api.deepsightimaging.ai/api/v1/upload

# Check status
curl -H "Authorization: Bearer $API_TOKEN" \
  https://api.deepsightimaging.ai/api/v1/status

# Get analysis
curl -H "Authorization: Bearer $API_TOKEN" \
  https://api.deepsightimaging.ai/api/v1/analysis/$IMAGE_ID
```

## Support

### Documentation
- Online documentation: https://docs.deepsightimaging.ai
- API reference: https://docs.deepsightimaging.ai/api

### Support Contact
- Email: support@deepsightimaging.ai
- Response time: 24 hours for standard requests
- Emergency support: Available for critical issues

### Status Page
- Service status: https://status.deepsightimaging.ai
- Incident updates and maintenance notifications

---

**Note**: This API is designed for medical imaging professionals and requires proper authentication. All usage is logged and monitored for security and compliance purposes.

