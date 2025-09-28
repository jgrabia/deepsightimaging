# Domain Update Summary - DeepSight Imaging AI

## Overview
Updated all domain references from `deepsight.ai` to `deepsightimaging.ai` across the entire codebase.

## Files Updated

### 1. `cloud_dicom_api.py`
- **API Base URL**: `https://api.deepsightimaging.ai`
- **Upload URLs**: Updated to use new domain
- **Configuration examples**: Updated all cURL commands and API endpoints
- **Customer configuration templates**: Updated with new domain

### 2. `MRI_Integration_Setup_Guide.md`
- **Support email**: `support@deepsightimaging.ai`
- **Documentation URL**: `docs.deepsightimaging.ai`
- **Contact information**: Updated throughout guide

### 3. `complete_dicom_app.py`
- **Branding**: Updated to "DeepSight Imaging AI" throughout
- **Configuration references**: Updated all customer-facing text
- **Firewall rules**: Updated comments and documentation

### 4. `siemens_mri_workflow.py`
- **Page titles**: Updated to "DeepSight Imaging AI"
- **Report headers**: Updated MRI report branding
- **Facility names**: Updated to "DeepSight Imaging AI Medical Imaging Center"
- **Radiologist signature**: Updated to "Dr. DeepSight Imaging AI"

### 5. `deepsight_workflow.py`
- **Dashboard title**: Updated to "DeepSight Imaging AI - Medical Imaging Platform"
- **Page configuration**: Updated page title and branding
- **Class documentation**: Updated class descriptions

### 6. `dicom_server_config.py`
- **Page titles**: Updated to "DeepSight Imaging AI"
- **Configuration references**: Updated all customer-facing text
- **Firewall documentation**: Updated comments and examples

## Key Changes Made

### API Endpoints
```
Old: https://api.deepsight.ai
New: https://api.deepsightimaging.ai
```

### Customer Configuration
```bash
# Updated API configuration
API_BASE_URL=https://api.deepsightimaging.ai
API_TOKEN={api_token}
CUSTOMER_ID={customer_id}
```

### Support Information
```
Old: support@deepsight.ai
New: support@deepsightimaging.ai

Old: docs.deepsight.ai
New: docs.deepsightimaging.ai
```

### Branding Updates
- **Company Name**: DeepSight AI → DeepSight Imaging AI
- **Medical Center**: DeepSight AI Medical Imaging Center → DeepSight Imaging AI Medical Imaging Center
- **Radiologist**: Dr. DeepSight AI → Dr. DeepSight Imaging AI

## Impact on Customers

### MRI Configuration
Customers will need to update their MRI scanner configuration to use the new domain:
- **API Base URL**: Change to `https://api.deepsightimaging.ai`
- **API Token**: No change required (tokens remain valid)
- **Customer ID**: No change required

### Documentation
- All customer-facing documentation now references the new domain
- Setup guides updated with correct endpoints
- Support contact information updated

### API Compatibility
- **Backward Compatibility**: Existing API tokens remain valid
- **Endpoint Changes**: Only domain changes, API structure remains the same
- **Security**: No changes to authentication or encryption

## Next Steps

1. **DNS Configuration**: Update DNS records for `deepsightimaging.ai`
2. **SSL Certificates**: Obtain SSL certificates for new domain
3. **Customer Notification**: Notify existing customers of domain change
4. **Documentation**: Update any external documentation or marketing materials
5. **Testing**: Verify all endpoints work with new domain

## Verification Checklist

- [ ] All API endpoints updated to new domain
- [ ] Customer configuration templates updated
- [ ] Support contact information updated
- [ ] Documentation URLs updated
- [ ] Branding consistent across all files
- [ ] No remaining references to old domain

## Files That May Need External Updates

1. **Marketing Materials**: Any external documentation or websites
2. **Customer Communications**: Email templates, support documentation
3. **Third-party Integrations**: Any external systems that reference the old domain
4. **Legal Documents**: Terms of service, privacy policies, etc.

---

**Note**: This update maintains full backward compatibility for existing customers while transitioning to the new domain. API tokens and customer IDs remain unchanged.

