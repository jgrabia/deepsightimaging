## Cloud-Based DICOM Downloader & MONAI Inference Web App

### Version
- **Document Version**: 1.0
- **Last Updated**: 2025-08-09

## 1. Overview
- **Goal**: Provide a browser-accessible application to search, download, view, annotate, and run AI inference on DICOM images using cloud services.
- **Scope**: Web frontend + cloud backend; integrates TCIA (NBIA API), cloud object storage, and a cloud-hosted MONAI Label server.

## 2. Functional Requirements

### 2.1. DICOM Series Search & Download
- **Search Filters**:
  - Collection
  - Body Part Examined
  - Modality
  - Manufacturer
  - Manufacturer Model Name
  - Patient ID
  - Study Instance UID
- **Search Results**: Display as a list with checkboxes and rich metadata (Series Description, Series UID, Patient ID, Study UID, Modality, Body Part, Manufacturer, Image Count).
- **Selection**: Select one or more series for download.
- **Download Target**: Save downloaded ZIP archives to cloud object storage (e.g., S3, GCS, Azure Blob).
- **Progress**: Show download progress in the UI; display success and error notifications.

### 2.2. DICOM Image Viewing & Annotation (Web)
- **Viewer**: Display DICOM images in-browser in a resizable canvas (default rendered area ~400x400; responsive layout).
- **Annotation Tools**:
  - Rectangle
  - Ellipse
  - Pencil (freehand)
  - Text labels
- **Interactions**: Select, move, resize, and delete annotations.
- **Persistence**: Annotations are non-persistent by default; optional future ability to save/load to cloud storage.

### 2.3. MONAI Label Inference (Cloud)
- **Invocation**: User selects a DICOM image to send to the MONAI Label server.
- **Endpoint**: Configurable server URL (cloud endpoint), default model `deepedit` (configurable per deployment).
- **Response Handling**: Display returned JSON; optional overlay rendering for segmentation masks in future.
- **Errors**: Surface server and network errors clearly in the UI.

### 2.4. Cloud Integration & Web Access
- **Deployment**: Application runs in the cloud and is accessible via web browser.
- **Storage**: Use cloud object storage for DICOM files, downloads, and optional artifacts.
- **Networking**: MONAI Label server accessible over private VPC or securely exposed public endpoint.
- **Config**: All endpoints and credentials are environment-configurable.

### 2.5. Authentication and Authorization (Configurable)
- **Auth**: Optional user login via cloud-native auth (e.g., Cognito, Firebase Auth, Azure AD) or basic auth.
- **RBAC**: Optional roles for viewer vs. admin; can be deferred.

## 3. Non-Functional Requirements

### 3.1. Usability
- **UI**: Intuitive, responsive, and performant on modern desktop browsers; basic mobile support for viewing.
- **Feedback**: Clear progress indicators, error handling, and confirmations.

### 3.2. Portability
- **Cloud-Agnostic**: Deployable on AWS, GCP, Azure, or similar.
- **Containers**: Dockerized services; optional Kubernetes (EKS/GKE/AKS) for scaling.

### 3.3. Performance
- **Latency**: Search and list responses under 3 seconds for typical queries; downloads dependent on network.
- **Throughput**: Support concurrent downloads and inference requests (tune via autoscaling if on Kubernetes).

### 3.4. Reliability
- **Resilience**: Handle API/network failures with retries and user-visible errors.
- **Data Integrity**: Verify download completeness (e.g., content-length or checksum when available).

### 3.5. Security
- **Transport**: Enforce HTTPS for all endpoints.
- **Data Access**: Use scoped credentials for storage; avoid embedding secrets in client.
- **Compliance**: For PHI scenarios, require HIPAA-eligible services and proper BAAs (deployment-time decision).

### 3.6. Observability
- **Logging**: Structured logs for backend services; client error telemetry (non-PII).
- **Metrics**: Basic metrics for requests, errors, latency; optional tracing.

## 4. Technology Stack (Recommended)

### 4.1. Frontend (Web)
- **Framework**: React (or Vue/Angular)
- **UI**: Component library (e.g., MUI/Chakra) and HTML5 Canvas/WebGL for annotations
- **Build/Host**: Static hosting (S3+CloudFront, GCS+Cloud CDN, Azure Static Web Apps) or served by backend

### 4.2. Backend
- **API**: Python FastAPI (or Flask/Django) for NBIA proxying, storage orchestration, and inference triggers
- **MONAI**: MONAI Label server in a separate service/container with GPU
- **Storage SDKs**: Cloud provider SDK (boto3/google-cloud-storage/azure-storage-blob)

### 4.3. Infrastructure
- **Compute**: Managed Kubernetes (EKS/GKE/AKS) or VM scale sets; GPU nodes for MONAI Label
- **Storage**: S3/GCS/Azure Blob for objects; optional database (Postgres) for metadata and annotations
- **Auth**: Cognito / Firebase Auth / Azure AD (optional)

## 5. Cloud Architecture (High-Level)
- **Frontend**: Deployed behind CDN with HTTPS.
- **Backend API**: Stateless container(s) handling NBIA queries, orchestrating downloads to object storage, and relaying inference requests.
- **MONAI Label**: GPU-enabled service accessible to backend via private networking.
- **Object Storage**: Central bucket/container for DICOM ZIPs and artifacts.

## 6. Optional/Future Features
- **Annotation Persistence**: Save/load annotation JSON to storage; export as DICOM-SEG/RTSTRUCT.
- **Overlay Rendering**: Visualize segmentation masks as overlays with opacity controls.
- **Batch Inference**: Queue multiple images/series for inference with status tracking.
- **PACS/FHIR Integration**: Connect to PACS or FHIR servers.
- **Multi-User Collaboration**: Share studies, comments, and annotation sessions.

## 7. Out of Scope (Initial Release)
- **On-Prem Only Deployments** (cloud first)
- **Advanced DICOM Editing/De-ID**
- **Production hardening of MONAI Label** (deployment responsibility and SOPs are separate)

## 8. User Stories
- **Researcher**: Search and download DICOM series from TCIA to cloud storage and review in browser.
- **Clinician**: Open a DICOM image, annotate regions of interest, and view AI-generated segmentations.
- **Developer**: Trigger MONAI inference on selected images using a cloud GPU service for testing.

## 9. Acceptance Criteria
- **End-to-End Flow**: Search → select → download to cloud → view in browser → annotate → run inference → view result.
- **Configurable**: Server URLs, models, and storage locations configurable via environment.
- **Stability**: No unhandled errors during typical workflows; clear error messages on failures.

## 10. Configuration & Environment
- **Env Vars**:
  - NBIA base URL and API keys (if required)
  - Cloud storage bucket/container name and region
  - MONAI Label server URL and default model
  - Auth provider configuration (optional)
- **Secrets**: Managed via cloud secrets manager (SSM/Secrets Manager/Secret Manager/Key Vault).

## 11. Deployment Targets (Examples)
- **AWS**: Frontend (S3+CloudFront), Backend (ECS/EKS/Fargate), MONAI (GPU EC2 or EKS nodegroup), Storage (S3), Auth (Cognito)
- **GCP**: Frontend (Cloud Storage+Cloud CDN), Backend (GKE/Cloud Run with CPU), MONAI (GKE GPU node), Storage (GCS), Auth (Identity Platform)
- **Azure**: Frontend (Static Web Apps/Blob+CDN), Backend (AKS/App Service), MONAI (AKS GPU), Storage (Blob), Auth (Azure AD B2C)

## 12. Assumptions
- **Data**: TCIA data is non-PHI; if PHI data is used later, additional compliance steps are required.
- **GPU**: Inference requires at least one GPU-enabled node for MONAI Label.
- **Scale**: Initial usage is small team testing; manual scaling acceptable at first.

## 13. Risks & Mitigations
- **GPU Cost/Availability**: Use spot/preemptible GPUs for dev; autoscale down when idle.
- **API Rate Limits/Changes**: Add retries and monitor NBIA API usage; cache non-sensitive metadata.
- **Network Latency**: Co-locate services in the same region; use private networking where possible.

## 14. Glossary
- **NBIA**: National Biomedical Imaging Archive (TCIA API)
- **MONAI Label**: Deep learning toolkit server for medical image labeling and inference
- **DICOM**: Digital Imaging and Communications in Medicine 