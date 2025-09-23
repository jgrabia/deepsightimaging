import requests

NBIA_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"

def getCollections():
    url = f"{NBIA_BASE_URL}/getCollectionValues"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def getBodyPartExaminedValues():
    url = f"{NBIA_BASE_URL}/getBodyPartValues"
    r = requests.get(url)
    r.raise_for_status()
    # Filter out any dicts that don't have 'BodyPartExamined'
    return [bp for bp in r.json() if 'BodyPartExamined' in bp]

def getModalityValues():
    url = f"{NBIA_BASE_URL}/getModalityValues"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def getManufacturerValues():
    url = f"{NBIA_BASE_URL}/getManufacturerValues"
    r = requests.get(url)
    r.raise_for_status()
    # Filter out any dicts that don't have 'Manufacturer'
    return [m for m in r.json() if 'Manufacturer' in m]

def getManufacturerModelNameValues():
    url = f"{NBIA_BASE_URL}/getManufacturerModelNameValues"
    try:
        r = requests.get(url)
        r.raise_for_status()
        return [m for m in r.json() if 'ManufacturerModelName' in m]
    except Exception as e:
        print(f"Warning: Failed to fetch manufacturer model names: {e}")
        return []

def getSeries(**filters):
    url = f"{NBIA_BASE_URL}/getSeries"
    params = {}
    # Map your filter keys to the API's expected parameter names
    if 'collection' in filters:
        params['Collection'] = filters['collection']
    if 'bodyPartExamined' in filters:
        params['BodyPartExamined'] = filters['bodyPartExamined']
    if 'modality' in filters:
        params['Modality'] = filters['modality']
    if 'manufacturer' in filters:
        params['Manufacturer'] = filters['manufacturer']
    if 'manufacturerModelName' in filters:
        params['ManufacturerModelName'] = filters['manufacturerModelName']
    if 'patientID' in filters:
        params['PatientID'] = filters['patientID']
    if 'studyInstanceUID' in filters:
        params['StudyInstanceUID'] = filters['studyInstanceUID']

    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Warning: Failed to fetch series: {e}")
        return []

def getSeriesSize(seriesInstanceUID):
    url = f"{NBIA_BASE_URL}/getSeriesSize"
    params = {'SeriesInstanceUID': seriesInstanceUID}
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Warning: Failed to fetch series size: {e}")
        return [{'ObjectCount': 0}] 