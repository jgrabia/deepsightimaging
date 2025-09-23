#!/usr/bin/env python3
"""
Download Normal/Control Breast DICOM Series from TCIA

Uses your existing tcia_utils.nbia helpers to:
- Query TCIA NBIA for BREAST series (by collection/modality)
- Heuristically filter for "normal/control" series using metadata keywords
- Download ZIP via NBIA getImage endpoint
- Extract DICOMs into dicom_download/images/<SeriesInstanceUID>
- Write a CSV manifest summarizing what was downloaded

Example:
  python download_tcia_normals.py --collection TCGA-BRCA --modality MR --max-series 50

Notes:
- TCIA does not always expose diagnosis in NBIA metadata; filtering for "normal" is heuristic.
- Use --strict-only to include only series explicitly labeled as normal/screening/negative.
"""

import os
import csv
import sys
import json
import time
import argparse
import tempfile
import zipfile
from typing import Dict, List, Tuple
import re

import requests

try:
    from tcia_utils import nbia
except Exception as e:
    print(f"❌ Failed to import tcia_utils.nbia: {e}")
    sys.exit(1)


NORMAL_KEYWORDS = [
    "normal",
    "no finding",
    "negative",
    "screening",
    "control",
    "healthy",
]

EXCLUDE_KEYWORDS = [
    "cancer",
    "malignant",
    "lesion",
    "mass",
    "tumor",
    "calcification",
    "biopsy",
]


def text_has_any(text: str, keywords: List[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in keywords)


def classify_series_meta(series_meta: Dict) -> Tuple[bool, str]:
    """Heuristically classify a series as normal/control.

    Returns (is_normal, reason)
    """
    fields = []
    for key in [
        "SeriesDescription",
        "StudyDescription",
        "ProtocolName",
        "SeriesNumber",
        "ImageComments",
    ]:
        val = series_meta.get(key)
        if isinstance(val, str):
            fields.append(val)
        elif val is not None:
            fields.append(str(val))
    blob = " | ".join(fields)

    if text_has_any(blob, EXCLUDE_KEYWORDS):
        return False, "excluded_keyword"
    if text_has_any(blob, NORMAL_KEYWORDS):
        return True, "normal_keyword"

    # Unknown/neutral metadata: treat as unknown (caller decides whether to include)
    return False, "unknown"


def ensure_dirs(base_out: str) -> Dict[str, str]:
    images_dir = os.path.join(base_out, "images")
    labels_dir = os.path.join(base_out, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    return {"images": images_dir, "labels": labels_dir}


def download_series_zip(series_uid: str, out_dir: str) -> str:
    url = (
        f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={series_uid}"
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp_path = tmp.name

    final_zip = os.path.join(out_dir, f"{series_uid}.zip")
    os.makedirs(out_dir, exist_ok=True)
    os.replace(tmp_path, final_zip)
    return final_zip


def extract_zip_to_series_dir(zip_path: str, images_root: str, series_uid: str) -> str:
    series_dir = os.path.join(images_root, series_uid)
    os.makedirs(series_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(series_dir)
    return series_dir


def write_manifest(manifest_rows: List[Dict[str, str]], manifest_path: str) -> None:
    fieldnames = [
        "series_uid",
        "patient_id",
        "study_uid",
        "collection",
        "modality",
        "is_normal",
        "reason",
        "zip_path",
        "extract_dir",
    ]
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TCIA normal/control breast DICOM series")
    parser.add_argument("--output-dir", default="dicom_download", help="Base output directory (default: dicom_download)")
    parser.add_argument("--collection", default=None, help="TCIA collection name (e.g., TCGA-BRCA, CBIS-DDSM, INbreast)")
    parser.add_argument("--modality", default="MR", choices=["MR", "MG", "CT", "PT", "CR"], help="Modality filter (default: MR)")
    parser.add_argument("--max-series", type=int, default=50, help="Max number of series to download (default: 50)")
    parser.add_argument("--strict-only", action="store_true", help="Include only series explicitly labeled as normal/control")
    parser.add_argument("--dry-run", action="store_true", help="List series that would be downloaded without downloading")
    parser.add_argument("--cbis-csv", nargs="*", help="One or more CBIS-DDSM metadata CSV files with pathology labels (used to filter benign/normal)")
    parser.add_argument("--cbis-csv-url", nargs="*", help="One or more URLs to CBIS-DDSM metadata CSVs (downloaded then used to filter)")
    parser.add_argument("--series-desc-include", nargs="*", help="Only keep series whose SeriesDescription contains any of these phrases (case-insensitive)")
    parser.add_argument("--series-desc-exclude", nargs="*", help="Drop series whose SeriesDescription contains any of these phrases (case-insensitive)")
    args = parser.parse_args()

    # Provide default CBIS-DDSM CSV URLs if requested collection is CBIS-DDSM and no CSVs are supplied
    if args.collection and str(args.collection).upper() == "CBIS-DDSM" and not args.cbis_csv and not args.cbis_csv_url:
        # Official TCIA wiki CSVs (train/test, mass/calc). These may change over time.
        CBIS_CSV_DEFAULTS = [
            "https://wiki.cancerimagingarchive.net/download/attachments/22516629/calc_case_description_train_set.csv?api=v2",
            "https://wiki.cancerimagingarchive.net/download/attachments/22516629/calc_case_description_test_set.csv?api=v2",
            "https://wiki.cancerimagingarchive.net/download/attachments/22516629/mass_case_description_train_set.csv?api=v2",
            "https://wiki.cancerimagingarchive.net/download/attachments/22516629/mass_case_description_test_set.csv?api=v2",
        ]
        print("   No CBIS CSV provided; using default TCIA CSV URLs (mass/calc train/test)…")
        args.cbis_csv_url = CBIS_CSV_DEFAULTS

    print("🏥 Querying TCIA NBIA for breast series...")
    series = nbia.getSeries(
        collection=args.collection,
        bodyPartExamined="BREAST",
        modality=args.modality,
    )
    print(f"   Found {len(series)} total series (collection={args.collection}, modality={args.modality})")

    # Sensible defaults for CBIS-DDSM: keep only full mammograms; drop ROI/cropped
    if args.collection and str(args.collection).upper() == "CBIS-DDSM":
        if not args.series_desc_include:
            args.series_desc_include = ["full mammogram images"]
        if not args.series_desc_exclude:
            args.series_desc_exclude = ["roi mask images", "cropped images"]

    def series_desc_pass(s: Dict) -> bool:
        desc = str(s.get("SeriesDescription", "")).lower()
        if args.series_desc_include:
            if not any(tok.lower() in desc for tok in args.series_desc_include):
                return False
        if args.series_desc_exclude:
            if any(tok.lower() in desc for tok in args.series_desc_exclude):
                return False
        return True

    series = [s for s in series if series_desc_pass(s)]
    print(f"   After SeriesDescription filter => {len(series)} series")

    # Optional: For CBIS-DDSM, use provided CSV(s) to keep only benign/normal PatientIDs
    benign_patient_ids: set = set()
    allowed_series_uids: set = set()
    allowed_study_uids: set = set()
    benign_core_ids: set = set()
    if args.collection and str(args.collection).upper() == "CBIS-DDSM" and (args.cbis_csv or args.cbis_csv_url):
        import io
        import pandas as pd

        def load_csv_path(path: str) -> "pd.DataFrame":
            return pd.read_csv(path)

        def load_csv_url(url: str) -> "pd.DataFrame":
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))

        dfs = []
        if args.cbis_csv:
            for p in args.cbis_csv:
                try:
                    dfs.append(load_csv_path(p))
                except Exception as e:
                    print(f"⚠️ Failed to read CSV '{p}': {e}")
        if args.cbis_csv_url:
            for u in args.cbis_csv_url:
                try:
                    dfs.append(load_csv_url(u))
                except Exception as e:
                    print(f"⚠️ Failed to fetch CSV '{u}': {e}")

        def normalize_columns(df):
            df = df.copy()
            df.columns = [c.strip().lower() for c in df.columns]
            return df

        def extract_core_id(text: str) -> str:
            """Extract CBIS-style core id like p_00038 from a string."""
            if not text:
                return ""
            m = re.search(r"p_\d{5}", str(text).lower())
            return m.group(0) if m else ""

        for df in dfs:
            df = normalize_columns(df)
            # Try typical CBIS-DDSM columns
            # pathology column
            path_col = None
            for c in ["pathology", "lbl_pathology", "label", "lesionpathology"]:
                if c in df.columns:
                    path_col = c
                    break
            # patient id column
            pid_col = None
            for c in ["patient_id", "patientid", "subject_id", "subject", "patient"]:
                if c in df.columns:
                    pid_col = c
                    break
            # optional UID columns
            series_uid_col = None
            for c in ["seriesinstanceuid", "series_uid", "seriesuid", "series"]:
                if c in df.columns:
                    series_uid_col = c
                    break
            study_uid_col = None
            for c in ["studyinstanceuid", "study_uid", "studyuid", "study"]:
                if c in df.columns:
                    study_uid_col = c
                    break
            if not path_col or not pid_col:
                print("⚠️ CSV missing expected columns (pathology/patient_id); skipping one CSV")
                continue
            for _, row in df.iterrows():
                pathology = str(row[path_col]).strip().lower()
                pid = str(row[pid_col]).strip()
                if not pid:
                    continue
                if any(k in pathology for k in ["benign", "negative", "normal", "benign without callback", "benign_no_callback"]):
                    benign_patient_ids.add(pid)
                    core = extract_core_id(pid)
                    if core:
                        benign_core_ids.add(core)
                    # also collect uids if present
                    if series_uid_col and row.get(series_uid_col):
                        allowed_series_uids.add(str(row[series_uid_col]).strip())
                    if study_uid_col and row.get(study_uid_col):
                        allowed_study_uids.add(str(row[study_uid_col]).strip())

        print(f"   CBIS-DDSM CSV filter loaded: {len(benign_patient_ids)} benign/normal PatientIDs, "
              f"{len(benign_core_ids)} core IDs, {len(allowed_series_uids)} series UIDs, {len(allowed_study_uids)} study UIDs")

    # Heuristic filtering for normals, with optional CBIS-DDSM patient filter
    candidates: List[Dict] = []
    def norm_id(x: str) -> str:
        x = (x or "").strip().lower()
        # drop non-alnum underscore for loose match
        return "".join(ch for ch in x if ch.isalnum() or ch == "_")

    norm_benign_ids = {norm_id(p) for p in benign_patient_ids}
    norm_series_uids = {norm_id(u) for u in allowed_series_uids}
    norm_study_uids = {norm_id(u) for u in allowed_study_uids}
    norm_benign_cores = {norm_id(x) for x in benign_core_ids}

    for s in series:
        # If CBIS benign CSV constraints are present, try multiple matching strategies
        if benign_patient_ids or allowed_series_uids or allowed_study_uids:
            s_series_uid = norm_id(str(s.get("SeriesInstanceUID", "")))
            s_study_uid = norm_id(str(s.get("StudyInstanceUID", "")))
            pid_candidates = []
            for k in [
                "PatientID",
                "PatientId",
                "SubjectID",
                "SubjectId",
                "Patient",
                "PatientName",
            ]:
                if k in s and s[k]:
                    pid_candidates.append(norm_id(str(s[k])))
            # also build possible core ids from NBIA PatientID strings
            core_candidates = []
            for raw in [s.get(k) for k in ["PatientID", "PatientName"] if k in s]:
                core = ""
                if raw:
                    m = re.search(r"p_\d{5}", str(raw).lower())
                    if m:
                        core = m.group(0)
                if core:
                    core_candidates.append(norm_id(core))
            # accept if any uid matches directly
            if s_series_uid and s_series_uid in norm_series_uids:
                candidates.append({**s, "_is_normal": True, "_reason": "cbis_csv_series_uid"})
                continue
            if s_study_uid and s_study_uid in norm_study_uids:
                candidates.append({**s, "_is_normal": True, "_reason": "cbis_csv_study_uid"})
                continue
            # accept if any pid candidate matches
            if any(pid in norm_benign_ids for pid in pid_candidates):
                candidates.append({**s, "_is_normal": True, "_reason": "cbis_csv_patient"})
                continue
            # accept if any derived core id matches
            if any(core in norm_benign_cores for core in core_candidates):
                candidates.append({**s, "_is_normal": True, "_reason": "cbis_csv_patient_core"})
                continue
            # no match => skip
            continue

        is_normal, reason = classify_series_meta(s)
        if args.strict_only:
            if is_normal and reason == "normal_keyword":
                candidates.append({**s, "_is_normal": True, "_reason": reason})
        else:
            # Lenient mode: include series that do NOT contain explicit disease terms.
            if reason != "excluded_keyword":
                candidates.append({**s, "_is_normal": is_normal, "_reason": reason})

    print(f"   After heuristic filtering => {len(candidates)} candidate 'normal/control' series")
    if args.dry_run:
        for i, s in enumerate(candidates[: args.max_series]):
            print(
                f"[{i+1:03d}] PatientID={s.get('PatientID')} | SeriesUID={s.get('SeriesInstanceUID')} | "
                f"Desc={s.get('SeriesDescription')} | Reason={s.get('_reason')}"
            )
        print("(dry-run) Exiting without download")
        return

    # Prepare output structure
    dirs = ensure_dirs(args.output_dir)
    zips_dir = os.path.join(args.output_dir, "zips")
    os.makedirs(zips_dir, exist_ok=True)
    manifest_rows: List[Dict[str, str]] = []

    # Download loop
    to_download = candidates[: args.max_series]
    print(f"📥 Downloading up to {len(to_download)} series...")
    for idx, s in enumerate(to_download, start=1):
        series_uid = s.get("SeriesInstanceUID")
        patient_id = s.get("PatientID", "")
        study_uid = s.get("StudyInstanceUID", "")
        try:
            print(f"   ({idx}/{len(to_download)}) Downloading SeriesInstanceUID={series_uid} PatientID={patient_id}")
            zip_path = download_series_zip(series_uid, zips_dir)
            extract_dir = extract_zip_to_series_dir(zip_path, dirs["images"], series_uid)

            manifest_rows.append(
                {
                    "series_uid": series_uid,
                    "patient_id": patient_id,
                    "study_uid": study_uid,
                    "collection": str(args.collection or ""),
                    "modality": args.modality,
                    "is_normal": str(s.get("_is_normal", False)),
                    "reason": s.get("_reason", ""),
                    "zip_path": zip_path,
                    "extract_dir": extract_dir,
                }
            )

            # Be polite to TCIA
            time.sleep(1)
        except Exception as e:
            print(f"❌ Failed series {series_uid}: {e}")

    # Write manifest
    manifest_path = os.path.join(args.output_dir, "manifest_normals.csv")
    write_manifest(manifest_rows, manifest_path)
    print("\n🎉 Done!")
    print(f"   Images: {dirs['images']}")
    print(f"   Zips:   {zips_dir}")
    print(f"   Manifest: {manifest_path}")


if __name__ == "__main__":
    main()


