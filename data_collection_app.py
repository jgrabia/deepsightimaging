#!/usr/bin/env python3
"""
Streamlit App for Automated Training Data Collection
Provides a user-friendly interface for downloading curated breast cancer datasets
"""

import streamlit as st
import os
import json
import time
from datetime import datetime
import pandas as pd
from automated_data_collector import AutomatedDataCollector

def main():
    st.set_page_config(
        page_title="AI Training Data Collector",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 AI Training Data Collector")
    st.markdown("Automated collection of curated breast cancer datasets with expert annotations")
    
    # Initialize collector
    if 'collector' not in st.session_state:
        st.session_state.collector = AutomatedDataCollector("training_data")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Dataset Overview", "🚀 Download Datasets", "📋 Training Manifest", "⚙️ Settings"]
    )
    
    if page == "📊 Dataset Overview":
        show_dataset_overview()
    elif page == "🚀 Download Datasets":
        show_download_interface()
    elif page == "📋 Training Manifest":
        show_training_manifest()
    elif page == "⚙️ Settings":
        show_settings()

def show_dataset_overview():
    st.header("📊 Available Datasets")
    st.markdown("Curated breast cancer datasets with expert annotations and clinical metadata")
    
    collector = st.session_state.collector
    
    # Create dataset cards
    col1, col2 = st.columns(2)
    
    for i, (dataset_name, dataset_info) in enumerate(collector.curated_datasets.items()):
        with col1 if i % 2 == 0 else col2:
            with st.container():
                st.markdown(f"### {dataset_name}")
                st.markdown(f"**{dataset_info['description']}**")
                
                # Dataset metrics
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Expected Images", dataset_info['expected_count'])
                with metrics_col2:
                    st.metric("Label Quality", dataset_info['label_quality'].split(' - ')[0])
                with metrics_col3:
                    # Check if dataset is already downloaded
                    dataset_dir = os.path.join(collector.output_dir, dataset_name)
                    if os.path.exists(dataset_dir):
                        zip_files = [f for f in os.listdir(dataset_dir) if f.endswith('.zip')]
                        st.metric("Downloaded", len(zip_files))
                    else:
                        st.metric("Downloaded", 0)
                
                # Dataset details
                with st.expander("Dataset Details"):
                    st.markdown(f"**Label Quality:** {dataset_info['label_quality']}")
                    st.markdown(f"**Modality:** {dataset_info['filters'].get('Modality', 'Mixed')}")
                    st.markdown(f"**Body Part:** {dataset_info['filters'].get('BodyPartExamined', 'BREAST')}")
                    
                    # Show filters
                    st.markdown("**Search Filters:**")
                    for key, value in dataset_info['filters'].items():
                        st.markdown(f"- {key}: {value}")
                
                st.divider()

def show_download_interface():
    st.header("🚀 Download Datasets")
    
    collector = st.session_state.collector
    
    # Dataset selection
    st.subheader("Select Datasets to Download")
    
    selected_datasets = []
    for dataset_name, dataset_info in collector.curated_datasets.items():
        # Check if already downloaded
        dataset_dir = os.path.join(collector.output_dir, dataset_name)
        already_downloaded = os.path.exists(dataset_dir)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            if st.checkbox(f"{dataset_name}", key=f"cb_{dataset_name}"):
                selected_datasets.append(dataset_name)
        with col2:
            if already_downloaded:
                st.success("✅ Downloaded")
            else:
                st.info("⏳ Pending")
        with col3:
            st.metric("Expected", dataset_info['expected_count'])
    
    # Download settings
    st.subheader("Download Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        max_series = st.slider(
            "Max Series per Dataset",
            min_value=10,
            max_value=200,
            value=50,
            help="Limit downloads to avoid overwhelming the system"
        )
    
    with col2:
        include_metadata = st.checkbox(
            "Include Metadata",
            value=True,
            help="Save clinical metadata and annotations"
        )
    
    # Download button
    if st.button("🚀 Start Download", type="primary", disabled=len(selected_datasets) == 0):
        if selected_datasets:
            download_datasets(selected_datasets, max_series, include_metadata)

def download_datasets(selected_datasets, max_series, include_metadata):
    collector = st.session_state.collector
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.container()
    
    total_datasets = len(selected_datasets)
    
    with log_container:
        st.subheader("Download Progress")
        log_placeholder = st.empty()
    
    all_logs = []
    
    for i, dataset_name in enumerate(selected_datasets):
        status_text.text(f"Processing {dataset_name} ({i+1}/{total_datasets})")
        
        # Update progress
        progress = (i / total_datasets)
        progress_bar.progress(progress)
        
        # Download dataset
        try:
            success = collector.download_dataset(dataset_name, max_series, include_metadata)
            
            if success:
                all_logs.append(f"✅ Successfully downloaded {dataset_name}")
            else:
                all_logs.append(f"❌ Failed to download {dataset_name}")
                
        except Exception as e:
            all_logs.append(f"❌ Error downloading {dataset_name}: {e}")
        
        # Update log display
        log_placeholder.markdown("\n".join(all_logs[-10:]))  # Show last 10 log entries
        
        # Wait between datasets
        time.sleep(2)
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text("Download complete!")
    
    # Show summary
    st.success("🎉 Download process completed!")
    
    # Create training manifest
    with st.spinner("Creating training manifest..."):
        manifest = collector.create_training_manifest()
    
    st.json(manifest)

def show_training_manifest():
    st.header("📋 Training Manifest")
    
    collector = st.session_state.collector
    manifest_path = os.path.join(collector.output_dir, "training_manifest.json")
    
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Images", manifest.get('total_images', 0))
        with col2:
            st.metric("Datasets", len(manifest.get('datasets', {})))
        with col3:
            st.metric("Created", manifest.get('created', 'Unknown')[:10])
        with col4:
            # Calculate total size
            total_size = 0
            for dataset_name, dataset_info in manifest.get('datasets', {}).items():
                dataset_dir = os.path.join(collector.output_dir, dataset_name)
                if os.path.exists(dataset_dir):
                    for file in os.listdir(dataset_dir):
                        if file.endswith('.zip'):
                            file_path = os.path.join(dataset_dir, file)
                            total_size += os.path.getsize(file_path)
            
            st.metric("Total Size", f"{total_size / (1024**3):.1f} GB")
        
        # Dataset details
        st.subheader("Dataset Details")
        
        for dataset_name, dataset_info in manifest.get('datasets', {}).items():
            with st.expander(f"{dataset_name} ({dataset_info.get('series_count', 0)} images)"):
                st.json(dataset_info)
        
        # Download manifest
        st.subheader("Export Manifest")
        st.download_button(
            label="📥 Download Manifest JSON",
            data=json.dumps(manifest, indent=2),
            file_name="training_manifest.json",
            mime="application/json"
        )
        
    else:
        st.warning("No training manifest found. Download some datasets first!")

def show_settings():
    st.header("⚙️ Settings")
    
    collector = st.session_state.collector
    
    st.subheader("Output Directory")
    st.text_input("Training Data Directory", value=collector.output_dir, disabled=True)
    
    st.subheader("Download Log")
    if os.path.exists(collector.download_log):
        with open(collector.download_log, 'r') as f:
            log_content = f.read()
        
        st.text_area("Recent Log Entries", value=log_content[-2000:], height=300)
        
        # Download log file
        st.download_button(
            label="📥 Download Full Log",
            data=log_content,
            file_name="download_log.txt",
            mime="text/plain"
        )
    else:
        st.info("No download log found yet.")
    
    st.subheader("Reset Data")
    if st.button("🗑️ Clear All Downloaded Data", type="secondary"):
        import shutil
        if os.path.exists(collector.output_dir):
            shutil.rmtree(collector.output_dir)
            st.success("All downloaded data cleared!")
            st.rerun()

if __name__ == "__main__":
    main()





