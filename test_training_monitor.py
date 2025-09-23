#!/usr/bin/env python3
"""
Lightweight Training Monitor Test Script
Tests the training progress monitoring functionality with simulated data
"""

import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tempfile
import os

# Configure Streamlit
st.set_page_config(
    page_title="Training Monitor Test",
    page_icon="📊",
    layout="wide"
)

def generate_simulated_training_logs(num_epochs=50):
    """Generate simulated training logs for testing"""
    logs = []
    start_time = datetime.now() - timedelta(hours=2)
    
    # Simulate training progress
    for epoch in range(1, num_epochs + 1):
        # Simulate decreasing loss and increasing accuracy
        train_loss = 0.8 * np.exp(-epoch/20) + 0.1 + np.random.normal(0, 0.02)
        val_loss = 0.9 * np.exp(-epoch/25) + 0.15 + np.random.normal(0, 0.03)
        train_acc = 0.5 + 0.4 * (1 - np.exp(-epoch/15)) + np.random.normal(0, 0.02)
        val_acc = 0.45 + 0.4 * (1 - np.exp(-epoch/18)) + np.random.normal(0, 0.03)
        auc = 0.5 + 0.45 * (1 - np.exp(-epoch/12)) + np.random.normal(0, 0.02)
        
        # Clamp values to realistic ranges
        train_loss = max(0.01, train_loss)
        val_loss = max(0.01, val_loss)
        train_acc = max(0.1, min(0.95, train_acc))
        val_acc = max(0.1, min(0.95, val_acc))
        auc = max(0.5, min(0.99, auc))
        
        log_entry = {
            'epoch': epoch,
            'total_epochs': num_epochs,
            'timestamp': (start_time + timedelta(minutes=epoch*2.5)).strftime('%Y-%m-%d %H:%M:%S'),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'train_acc': float(train_acc),
            'val_acc': float(val_acc),
            'auc': float(auc)
        }
        logs.append(log_entry)
    
    return logs

def show_training_monitor():
    """Training progress monitor for testing"""
    st.header("📊 Training Progress Monitor Test")
    
    st.info("Testing training progress monitoring with simulated data.")
    
    # Generate or load training logs
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Training Data")
    
    with col2:
        if st.button("🧪 Generate Simulated Training Data"):
            st.session_state.simulated_logs = generate_simulated_training_logs(50)
            st.success("✅ Simulated training data generated!")
    
    # Check if we have logs
    logs = st.session_state.get('simulated_logs', [])
    
    if not logs:
        st.warning("No training data available. Click 'Generate Simulated Training Data' to test the monitor.")
        
        # Show what the monitor would look like
        st.subheader("📋 Expected Features")
        st.markdown("""
        **The training monitor will show:**
        - Real-time metrics (epoch, loss, accuracy, AUC)
        - Interactive training progress charts
        - Loss curves (training vs validation)
        - Accuracy progression
        - AUC score tracking
        - Training time visualization
        - Performance statistics
        - Auto-refresh capability
        """)
        return
    
    # Get latest log entry
    latest_log = logs[-1]
    
    # Display current status
    st.subheader("📈 Current Training Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Epoch", f"{latest_log.get('epoch', 'N/A')}/{latest_log.get('total_epochs', 'N/A')}")
    
    with col2:
        st.metric("Training Loss", f"{latest_log.get('train_loss', 0):.4f}")
    
    with col3:
        st.metric("Validation Loss", f"{latest_log.get('val_loss', 0):.4f}")
    
    with col4:
        st.metric("Validation Accuracy", f"{latest_log.get('val_acc', 0)*100:.2f}%")
    
    # Training progress visualization
    st.subheader("📊 Training Progress")
    
    # Prepare data for plotting
    epochs = [log.get('epoch', 0) for log in logs]
    train_losses = [log.get('train_loss', 0) for log in logs]
    val_losses = [log.get('val_loss', 0) for log in logs]
    train_accs = [log.get('train_acc', 0) * 100 for log in logs]
    val_accs = [log.get('val_acc', 0) * 100 for log in logs]
    auc_scores = [log.get('auc', 0) for log in logs]
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    ax1.plot(epochs, train_losses, label='Training Loss', color='blue', alpha=0.7, linewidth=2)
    ax1.plot(epochs, val_losses, label='Validation Loss', color='red', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, label='Training Accuracy', color='blue', alpha=0.7, linewidth=2)
    ax2.plot(epochs, val_accs, label='Validation Accuracy', color='red', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # AUC plot
    ax3.plot(epochs, auc_scores, label='AUC Score', color='green', alpha=0.7, linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUC Score')
    ax3.set_title('Area Under Curve (AUC)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Training time plot
    timestamps = [log.get('timestamp', '') for log in logs]
    if timestamps:
        # Convert timestamps to relative time
        start_time = datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S')
        relative_times = [(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') - start_time).total_seconds() / 60 
                        for ts in timestamps]
        ax4.plot(relative_times, epochs, color='purple', alpha=0.7, linewidth=2)
        ax4.set_xlabel('Time (minutes)')
        ax4.set_ylabel('Epoch')
        ax4.set_title('Training Progress Over Time')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Training statistics
    st.subheader("📋 Training Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Configuration:**")
        st.write(f"- Total Epochs: {latest_log.get('total_epochs', 'N/A')}")
        st.write(f"- Current Epoch: {latest_log.get('epoch', 'N/A')}")
        st.write(f"- Progress: {latest_log.get('epoch', 0)/latest_log.get('total_epochs', 1)*100:.1f}%")
        st.write(f"- Last Updated: {latest_log.get('timestamp', 'N/A')}")
    
    with col2:
        st.write("**Performance Metrics:**")
        st.write(f"- Best Training Loss: {min(train_losses):.4f}")
        st.write(f"- Best Validation Loss: {min(val_losses):.4f}")
        st.write(f"- Best Training Accuracy: {max(train_accs):.2f}%")
        st.write(f"- Best Validation Accuracy: {max(val_accs):.2f}%")
        st.write(f"- Best AUC Score: {max(auc_scores):.4f}")
    
    # Controls
    st.subheader("🎛️ Monitor Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Data"):
            st.session_state.simulated_logs = generate_simulated_training_logs(50)
            st.rerun()
    
    with col2:
        if st.button("📊 Export Logs"):
            # Create downloadable JSON
            logs_json = json.dumps(logs, indent=2)
            st.download_button(
                label="📥 Download Training Logs",
                data=logs_json,
                file_name=f"training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("🧹 Clear Data"):
            if 'simulated_logs' in st.session_state:
                del st.session_state.simulated_logs
            st.rerun()
    
    # Auto-refresh info
    st.markdown("---")
    st.info("🔄 In the real application, this page would auto-refresh every 30 seconds during training.")

def main():
    """Main function"""
    st.title("📊 Training Monitor Test Application")
    
    st.markdown("""
    This is a test version of the training progress monitor.
    It uses simulated data to demonstrate the functionality.
    """)
    
    show_training_monitor()

if __name__ == "__main__":
    main()

