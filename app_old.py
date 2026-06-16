import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
from pathlib import Path
import tempfile
import cv2

from crowd_analyzer import get_analyzer


# Page configuration
st.set_page_config(
    page_title="Crowd Density Monitoring System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="main-header">Crowd Density Monitoring System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered Crowd Counting and Density Analysis using P2PNet</div>', unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    with st.spinner("Loading P2PNet model... This may take a moment..."):
        st.session_state.analyzer = get_analyzer()
        st.session_state.model_loaded = True

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    
    st.info("""
    **Model:** P2PNet  
    **Dataset:** ShanghaiTech Part A  
    **Framework:** PyTorch  
    **Backbone:** VGG16-BN  
    **Anchor Points:** 2x2
    """)
    
    st.divider()
    
    st.header("⚙️ Settings")
    
    # Area input with tooltip
    area = st.number_input(
        "Visible Area (m²)",
        min_value=1.0,
        max_value=10000.0,
        value=300.0,
        step=10.0,
        help="Enter the approximate real-world area visible in the image in square meters."
    )
    
    st.divider()
    
    st.header("📝 Instructions")
    st.markdown("""
    1. Upload a crowd image  
    2. Enter the visible area  
    3. Click "Analyze Crowd"  
    4. View results and downloads
    """)
    
    st.divider()
    
    st.header("ℹ️ About Risk Levels")
    st.markdown("""
    - **LOW** (< 2 persons/m²): Safe  
    - **MODERATE** (2-4 persons/m²): Normal  
    - **HIGH** (4-6 persons/m²): Crowded  
    - **CRITICAL** (> 6 persons/m²): Dangerous
    """)

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Crowd Image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        
        # Display uploaded image preview
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Analyze button
    if st.session_state.uploaded_file is not None:
        if st.button("🔍 Analyze Crowd", type="primary", use_container_width=True):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(st.session_state.uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                progress_bar.progress(30)
                status_text.text("Running crowd detection...")
                
                # Run analysis
                results = st.session_state.analyzer.predict(tmp_path, area)
                
                progress_bar.progress(70)
                status_text.text("Generating visualizations...")
                
                # Generate heatmap
                heatmap = st.session_state.analyzer.generate_heatmap(
                    results['points'], 
                    results['original_image'].size[::-1]  # (height, width)
                )
                results['heatmap'] = heatmap
                
                # Store results
                st.session_state.analysis_results = results
                
                # Clean up
                os.unlink(tmp_path)
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                progress_bar.empty()
                status_text.empty()

# Display results
if st.session_state.analysis_results is not None:
    results = st.session_state.analysis_results
    
    st.divider()
    st.header("📈 Analysis Results")
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Crowd count
        count_metric = st.metric(
            label="👥 Estimated Crowd Count",
            value=f"{results['count']}",
            delta=None
        )
    
    with col2:
        # Density
        density_metric = st.metric(
            label="📏 Crowd Density",
            value=f"{results['density']:.2f} persons/m²",
            delta=None
        )
    
    with col3:
        # Risk level
        risk = results['risk']
        risk_color = results['risk']
        
        if risk_color == "LOW":
            st.success(f"⚠️ Risk Level: {risk}")
        elif risk_color == "MODERATE":
            st.info(f"⚠️ Risk Level: {risk}")
        elif risk_color == "HIGH":
            st.warning(f"⚠️ Risk Level: {risk}")
        else:
            st.error(f"⚠️ Risk Level: {risk}")
    
    # Risk alert
    st.divider()
    if risk == "LOW":
        st.success("✅ The crowd density is within safe limits.")
    elif risk == "MODERATE":
        st.info("ℹ️ The crowd density is normal. Monitor the situation.")
    elif risk == "HIGH":
        st.warning("⚠️ The crowd density is high. Consider crowd management measures.")
    else:
        st.error("🔴 CRITICAL: The crowd density is dangerous! Immediate action required.")
    
    # Gauge chart for density
    st.divider()
    st.subheader("📊 Density Gauge")
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = results['density'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Crowd Density (persons/m²)", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 8], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 2], 'color': "#green", 'name': 'Safe'},
                {'range': [2, 4], 'color': "#yellow", 'name': 'Moderate'},
                {'range': [4, 6], 'color': "#orange", 'name': 'High'},
                {'range': [6, 8], 'color': "#red", 'name': 'Critical'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': results['density']
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=14)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Image tabs
    st.divider()
    st.subheader("🖼️ Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Original Image", "Annotated Image", "Density Heatmap"])
    
    with tab1:
        st.image(results['original_image'], caption="Original Image", use_column_width=True)
    
    with tab2:
        # Convert BGR to RGB for display
        annotated_rgb = cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption=f"Annotated Image ({results['count']} people detected)", use_column_width=True)
    
    with tab3:
        # Convert BGR to RGB for display
        heatmap_rgb = cv2.cvtColor(results['heatmap'], cv2.COLOR_BGR2RGB)
        st.image(heatmap_rgb, caption="Density Heatmap", use_column_width=True)
    
    # Analytics table
    st.divider()
    st.subheader("📋 Analytics Panel")
    
    analytics_data = {
        "Metric": [
            "Image Name",
            "Image Resolution",
            "Predicted Crowd Count",
            "Visible Area",
            "Crowd Density",
            "Risk Level",
            "Inference Time"
        ],
        "Value": [
            st.session_state.uploaded_file.name if st.session_state.uploaded_file else "Unknown",
            f"{results['image_resolution'][0]} x {results['image_resolution'][1]}",
            results['count'],
            f"{area:.2f} m²",
            f"{results['density']:.2f} persons/m²",
            results['risk'],
            f"{results['inference_time']:.2f} seconds"
        ]
    }
    
    df_analytics = pd.DataFrame(analytics_data)
    st.table(df_analytics)
    
    # Download section
    st.divider()
    st.subheader("💾 Downloads")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download annotated image
        annotated_rgb = cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB)
        st.download_button(
            label="📥 Download Annotated Image",
            data=cv2.imencode('.jpg', results['annotated_image'])[1].tobytes(),
            file_name=f"annotated_{st.session_state.uploaded_file.name if st.session_state.uploaded_file else 'result'}.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
    
    with col2:
        # Download heatmap
        st.download_button(
            label="📥 Download Heatmap",
            data=cv2.imencode('.jpg', results['heatmap'])[1].tobytes(),
            file_name=f"heatmap_{st.session_state.uploaded_file.name if st.session_state.uploaded_file else 'result'}.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
    
    with col3:
        # Generate and download CSV report
        csv_data = {
            "Image Name": [st.session_state.uploaded_file.name if st.session_state.uploaded_file else "Unknown"],
            "Crowd Count": [results['count']],
            "Area (m²)": [area],
            "Density (persons/m²)": [results['density']],
            "Risk Level": [results['risk']],
            "Inference Time (s)": [results['inference_time']],
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        
        df_csv = pd.DataFrame(csv_data)
        csv = df_csv.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Report CSV",
            data=csv,
            file_name=f"crowd_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>Crowd Density Monitoring System | Powered by P2PNet & Streamlit</p>
    <p>Designed for temple queues, public festivals, and crowd management</p>
</div>
""", unsafe_allow_html=True)