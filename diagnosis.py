import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Breast Cancer Diagnosis Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #FFF5F7;
        }
        .stButton>button {
            background-color: #D53F8C;
            color: white;
            border-radius: 20px;
            padding: 10px 25px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #B83280;
            border-color: #B83280;
        }
        .st-bb {
            border-bottom: 2px solid #FED7E2;
        }
        .st-eb {
            border: 2px solid #FED7E2;
        }
        h1 {
            color: #B83280;
        }
        h2 {
            color: #D53F8C;
        }
        h3 {
            color: #D53F8C;
        }
        .highlight {
            background-color: #FED7E2;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .info-box {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #FED7E2;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

def load_model():
    try:
        with open('logistic_regression_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def create_feature_input():
    st.markdown("""
    <div class="info-box">
        <h3>📋 Measurement Guidelines</h3>
        <p>Please enter the measurements from the FNA analysis. All values should be in standard medical units.</p>
    </div>
    """, unsafe_allow_html=True)
    
    features = {}
    
    # Mean features
    st.markdown("<h2>Mean Values</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        features['radius_mean'] = st.number_input('Radius (mean)', min_value=0.0, max_value=40.0, value=15.0)
        features['texture_mean'] = st.number_input('Texture (mean)', min_value=0.0, max_value=40.0, value=15.0)
        features['perimeter_mean'] = st.number_input('Perimeter (mean)', min_value=0.0, max_value=200.0, value=100.0)
        features['area_mean'] = st.number_input('Area (mean)', min_value=0.0, max_value=2500.0, value=500.0)
    
    with col2:
        features['smoothness_mean'] = st.number_input('Smoothness (mean)', min_value=0.0, max_value=1.0, value=0.1)
        features['compactness_mean'] = st.number_input('Compactness (mean)', min_value=0.0, max_value=1.0, value=0.1)
        features['concavity_mean'] = st.number_input('Concavity (mean)', min_value=0.0, max_value=1.0, value=0.1)
        features['concave_points_mean'] = st.number_input('Concave points (mean)', min_value=0.0, max_value=1.0, value=0.1)
    
    with col3:
        features['symmetry_mean'] = st.number_input('Symmetry (mean)', min_value=0.0, max_value=1.0, value=0.2)
        features['fractal_dimension_mean'] = st.number_input('Fractal dimension (mean)', min_value=0.0, max_value=1.0, value=0.1)

    # SE features
    st.markdown("<h2>Standard Error Values</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        features['radius_se'] = st.number_input('Radius (SE)', min_value=0.0, max_value=5.0, value=0.5)
        features['texture_se'] = st.number_input('Texture (SE)', min_value=0.0, max_value=5.0, value=0.5)
        features['perimeter_se'] = st.number_input('Perimeter (SE)', min_value=0.0, max_value=20.0, value=2.0)
        features['area_se'] = st.number_input('Area (SE)', min_value=0.0, max_value=250.0, value=25.0)
    
    with col2:
        features['smoothness_se'] = st.number_input('Smoothness (SE)', min_value=0.0, max_value=0.1, value=0.01)
        features['compactness_se'] = st.number_input('Compactness (SE)', min_value=0.0, max_value=0.1, value=0.01)
        features['concavity_se'] = st.number_input('Concavity (SE)', min_value=0.0, max_value=0.1, value=0.01)
        features['concave_points_se'] = st.number_input('Concave points (SE)', min_value=0.0, max_value=0.1, value=0.01)
    
    with col3:
        features['symmetry_se'] = st.number_input('Symmetry (SE)', min_value=0.0, max_value=0.1, value=0.02)
        features['fractal_dimension_se'] = st.number_input('Fractal dimension (SE)', min_value=0.0, max_value=0.1, value=0.01)

    # Worst features
    st.markdown("<h2>Worst Value Measurements</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        features['radius_worst'] = st.number_input('Radius (worst)', min_value=0.0, max_value=50.0, value=20.0)
        features['texture_worst'] = st.number_input('Texture (worst)', min_value=0.0, max_value=50.0, value=20.0)
        features['perimeter_worst'] = st.number_input('Perimeter (worst)', min_value=0.0, max_value=250.0, value=120.0)
        features['area_worst'] = st.number_input('Area (worst)', min_value=0.0, max_value=4000.0, value=750.0)
    
    with col2:
        features['smoothness_worst'] = st.number_input('Smoothness (worst)', min_value=0.0, max_value=1.0, value=0.15)
        features['compactness_worst'] = st.number_input('Compactness (worst)', min_value=0.0, max_value=1.0, value=0.15)
        features['concavity_worst'] = st.number_input('Concavity (worst)', min_value=0.0, max_value=1.0, value=0.15)
        features['concave_points_worst'] = st.number_input('Concave points (worst)', min_value=0.0, max_value=1.0, value=0.15)
    
    with col3:
        features['symmetry_worst'] = st.number_input('Symmetry (worst)', min_value=0.0, max_value=1.0, value=0.25)
        features['fractal_dimension_worst'] = st.number_input('Fractal dimension (worst)', min_value=0.0, max_value=1.0, value=0.15)
    
    return features

def main():
    # Title with pink theme
    st.markdown("""
        <div class="highlight" style="text-align: center;">
            <h1>🎀 Breast Cancer Diagnosis Assistant</h1>
            <p style="color: #B83280;">Supporting early detection and diagnosis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Information box
    st.markdown("""
        <div class="info-box">
            <h3>ℹ️ About This Tool</h3>
            <p>This diagnostic assistant uses machine learning to analyze cellular features from breast mass images.
            It provides a preliminary assessment to support medical professionals in their diagnosis process.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model()
    
    if model is not None and scaler is not None:
        # Create input fields
        features = create_feature_input()
        
        # Center the predict button
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            predict_button = st.button('Generate Diagnosis')
        
        if predict_button:
            with st.spinner('Analyzing measurements...'):
                # Create DataFrame with all features
                input_df = pd.DataFrame([features])
                
                # Make prediction
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)
                
                # Display results in a styled box
                st.markdown("""
                    <div class="highlight">
                        <h2 style="text-align: center;">Diagnosis Results</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    diagnosis = "Malignant (M)" if prediction[0] == 1 else "Benign (B)"
                    st.metric("Diagnosis", diagnosis)
                with col2:
                    confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
                    st.metric("Confidence Score", f"{confidence:.2%}")
                
                st.markdown("""
                    <div class="info-box">
                        <h3>⚠️ Important Medical Notice</h3>
                        <p>This prediction is based on a machine learning model and should be used only as a supporting tool.
                        Final diagnosis should always be made by qualified healthcare professionals considering multiple factors
                        and additional diagnostic tests.</p>
                    </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div style='text-align: center; color: #B83280; padding: 20px; margin-top: 2rem;'>
            <p>💖 Supporting breast cancer awareness and early detection</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()