import sys
import os

# Add custom TensorFlow installation path only for local Windows environment
if os.name == 'nt' and os.path.exists(r'C:\tf'):
    sys.path.insert(0, r'C:\tf')

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from PIL import Image
import io
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

MODEL_PATH = "tf-cnn-model.h5"

@st.cache_resource
def load_model_cached():
    """Load the model once and cache it"""
    model = models.load_model(MODEL_PATH, compile=False)
    return model

@st.cache_data
def evaluate_model_performance():
    """Evaluate model on MNIST test set and cache results"""
    try:
        # Load model
        model = load_model_cached()
        
        # Load MNIST test data
        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0
        
        # Make predictions on full test set
        predictions = model.predict(x_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='weighted') * 100
        recall = recall_score(y_test, y_pred, average='weighted') * 100
        f1 = f1_score(y_test, y_pred, average='weighted') * 100
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Per-class accuracy
        per_class_acc = {}
        for i in range(10):
            class_correct = cm[i, i]
            class_total = cm[i, :].sum()
            per_class_acc[i] = (class_correct / class_total) * 100
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'per_class_accuracy': per_class_acc
        }
    except Exception as e:
        st.error(f"Error evaluating model: {e}")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for prediction"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # Invert if background is dark (MNIST has white digits on black background)
    # Check if the mean pixel value is high (white background)
    if np.mean(img_gray) > 127:
        img_gray = 255 - img_gray
    
    # Resize to 28x28
    img_resized = cv2.resize(img_gray, (28, 28))
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Reshape for model input
    img_final = img_normalized.reshape(1, 28, 28, 1)
    
    return img_final, img_resized

def predict_digit(model, image):
    """Predict the digit from the preprocessed image with improved confidence calculation"""
    # Get predictions
    prediction = model.predict(image, verbose=0)[0]
    
    # Get top 3 predictions
    top_3_indices = np.argsort(prediction)[-3:][::-1]
    top_3_probs = prediction[top_3_indices]
    
    # Main prediction
    predicted_digit = top_3_indices[0]
    confidence = top_3_probs[0] * 100
    
    # Calculate prediction gap (difference between top 2 predictions)
    prediction_gap = (top_3_probs[0] - top_3_probs[1]) * 100 if len(top_3_probs) > 1 else 100
    
    # Calculate entropy (measure of uncertainty)
    # Lower entropy = more confident, higher entropy = more uncertain
    epsilon = 1e-10  # Small constant to avoid log(0)
    entropy = -np.sum(prediction * np.log(prediction + epsilon))
    max_entropy = np.log(10)  # Maximum entropy for 10 classes
    normalized_entropy = entropy / max_entropy
    uncertainty = normalized_entropy * 100
    
    return {
        'predicted_digit': predicted_digit,
        'confidence': confidence,
        'prediction_gap': prediction_gap,
        'uncertainty': uncertainty,
        'probabilities': prediction,
        'top_3_predictions': list(zip(top_3_indices, top_3_probs * 100))
    }

# Main app
def main():
    # Title and description
    st.title("🔢 Handwritten Digit Recognition")
    st.markdown("""
    ### MNIST Digit Classifier
    Upload an image of a handwritten digit (0-9) and the model will predict what digit it is!
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model_cached()
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["🎯 Predict Digits", "📊 Model Performance"])
    
    with tab2:
        display_model_metrics()
    
    with tab1:
        predict_digit_interface(model)
    
def display_model_metrics():
    """Display model performance metrics"""
    st.header("📊 Model Performance Metrics")
    st.markdown("Real-time evaluation on MNIST test dataset (10,000 images)")
    
    with st.spinner("Evaluating model performance..."):
        metrics = evaluate_model_performance()
    
    if metrics is None:
        st.error("Failed to evaluate model. Please check if the model file exists.")
        return
    
    # Display overall metrics in columns
    st.subheader("Overall Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Accuracy",
            value=f"{metrics['accuracy']:.2f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            label="🎪 Precision",
            value=f"{metrics['precision']:.2f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            label="📈 Recall",
            value=f"{metrics['recall']:.2f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            label="⚖️ F1 Score",
            value=f"{metrics['f1_score']:.2f}%",
            delta=None
        )
    
    st.markdown("---")
    
    # Create two columns for visualizations
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Confusion Matrix")
        # Create confusion matrix heatmap using plotly
        cm = metrics['confusion_matrix']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=[str(i) for i in range(10)],
            y=[str(i) for i in range(10)],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Count")
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=400,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("Per-Class Accuracy")
        # Create bar chart for per-class accuracy
        per_class_df = pd.DataFrame({
            'Digit': list(metrics['per_class_accuracy'].keys()),
            'Accuracy (%)': list(metrics['per_class_accuracy'].values())
        })
        
        fig = px.bar(
            per_class_df,
            x='Digit',
            y='Accuracy (%)',
            text='Accuracy (%)',
            color='Accuracy (%)',
            color_continuous_scale='Viridis',
            range_color=[90, 100]
        )
        
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_title="Digit",
            yaxis_title="Accuracy (%)",
            yaxis_range=[0, 105],
            showlegend=False,
            width=400,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.markdown("---")
    st.subheader("Detailed Per-Class Metrics")
    
    # Create detailed table
    detailed_data = []
    cm = metrics['confusion_matrix']
    for i in range(10):
        class_correct = cm[i, i]
        class_total = cm[i, :].sum()
        class_accuracy = metrics['per_class_accuracy'][i]
        detailed_data.append({
            'Digit': i,
            'Correct Predictions': class_correct,
            'Total Samples': class_total,
            'Accuracy (%)': f"{class_accuracy:.2f}%"
        })
    
    detailed_df = pd.DataFrame(detailed_data)
    st.dataframe(detailed_df, use_container_width=True, hide_index=True)
    
    # Summary statistics
    st.markdown("---")
    st.subheader("📈 Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    accuracies = list(metrics['per_class_accuracy'].values())
    
    with col1:
        best_digit = max(metrics['per_class_accuracy'].items(), key=lambda x: x[1])
        st.info(f"**Best Performing Digit:** {best_digit[0]} ({best_digit[1]:.2f}%)")
    
    with col2:
        worst_digit = min(metrics['per_class_accuracy'].items(), key=lambda x: x[1])
        st.warning(f"**Most Challenging Digit:** {worst_digit[0]} ({worst_digit[1]:.2f}%)")
    
    with col3:
        avg_accuracy = sum(accuracies) / len(accuracies)
        st.success(f"**Average Class Accuracy:** {avg_accuracy:.2f}%")

def predict_digit_interface(model):
    """Interface for predicting uploaded digits"""
    st.markdown("### Upload Your Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image of a handwritten digit (0-9)"
    )
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Your uploaded image", use_container_width=True)
        
        # Preprocess and predict
        with st.spinner("Processing..."):
            preprocessed_img, resized_img = preprocess_image(image)
            result = predict_digit(model, preprocessed_img)
        
        with col2:
            st.subheader("Preprocessed (28×28)")
            st.image(resized_img, caption="Grayscale 28×28", use_container_width=True)
        
        # Display prediction results
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        # Determine confidence level for styling
        confidence = result['confidence']
        prediction_gap = result['prediction_gap']
        
        # Color based on confidence and prediction gap
        if confidence >= 95 and prediction_gap >= 50:
            confidence_color = "#28a745"  # Green - Very confident
            confidence_text = "Very Confident"
        elif confidence >= 80 and prediction_gap >= 30:
            confidence_color = "#17a2b8"  # Blue - Confident
            confidence_text = "Confident"
        elif confidence >= 60:
            confidence_color = "#ffc107"  # Yellow - Moderate
            confidence_text = "Moderate Confidence"
        else:
            confidence_color = "#dc3545"  # Red - Low confidence
            confidence_text = "Low Confidence"
        
        # Show predicted digit with confidence indicator
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; border-left: 5px solid {confidence_color};'>
            <h1 style='font-size: 72px; margin: 0; color: #1f77b4;'>{result['predicted_digit']}</h1>
            <p style='font-size: 24px; color: {confidence_color}; font-weight: bold;'>{confidence_text}</p>
            <p style='font-size: 18px; color: #555;'>Confidence: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show confidence metrics
        st.markdown("### 📊 Confidence Metrics")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Confidence", f"{confidence:.2f}%")
        
        with col_b:
            st.metric("Prediction Gap", f"{prediction_gap:.2f}%", 
                     help="Difference between top 2 predictions. Higher is better.")
        
        with col_c:
            st.metric("Uncertainty", f"{result['uncertainty']:.2f}%",
                     help="Model's uncertainty level. Lower is better.")
        
        # Warning for low confidence
        if confidence < 70 or prediction_gap < 20:
            st.warning("⚠️ **Low confidence prediction!** The model is uncertain about this prediction. Consider:")
            st.markdown("""
            - The image quality might be poor
            - The digit might be ambiguous or unclear
            - The digit might be written in an unusual style
            - The image preprocessing might need adjustment
            """)
        
        # Show top 3 predictions
        st.markdown("### 🏆 Top 3 Predictions")
        for i, (digit, prob) in enumerate(result['top_3_predictions']):
            medal = ["🥇", "🥈", "🥉"][i]
            st.write(f"{medal} **Digit {digit}**: {prob:.2f}%")
        
        # Show probability distribution
        st.markdown("### 📊 Probability Distribution")
        
        # Create a more informative bar chart
        prob_df = pd.DataFrame({
            'Digit': [str(i) for i in range(10)],
            'Probability (%)': result['probabilities'] * 100
        })
        
        fig = px.bar(
            prob_df,
            x='Digit',
            y='Probability (%)',
            text='Probability (%)',
            color='Probability (%)',
            color_continuous_scale='RdYlGn',
            range_color=[0, 100]
        )
        
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_title="Digit",
            yaxis_title="Probability (%)",
            yaxis_range=[0, max(result['probabilities'] * 100) * 1.15],
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed probabilities
        with st.expander("View detailed probabilities"):
            for digit in range(10):
                prob_percent = float(result['probabilities'][digit] * 100)
                st.progress(prob_percent / 100, text=f"Digit {digit}: {prob_percent:.2f}%")
    
    else:
        # Show example images when no file is uploaded
        st.info("👆 Please upload an image to get started!")
        
        st.markdown("---")
        st.subheader("📝 Tips for best results:")
        st.markdown("""
        - Use a clear, well-lit image
        - The digit should be centered in the image
        - Both black-on-white and white-on-black work (auto-detected)
        - Avoid cluttered backgrounds
        - Single digit per image
        - Clear, bold writing works best
        """)
        
        # Show sample predictions
        st.markdown("---")
        st.subheader("🖼️ Try these sample images:")
        st.markdown("You can find sample images in the `assets/images/` folder!")
        
        # Add a note about model performance
        st.markdown("---")
        st.info("💡 **Tip:** Check the 'Model Performance' tab to see detailed accuracy metrics!")

if __name__ == "__main__":
    main()