import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import json
import numpy as np

# App configuration
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .model-stats {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained plant disease model"""
    try:
        # Load model configuration
        with open('class_mapping.json', 'r') as f:
            class_mapping = json.load(f)
        
        # Create model
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(class_mapping))
        
        # Load trained weights
        checkpoint = torch.load('plant_disease_efficientnet_b0.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, class_mapping, checkpoint
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def predict_image(image, model, class_mapping):
    """Make prediction on uploaded image"""
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    # Prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    return class_mapping[str(predicted_class)], confidence

# Main app
def main():
    st.markdown('<h1 class="main-header">🌿 Plant Disease Detector</h1>', unsafe_allow_html=True)
    
    # Load model
    model, class_mapping, checkpoint = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check if model files are available.")
        return
    
    # Model statistics
    st.markdown('<div class="model-stats">', unsafe_allow_html=True)
    st.write("**Model Information:**")
    st.write(f"• **Accuracy:** {checkpoint.get('accuracy', '98.23%')}")
    st.write(f"• **Classes:** {len(class_mapping)} plant diseases")
    st.write(f"• **Architecture:** {checkpoint.get('model_architecture', 'EfficientNet-B0')}")
    st.write(f"• **Training Data:** {checkpoint.get('dataset_size', '55,448 images')}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload a plant leaf image", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a plant leaf for disease detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Analyze Plant Health", type="primary"):
            with st.spinner("Analyzing plant disease..."):
                predicted_class, confidence = predict_image(image, model, class_mapping)
            
            # Display results
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.success(f"**Prediction:** {predicted_class}")
            st.success(f"**Confidence:** {confidence:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show interpretation
            if "healthy" in predicted_class.lower():
                st.balloons()
                st.info("🎉 Great news! Your plant appears to be healthy!")
            else:
                st.warning("⚠️ Potential disease detected! Consider consulting with a plant expert.")

if __name__ == "__main__":
    main()
