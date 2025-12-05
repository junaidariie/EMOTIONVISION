import streamlit as st
from PIL import Image
import io
from predict_helper_cv import predict_image

# Page configuration
st.set_page_config(
    page_title="Face Expression Recognition",
    page_icon="😊",
    layout="centered"
)

# Title and description
st.title("😊 Face Expression Recognition")
st.markdown("Upload an image to detect facial expressions")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    help="Upload a face image to detect the emotion"
)

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    # Make prediction
    with st.spinner("Analyzing expression..."):
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Get prediction
        result = predict_image(temp_path)
        
        # Clean up temp file
        import os
        os.remove(temp_path)
    
    # Display results
    with col2:
        st.subheader("Prediction Results")
        st.metric("Detected Emotion", result['emotion'])
        st.metric("Confidence", f"{result['confidence']*100:.2f}%")
        
        # Progress bar for confidence
        st.progress(result['confidence'])
    
    # Emoji mapping
    emoji_map = {
        "Angry": "😠",
        "Fear": "😨",
        "Happy": "😊",
        "Sad": "😢",
        "Surprise": "😲"
    }
    
    st.success(f"Detected: {emoji_map.get(result['emotion'], '😐')} {result['emotion']}")

else:
    st.info("👆 Please upload an image to get started")
    
    # Show example of what emotions can be detected
    st.subheader("Detectable Emotions:")
    cols = st.columns(5)
    emotions = ["😠 Angry", "😨 Fear", "😊 Happy", "😢 Sad", "😲 Surprise"]
    for col, emotion in zip(cols, emotions):
        with col:
            st.markdown(f"**{emotion}**")