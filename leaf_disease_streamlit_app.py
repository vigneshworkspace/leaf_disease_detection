import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set Streamlit page to wide layout
st.set_page_config(layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('leaf_disease_model.h5')

model = load_model()

# Define the correct tomato disease class names
class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Create a cleaner display name mapping
display_names = {
    'Tomato___Bacterial_spot': 'Bacterial Spot',
    'Tomato___Early_blight': 'Early Blight',
    'Tomato___Late_blight': 'Late Blight', 
    'Tomato___Leaf_Mold': 'Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Septoria Leaf Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spider Mites',
    'Tomato___Target_Spot': 'Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Yellow Leaf Curl Virus',
    'Tomato___Tomato_mosaic_virus': 'Mosaic Virus',
    'Tomato___healthy': 'Healthy'
}

st.title('🍅 Tomato Leaf Disease Classifier')
st.markdown('Upload an image of a tomato leaf to detect diseases')

# Create two columns: left for upload/preview, right for results (wider)
col1, col2 = st.columns([1.3, 1.3])

with col1:
    st.markdown("### 📁 Upload Image")
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        st.write(f'**File name:** {uploaded_file.name}')

with col2:
    st.markdown("### 📊 Analysis Results")
    
    if uploaded_file is not None:
        # Preprocess the image
        img = image.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        with st.spinner('Analyzing image...'):
            prediction = model.predict(img_array, verbose=0)
        
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_names[predicted_class]
        confidence = prediction[0][predicted_class] * 100
        
        # Main prediction
        st.metric(
            label="Predicted Disease", 
            value=display_names[predicted_class_name],
            delta=f"{confidence:.1f}% confidence"
        )
        
        # Status indicator
        status = "🟢 Healthy" if "healthy" in predicted_class_name.lower() else "🔴 Diseased"
        st.metric(label="Status", value=status)
        
        # Show confidence for all classes
        st.markdown('#### Confidence Scores')
        
        # Create a sorted list of predictions with class names
        predictions_with_names = [(class_names[i], prediction[0][i] * 100) for i in range(len(class_names))]
        predictions_with_names.sort(key=lambda x: x[1], reverse=True)
        
        # Display top 5 predictions
        for i, (class_name, conf) in enumerate(predictions_with_names[:5]):
            display_name = display_names[class_name]
            if i == 0:  # Top prediction
                st.markdown(f"**🥇 {display_name}:** {conf:.1f}%")
            elif i == 1:  # Second prediction
                st.markdown(f"🥈 {display_name}: {conf:.1f}%")
            elif i == 2:  # Third prediction
                st.markdown(f"🥉 {display_name}: {conf:.1f}%")
            else:
                st.markdown(f"• {display_name}: {conf:.1f}%")
        
        # Show full prediction array for debugging (collapsible)
        with st.expander("View raw prediction data"):
            st.write("Raw prediction array:")
            st.write(prediction)
            st.write("Class indices and names:")
            for i, name in enumerate(class_names):
                st.write(f"{i}: {name}")
    else:
        st.info("👈 Please upload an image on the left to see analysis results here.")
