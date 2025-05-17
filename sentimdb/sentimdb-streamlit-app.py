import streamlit as st
import joblib
import nltk

# Load NLTK data with caching
@st.cache_resource
def load_nltk_data():
    nltk.download('punkt')
    nltk.download('wordnet')

load_nltk_data()

# Load the model
try:
   model = joblib.load("sentimdb/sentiment_pipeline.joblib")
except Exception as e:
    st.write(f"Error loading model: {e}")
    st.stop()

# App title and subtitle
st.title("SentiMDB")
st.write("_An End-to-End Deployment-Ready Sentiment Analysis Pipeline based on IMDb movie reviews with 91.67% prediction accuracy_")

# Create two columns for split layout
col1, col2 = st.columns(2)

# Left panel: Input and controls
with col1:
    st.subheader("Pipeline")
    # Display pipeline steps in a box
    st.markdown("""
    <div style="border:1px solid black; padding:10px; border-radius:5px;">
    📝 Input Text → 🔧 TextPreprocessor → 📊 TF-IDF → 🤖 Logistic Regressor
    </div>
    """, unsafe_allow_html=True)
    
    # Text input
    text = st.text_area("Type your movie review...")
    
    # Checkbox for preprocessed text
    show_pre = st.checkbox("Show preprocessed text", value=True)
    
    # Analyze button
    analyze = st.button("Analyze")

# Right panel: Results
with col2:
    st.subheader("Analysis Results")
    if analyze:
        if text:
            # Make prediction
            try:
                probs = model.predict_proba([text])[0]
                confidence = round(max(probs) * 100, 2)
                sentiment = "Positive 😀" if probs[1] >= 0.5 else "Negative 😞"
                st.write(sentiment)
                st.write(text)
                st.write(f"**Prediction Confidence:** {confidence}%")
                
                # Show preprocessed text if checked
                if show_pre:
                    preprocessed = model.named_steps['textpreprocessor'].transform([text])[0]
                    st.write("**Preprocessed Text:**")
                    st.write(preprocessed)
            except Exception as e:
                st.write(f"Error during prediction: {e}")
        else:
            st.write("Please enter some text to analyze.")
    else:
        st.write("Enter some text on the left and hit 'Analyze' to reveal its mood.")
