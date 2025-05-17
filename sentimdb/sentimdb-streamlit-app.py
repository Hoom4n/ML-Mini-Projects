import streamlit as st
import joblib
import os

# Load the model relative to this file
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "sentiment_pipeline.joblib")
    return joblib.load(model_path)

model = load_model()

# Streamlit app layout
st.title("Sentiment Analyzer")
st.write("_Classify text as Positive or Negative based on model prediction_")

text = st.text_area("Enter your movie review here:")

if st.button("Analyze"):
    if text.strip():
        try:
            # Get probability prediction
            probs = model.predict_proba([text])[0]
            st.markdown(f"{probs}")
            #st.markdown(f"{model.predict([text])}")
            confidence = round(max(probs) * 100, 2)
            label = "Positive 😀" if probs[1] >= 0.5 else "Negative 😞"

            st.markdown(f"### Prediction: **{label}**")
            st.markdown(f"**Confidence:** {confidence}%")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter some text first.")
