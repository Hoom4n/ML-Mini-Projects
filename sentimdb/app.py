import gradio as gr
import joblib
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('wordnet')

# Load the model
model = joblib.load("sentiment_pipeline.joblib")

# Extract transformer for preprocessed output
preprocessor = model.named_steps['textpreprocessor']

# Define prediction function
def predict_sentiment(text, show_preprocessed=False):
    proba = model.predict_proba([text])[0]
    sentiment = "Positive 😀" if proba[1] >= 0.5 else "Negative 😞"
    confidence = round(max(proba) * 100, 2)

    if show_preprocessed:
        pre_text = preprocessor.transform([text])[0]
        return sentiment, f"{confidence}%", pre_text
    else:
        return sentiment, f"{confidence}%", ""

# Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=[
        gr.Textbox(lines=5, label="Type your movie review..."),
        gr.Checkbox(label="Show preprocessed text"),
    ],
    outputs=[
        gr.Label(label="Sentiment"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Preprocessed Text"),
    ],
    title="🎬 SentiMDB - Sentiment Analysis",
    description="A pipeline that classifies IMDb movie reviews with 91.67% accuracy using Logistic Regression.",
    allow_flagging="never"
)

iface.launch()
