import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model pipeline
model = joblib.load("clf_pipeline_model.pkl")  # Make sure this is the full pipeline

# Page configuration
st.set_page_config(page_title="🧠 Personality Predictor", layout="centered")

# Title
st.title("🧠 Personality Predictor Web App")
st.markdown("Welcome! Fill in your details to predict whether you're an **Introvert** or **Extrovert**.")

# Input form
with st.form("personality_form"):
    st.subheader("📝 Enter Your Social Behavior Data")

    time_spent = st.slider("⏳ Time Spent Alone (hours/day)", 0.0, 24.0, 2.0)
    social_events = st.slider("🎉 Social Event Attendance (per month)", 0, 30, 8)
    going_outside = st.slider("🚪 Days Going Outside (per week)", 0.0, 7.0, 5.0)
    friends = st.slider("👥 Friends Circle Size", 0, 50, 12)
    posts = st.slider("📱 Social Media Post Frequency (per week)", 0, 50, 6)

    stage_fear = st.radio("😨 Do you have Stage Fear?", ["Yes", "No"])
    drained = st.radio("😩 Do you feel Drained after Socializing?", ["Yes", "No"])

    submit = st.form_submit_button("🔮 Predict Personality")

# Prediction logic
if submit:
    # Convert categorical to expected format if needed (1/0 or "Yes"/"No")
    input_data = pd.DataFrame([{
        'Time_spent_Alone': time_spent,
        'Social_event_attendance': social_events,
        'Going_outside': going_outside,
        'Friends_circle_size': friends,
        'Post_frequency': posts,
        'Stage_fear': stage_fear,  # if model expects string
        'Drained_after_socializing': drained  # if model expects string
    }])

    # If your model needs encoded values (1/0), change to:
    # input_data['Stage_fear'] = 1 if stage_fear == "Yes" else 0
    # input_data['Drained_after_socializing'] = 1 if drained == "Yes" else 0

    try:
        prediction = model.predict(input_data)
        personality = "🧘‍♂️ Introvert" if prediction == 0 else "🎉 Extrovert"
        st.success(f"**Predicted Personality:** {personality}")
    except Exception as e:
        st.error("🚨 Prediction failed. Please check model input or retrain the pipeline.")
        st.exception(e)
