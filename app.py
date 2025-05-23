import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os

# Check if models exist
if not os.path.exists("models/revenue_model.pkl"):
    st.error("Models not found! Please run preprocess_and_train.py first.")
    st.stop()

# Load models
try:
    model = joblib.load("models/revenue_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    kmeans = joblib.load("models/clustering_model.pkl")
    feature_list = joblib.load("models/feature_list.pkl")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.title("🎬 Box Office Brains – Movie Revenue & Strategy Predictor")
st.markdown("Predict box office revenue and get marketing advice based on your movie's features.")

# Sidebar with project info
with st.sidebar:
    st.header("About")
    st.info("This app uses ML to predict movie box office revenue based on movie characteristics.")
    st.markdown("### How It Works")
    st.markdown("1. Enter your movie details")
    st.markdown("2. Get revenue prediction")
    st.markdown("3. Receive marketing strategy recommendations")

# Inputs
col1, col2 = st.columns(2)

with col1:
    budget = st.number_input("Budget ($)", 1_000_000, 300_000_000, step=5_000_000, value=50_000_000)
    runtime = st.slider("Runtime (minutes)", 60, 240, 120)
    release_date = st.date_input("Release Date", datetime.date.today())
    release_month = release_date.month
    release_year = release_date.year

with col2:
    num_genres = st.slider("Number of Genres", 1, 5, 2)
    production_companies = st.slider("Number of Production Companies", 1, 10, 3)
    keywords_count = st.slider("Number of Keywords/Tags", 5, 30, 15)
    popularity = st.slider("Expected Popularity (1-100)", 1, 100, 50) / 10  # Scale to match dataset
    vote_average = st.slider("Expected Vote Average (1-10)", 1.0, 10.0, 7.0, 0.1)
    vote_count = st.slider("Expected Vote Count", 100, 10000, 1000)

# Create feature dictionary dynamically based on feature_list
feature_dict = {
    'budget': budget,
    'runtime': runtime,
    'release_month': release_month,
    'release_year': release_year,
    'num_genres': num_genres,
    'production_companies_count': production_companies,
    'keywords_count': keywords_count,
    'popularity': popularity,
    'vote_average': vote_average,
    'vote_count': vote_count
}

# Create input DataFrame using feature_list to ensure correct order
user_input = pd.DataFrame([feature_dict])

# Add any missing columns with zeros
for feature in feature_list:
    if feature not in user_input.columns:
        user_input[feature] = 0

# Ensure correct column order
user_input = user_input[feature_list]

# Prediction button
if st.button("Predict Revenue 💰"):
    with st.spinner("Analyzing movie potential..."):
        try:
            # Prediction
            scaled_input = scaler.transform(user_input)
            revenue_prediction = model.predict(scaled_input)[0]
            cluster = kmeans.predict(scaled_input)[0]
            
            # ROI calculation
            roi = (revenue_prediction - budget) / budget * 100
            
            # Results
            st.success(f"💰 Predicted Revenue: **${int(revenue_prediction):,}**")
            
            # ROI indicator
            if roi > 200:
                st.success(f"📈 Expected ROI: {roi:.1f}% (Excellent)")
            elif roi > 100:
                st.success(f"📈 Expected ROI: {roi:.1f}% (Good)")
            elif roi > 0:
                st.warning(f"📊 Expected ROI: {roi:.1f}% (Moderate)")
            else:
                st.error(f"📉 Expected ROI: {roi:.1f}% (Loss expected)")
            
            # Cluster Strategy
            cluster_insight = {
                0: "Blockbuster: High budget + high expected return. Use multi-platform global marketing.",
                1: "Indie/Drama: Lower budget, focus on niche audiences and festivals.",
                2: "Family/Animation: Target families and deploy cross-platform marketing.",
                3: "Genre Film: Leverage genre fanbase with targeted social media campaigns."
            }
            
            st.info(f"🎯 Movie Type: Cluster {cluster} – {cluster_insight.get(cluster, 'Unknown category')}")
            
            # Marketing Tip based on release month
            def get_marketing_tip(month):
                if month in range(5, 9):
                    return "🔥 Summer Blockbuster Season – Heavy focus on YouTube, TikTok and TV ads."
                elif month in [11, 12]:
                    return "🎄 Holiday Season – Push family-friendly elements and escapism themes."
                elif month in [1, 2]:
                    return "🏆 Award Season – Position for critics and awards consideration."
                return "📱 Standard Season – Focus on social media and influencer marketing."
            
            st.markdown(f"🧠 **Marketing Recommendation**: {get_marketing_tip(release_month)}")
            
            # Advanced insights
            with st.expander("View Advanced Insights"):
                st.subheader("Performance Factors")
                
                factors = [
                    ("Budget Efficiency", f"${budget/1_000_000:.1f}M → ${revenue_prediction/1_000_000:.1f}M", 
                     "Higher budget doesn't always mean higher returns"),
                    ("Runtime Impact", f"{runtime} minutes", 
                     "Optimal runtime varies by genre - comedies do better shorter, epics longer"),
                    ("Release Timing", f"Month {release_month} (Year: {release_year})", 
                     "Summer/holiday releases typically perform better for commercial films")
                ]
                
                for title, value, description in factors:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"**{title}:**")
                    with col2:
                        st.markdown(f"{value} - *{description}*")
                        
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Show raw input data if requested
with st.expander("Show Raw Input Data"):
    st.dataframe(user_input)

# Add footer
st.markdown("---")
st.caption("Box Office Brains - Movie Revenue Prediction System")
