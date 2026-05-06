# Cricket Pressure Index 🏏

A machine learning web app that quantifies batting team pressure 
in IPL run chases using real-time win probability prediction.

🔴 [Live Demo](https://cricket-pressure-index-ml.streamlit.app/) | 
📁 [Dataset](https://www.kaggle.com/datasets/dgsports/ipl-ball-by-ball-2008-to-2022)



## What it does
Predicts the **win probability** of the batting team at any ball 
of a 2nd innings chase — turning match state into a dynamic 
pressure score rather than a binary win/loss outcome.



## How it works
1. User inputs current match state (teams, venue, score, balls, wickets)
2. App computes CRR and RRR on the fly
3. Logistic Regression model returns win probability in real time



## Features engineered
- Cumulative score, runs left, balls left, wickets remaining
- Current Run Rate (CRR) and Required Run Rate (RRR)
- OneHotEncoding for batting team, bowling team, venue
- Filtered D/L affected matches for clean training data



## Dataset
- Ball-by-ball IPL data from 2008–2025 (17 seasons)
- Merged deliveries + match-level files
- Handled team renames (e.g. Deccan Chargers → Sunrisers Hyderabad)
- Standardized 20+ venues



## Model

 Algorithm -> Logistic Regression (liblinear) 
 Train/Test Split -> 80/20 
 Output -> Win probability (0–1) 
 Deployment -> Streamlit + Pickle 

---

## Tech stack
Python • Scikit-learn • Pandas • Streamlit

---

## Run locally
git clone https://github.com/Aaryan107/cricket-pressure-index-ML
cd cricket-pressure-index-ML
pip install -r requirements.txt
streamlit run app.py
