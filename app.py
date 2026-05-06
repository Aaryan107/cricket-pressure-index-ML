import streamlit as st
import pandas as pd
import pickle


trf = pickle.load(open('trf.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("🏏 IPL Win Predictor")

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select Batting Team', ['Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Punjab Kings',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Lucknow Super Giants',
    'Gujarat Titans'])
with col2:
    bowling_team = st.selectbox('Select Bowling Team', ['Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Rajasthan Royals', 'Sunrisers Hyderabad', 'Delhi Capitals', 'Punjab Kings', 'Gujarat Titans', 'Lucknow Super Giants'])

venue = st.selectbox('Select venue', [
    "M Chinnaswamy Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Arun Jaitley Stadium",
    "Eden Gardens",
    "Wankhede Stadium",
    "Sawai Mansingh Stadium",
    "Rajiv Gandhi International Stadium",
    "MA Chidambaram Stadium",
    "Dr DY Patil Sports Academy",
    "Newlands",
    "St George's Park",
    "Kingsmead",
    "SuperSport Park",
    "Buffalo Park",
    "New Wanderers Stadium",
    "De Beers Diamond Oval",
    "OUTsurance Oval",
    "Brabourne Stadium",
    "Narendra Modi Stadium",
    "Barabati Stadium",
    "Vidarbha Cricket Association Stadium, Jamtha",
    "Himachal Pradesh Cricket Association Stadium",
    "Nehru Stadium",
    "Holkar Cricket Stadium",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    "Maharashtra Cricket Association Stadium",
    "Shaheed Veer Narayan Singh International Stadium",
    "JSCA International Stadium Complex",
    "Sheikh Zayed Stadium",
    "Sharjah Cricket Stadium",
    "Dubai International Cricket Stadium",
    "Saurashtra Cricket Association Stadium",
    "Green Park",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "Barsapara Cricket Stadium, Guwahati",
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur"
])

target = st.number_input('Target Score', min_value=0)

col3, col4, col5 ,col6= st.columns(4)
with col3:
    score = st.number_input('Current Score', min_value=0)
with col4:
    balls = st.number_input('Current Ball of this Over', min_value=0, max_value=6)
with col5:
    overs=st.number_input('Current Over',min_value=1,max_value=19)
with col6:
    wickets_fallen = st.number_input('Wickets Fallen', min_value=0, max_value=10)


if st.button('Predict Win Probability'):
    
   if batting_team == bowling_team:
    st.error("Batting and Bowling teams can't be the same!")
else:
    
    runs_left = target - score
    wickets_remaining = 10 - wickets_fallen
    balls_bowled = (overs-1)*6+balls
    balls_left=120-balls_bowled
    
    
    crr = (score * 6) / balls_bowled if balls_bowled > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
    
    
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'venue': [venue],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets remaining': [wickets_remaining],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })
    
    
    input_transformed = trf.transform(input_df)
    
   
    result = model.predict_proba(input_transformed)
    
    loss = result[0][0]
    win = result[0][1]
    
    st.header(f"{batting_team} - {round(win*100)}%")
    st.header(f"{bowling_team} - {round(loss*100)}%")