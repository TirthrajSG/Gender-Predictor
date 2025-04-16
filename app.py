import pickle
import numpy as np
import pandas as pd
import streamlit as st 

with open('info.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data["model"]


def predict(h,w):
    inp = pd.DataFrame([[h,w]], columns=["Height", "Weight"])
    out = model.predict(inp)
    return out[0]

        

def create():

    st.set_page_config(
    page_title="Gender Predictor from Height and Weight", 
    page_icon="🏋️‍♂️",          
    layout="centered",                  
    initial_sidebar_state="auto"        
    )

    st.markdown("""
    <div style='text-align: right'>
        <a href="https://github.com/tirthrajsg" target="_blank">
            🐙 GitHub
        </a> &nbsp; | &nbsp;
        <a href="https://www.linkedin.com/in/tirthraj-girawale-3b2470216/" target="_blank">
            💼 LinkedIn
        </a> &nbsp; | &nbsp;
        <a href="https://www.instagram.com/tirthrajsg/" target="_blank">
            🐦 Instagram
        </a> &nbsp; | &nbsp; 
        Tirthraj Girawale
    </div>
    """, unsafe_allow_html=True)


    st.title("Predict your Gender using your Height and Weight")

    height = st.slider("Height in cm", 140.0, 200.0, 175.0)
    weight = st.slider("Weight in kg", 30.0, 120.0, 75.0)

    st.image(data['img'], use_container_width=True)

    gender = predict(height, weight)

    bmi = (weight*10000)/(height**2)


    st.header(f"You have a *{gender}* Body Type and your BMI (Body Mass Index) is *{bmi:.2f}*")
    st.write("This predictions is made by ***Decision Tree Classifier*** Model")

    



if __name__ == "__main__":
    create()
