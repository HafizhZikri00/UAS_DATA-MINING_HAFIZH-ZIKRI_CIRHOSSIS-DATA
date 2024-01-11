import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

#import model 
rfc = pickle.load(open('cirhossis.pkl','rb'))

#load dataset
data = pd.read_csv('Cirhossis.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Cirhossiss')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Diabetes Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['Rfc','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset PIMA Indian</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#train test split
X = data.drop('Stage',axis=1)
y = data['Stage']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    id = st.sidebar.slider('ID',0,20,1)
    n_days = st.sidebar.slider('N_Days',0,200,108)
    age = st.sidebar.slider('Age',0,100,25)
    bilirubin = st.sidebar.slider('Bilirubin',0,200,108)
    cholesterol = st.sidebar.slider('Cholesterol',0,140,40)
    albumin = st.sidebar.slider('Albumin',0,140,40)
    copper = st.sidebar.slider('Copper',0,100,25)
    alk_Phos = st.sidebar.slider('Alk_Phos',0,1000,120)
    sgot = st.sidebar.slider('SGOT',0,80,25)
    tryglicerides = st.sidebar.slider('Tryglicerides', 0.05,2.5,0.45)
    platelets = st.sidebar.slider('Platelets',21,100,24)
    prothrombin = st.sidebar.slider('Prothrombin', 0.05,2.5,0.45)
    
    user_report_data = {
        'ID':id,
        'N_Days':n_days,

        'Age':age,
        'Bilirubin':bilirubin,
        'Cholesterol':cholesterol,
        'Albumin':albumin,
        'Copper':copper,
        'Alk_Phos':alk_Phos,
        'SGOT':sgot,
        'Tryglicerides':tryglicerides,
        'Platelets':platelets,
        'Prothrombin':prothrombin
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = rfc.predict(user_data)
svc_score = accuracy_score(y_test,rfc.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu Aman'
else:
    output ='Kamu terkena diabetes'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(svc_score*100)+'%')