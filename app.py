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
svm = pickle.load(open('cirhossis.pkl','rb'))

#load dataset
data = pd.read_csv('Cirhossis.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Cirhossis')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Cirhossis Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['SVM','Model Lain']
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

import streamlit as st
import numpy as np

def user_report():
    id = st.text_input('ID', '')
    status = st.radio('Status', ('On Treatment', 'Off Treatment'))
    drug = st.radio('Drug', ('Asclera', 'Alopurinol', 'Oral Crystallographic Medecine', 'Efferal Prozol', 'Zenpro', 'Nabumetone', 'Acetaminophen'))
    sex = st.radio('Sex', ('Male', 'Female'))
    ascites = st.radio('Ascites', ('Yes', 'No'))
    hepatomegaly = st.radio('Hepatomegaly', ('Yes', 'No'))
    spiders = st.radio('Spiders', ('Yes', 'No'))
    edema = st.radio('Edema', ('Yes', 'No'))
    n_days = st.slider('Number of Days with Drug', 0, 365, 14)
    age = st.slider('Age', 0, 100, 45)
    bilirubin = st.slider('Bilirubin', 0, 100, 32)
    cholesterol = st.slider('Cholesterol', 0, 200, 98)
    albumin = st.slider('Albumin', 0, 50, 45)
    copper = st.slider('Copper', 0, 200, 98)
    alk_phos = st.slider('Alk_Phos', 0, 1000, 68)
    sgot = st.sidebar.slider('SGOT',21,100,24)
    tryglicerides = st.sidebar.slider('Tryglicerides',0,20,1)
    platelets = st.sidebar.slider('Platelets',0,200,108)
    prothrombin = st.sidebar.slider('Prothrombin',0,140,40)
    stage = st.sidebar.slider('Stage',0,100,25)
    
    age_bins = np.linspace(0, 100, 16)
    age_labels = list(range(1, 16))
    age_group = age_labels[np.digitize(age, age_bins, right=True) - 1]

    user_report_data = {
        'ID' :id,
        'Status' :status,
        'Drug' :drug,
        'Sex' :sex,
        'Ascites' :ascites,
        'Hepatomegaly' :hepatomegaly,
        'Spiders' :spiders,
        'Edema' :edema,
        'N_Days':n_days,
        'Age':age_group,
        'Bilirubin':bilirubin,
        'Cholesterol':cholesterol,
        'Albumin':albumin,
        'Copper':copper,
        'Alk_Phos':alk_phos,
        'SGOT':sgot,
        'Tryglicerides':tryglicerides,
        'Platelets':platelets,
        'Prothrombin':prothrombin,
        'Stage':stage

       
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = svm.predict(user_data)
svc_score = accuracy_score(y_test,svm.predict(X_test))

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