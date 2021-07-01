

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
#from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression

#cols_to_use = [['gender', ['age'], ['hypertension'], ['heart_disease'], ['work_type'],['avg_glucose_level'],
#['']]]
df = pd.read_csv('stroke_data.csv')



#imputation

DT_bmi_pipe = Pipeline( steps=[ 
                               ('scale',StandardScaler()),
                               ('lr',DecisionTreeRegressor(random_state=42))
                              ])
X_ = df[['age','gender','bmi']].copy()
X_.gender = X_.gender.replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)

Missing = X_[X_.bmi.isna()]
X_ = X_[~X_.bmi.isna()]
Y_ = X_.pop('bmi')
DT_bmi_pipe.fit(X_,Y_)
predicted_bmi = pd.Series(DT_bmi_pipe.predict(Missing[['age','gender']]),index=Missing.index)
df.loc[Missing.index,'bmi'] = predicted_bmi







# Encoding categorical values


df['gender'] = df['gender'].replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)
df['Residence_type'] = df['Residence_type'].replace({'Rural':0,'Urban':1}).astype(np.uint8)
df['work_type'] = df['work_type'].replace({'Private':0,'Self-employed':1,'Govt_job':2,'children':-1,'Never_worked':-2}).astype(np.uint8)


sex_dictionary = {'Male':0,'Female':1,'Other':-1}#.astype(np.uint8)
work_type = {'Private':0,'Self-employed':1,'Govt_job':2,'children':-1,'Never_worked':-2}#.astype(np.uint8)


X  = df[['gender','age','hypertension','heart_disease','work_type','avg_glucose_level','bmi']]
y = df['stroke']





# HEADINGS
st.text("author: Siranjeevi")
st.title('stroke detection')
#st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

def user_report():
  gender = st.slider('gender', -1,1, 0)
  age = st.slider('age', 0,82, 20 )
  hypertension = st.slider('hypertension', 0, 1, 1)
  heart_disease = st.slider('heart_disease', 0,1, 0 )
  work_type = st.slider('work_type', -2,2, 0 )
  avg_glucose_level = st.slider('avg_glucose_level', 55,271, 120 )
  bmi = st.slider('bmi', 10,97, 20 )
 
    #Residence_type = st.slider('Residence_type', 0,1, 1 )
    


  user_report_data = {
	      'gender'           :gender,
	      'age'              :age,
	      'hypertension'     :hypertension,
	      'heart_disease'    :heart_disease,
	      'work_type'        :work_type,
	      'avg_glucose_level':avg_glucose_level,
	      'bmi'              :bmi,
	      
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data


#Patent data
user_data = user_report()
st.subheader('Patent Data')
st.write(user_data)










#Model
dt =  DecisionTreeClassifier()
dt.fit(X, y)
user_result = dt.predict(user_data)

#Output



st.subheader('Your Report: ')
output=''
if user_result[0]==0:
	output = 'You are good'
else:
	output = 'Result was found to be Stroke'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y, dt.predict(X))*100)+'%')




  









