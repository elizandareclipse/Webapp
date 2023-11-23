# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:51:17 2023

@author: Annan
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))

#creating a function for prediction
def diabetes_prediction(input_data):
    
   

    #changing the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped= input_data_as_numpy_array.reshape(1, -1)


    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'

    
def main():
    
    
    #Giving a title for the Web App
    st.title('Diabetes Prediction Web App')
    
    #Getting the input data from the user
  
    Pregnancies= st.text_input('Gap between Tests')
    Glucose= st.text_input('Glucose Level')
    BloodPressure= st.text_input('Blood Pressure Value')
    SkinThickness= st.text_input('Measurement of Biceps')
    Insulin= st.text_input('Insulin Level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input(' Diabetes Pedigree Function Value ')
    Age= st.text_input('Age of the patient')
    
    #code for prediction
    diagnosis= ''
    
    #creating a button for prediction
    
    if st.button("Diabetes Test Result"):
        diagnosis= diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
    st.success(diagnosis)
    
    
if __name__ =='__main__':
    main() 
