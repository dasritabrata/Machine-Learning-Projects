import numpy as np
import pickle

import streamlit as st
filename="diabetes_trained_model.sav"
loaded_model=pickle.load(open(filename,"rb"))

def diabetes_prediction(input_data):
    # input_data=(4,110,92,0,0, 37.6,0.191,30)
# changing input data to np array
    input_data_as_nparray=np.asarray(input_data)

# reshape the array

    input_data_reshaped=input_data_as_nparray.reshape(1,-1)

# standardizing input data

# std_data=scaler.transform(input_data_reshaped)
# print(std_data)

    prediction=loaded_model.predict(input_data_reshaped)
# print(prediction)

    if(prediction[0]==0):
        return "Non Diabetic"
    else:
        return "Diabetic"   
    
def main():

    st.title("Diabetes Prediction App")


    # getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()