'''Use this code to pull in new data and use the trained model to predict outcomes. Eventually need to embed this in Streamlit. \n
Assume the input can be any variables but only select the ones required fro the model. Using only the hist grad model. Dont use any other options. Code relies on teh variable lists '''

import pandas as pd
import numpy as np
import math
import joblib
import os
import random as rand
import matplotlib.pyplot as plt
from matplotlib import cm 
import csv
from datetime import datetime,timedelta
import Code as noml
import streamlit as st
import ast
from io import StringIO

#======Functions===========================================================================
def read_input(csv_file):
    '''Reads the csv and converts to a dict list and then applies the exclusion criteria. returns the list of dicts'''
    data= noml.csv_to_dict(csv_file) #create a list of dicts of all episodes. May contain multiple copies of baby.
    print('Applying exclusion criteria')
    exclude_gest=['22','23','24','25','26','36','37','38','39','40']#exclude all episodes of these birth gestation
    for i,case in enumerate(data):
        for crit in exclude_gest:
            if case['GestationWeeks']==crit:
                exclude_HN=case['NationalIDBabyAnon']
                del data[i]
    readmission_list=set()
    for i,case in enumerate(data):
        if case['Readmission']=='1':
            readmission_HN=case['NationalIDBabyAnon']
            #print(readmission_HN)
            readmission_list.add(readmission_HN)

    for i,exclude in enumerate(readmission_list):#remove readmitted cases from data as per discussion with GG
        for j,case in enumerate(data):
            if exclude==case['NationalIDBabyAnon']:
                del data[j]
    HN_list=[]
    for i,case in enumerate(data):
        HN_list.append(case['NationalIDBabyAnon'])
    dups=noml.find_duplicates(HN_list)
    #print(len(dups))
    for j,case in enumerate(data):#Now remove all duplicate entries
        for dup in dups:
            if case['NationalIDBabyAnon']==dup:
                del data[j]
    if len(data) > 0:
        return(data)
    else:
        print('No valid cases found')
        return([])


def clean_data(data,input_vars):#need to confirm how girish wants the input variables inserted. This way should be agnostic of which vars are chosen
    '''Convert dict list data to a dict of values and then clean the data and get the list f birth times and gestations. Returns clean data dict and needs input vars as second arg'''
    print('Converting to list of vals')
    data_dict={}
    for keyname in input_vars:
        vals=noml.convert_to_np(data,keyname)
        data_dict.update({keyname:vals})
    #print(data_dict['GestationWeeks'])    

    '''Now clean the data -Convert gestation to days val. Get the discharge time. Convert to ints or floats or lists where appropriate. Replace missing vals. '''

    print('Cleaning data')
    clean_data_dict={}
    for variable in int_variables:
        for var in input_vars:
            if variable==var:
                #print(variable)
                int_vals=noml.convert_to_int(data_dict[variable])
                clean_data_dict.update({variable:int_vals})
    for variable in float_variables+float_variables_zero:
        for var in input_vars:
            if variable==var:
                #print(variable)
                float_vals=noml.convert_to_float(data_dict[variable])
                clean_data_dict.update({variable:float_vals})
    for variable in list_variables:
        for var in input_vars:
            if variable==var:
                #print(variable)
                list_vals=noml.convert_to_list(data_dict[variable])
                clean_data_dict.update({variable:list_vals})
    for variable in list_variables_str:
        for var in input_vars:
            if variable==var:
                 #print(variable)
                 list_vals=noml.convert_to_str_list(data_dict[variable])
                 clean_data_dict.update({variable:list_vals})   
    for variable in string_variables:
        for var in input_vars:
            if variable==var:
                str_vals=noml.convert_to_str(data_dict[variable])
                #print(variable)
                clean_data_dict.update({variable:str_vals})


    Birthtime_list=[]#just need the list of birth times in order to preduct date at the end. 
    for time in data_dict['BirthTimeBaby']:
        Birthtime_list.append(time)    
    clean_data_dict.update({'Birth_time':Birthtime_list})

    #print(clean_data_dict['duration_of_stay'])
    gestation_days=clean_data_dict['GestationWeeks']*7+clean_data_dict['GestationDays']
    clean_array=noml.convert_to_float(gestation_days)
    clean_data_dict.update({'gestation_days':clean_array})
    return(clean_data_dict)

def pre_process(data,input_vars):
    '''Pre pocess data ie one hot encoding where necessary. Input data must be a dict. second arg is input vars. returns preprocessed data'''
    '''Now preprocess the data - one hot encoding and replace missing vals with median or 0. Store the category orders somewhere'''

    print('Preprocessing data')
    #print(clean_data_dict['ProblemsPregnancyMother'])
    #print(encode_list(clean_data_dict['ProblemsPregnancyMother'],verbose=True))
    for variable in list_variables+list_variables_str:
        for var in input_vars:
            if variable==var:
                print('encoding '+variable)
                encoded_array=noml.encode_list(data[variable],verbose=verbose)
                #print(encoded_array[0],'classes',encoded_array[1])
                data.update({'encoded_'+variable:encoded_array[0]})#get the values
                data.update({'encoded_'+variable+'_classes':encoded_array[1]})#get the one hot encoded categories too
    for variable in int_variables+string_variables:
        for var in input_vars:
            if variable==var:
                print('encoding '+variable)
                encoded_array=noml.encode_int(data[variable],verbose=verbose)
                #print(encoded_array[0],'classes',encoded_array[1])
                data.update({'encoded_'+variable:encoded_array[0]})
                data.update({'encoded_'+variable+'_classes':encoded_array[1]})  

    for variable in float_variables+new_variables:#replace missing values in floats with median
        for var in input_vars:
            if variable==var:
                print('cleaning '+variable)
                median_val=np.nanmedian(data[variable])
                for i,val in enumerate(data[variable]):
                    if np.isnan(val):
                        data[variable][i]=median_val
    for variable in float_variables_zero:#replace missing values in floats with 0
        for var in input_vars:
            if variable==var:
                print('cleaning '+variable)
                median_val=np.nanmedian(data[variable])
                for i,val in enumerate(data[variable]):
                    if np.isnan(val):
                        data[variable][i]=0
    return(data)


def make_input_array(data):
    '''Convert the data to an input array that the model can use to predict y. Input is clean data dict with encoded variables. Returns X'''
    zero_d=len(data['NationalIDBabyAnon'])
    one_d= len(new_input_variables_cont)+len(new_input_variables_cat)
    cont_d=len(new_input_variables_cont)
    cat_d=len(new_input_variables_cat)
    print('variable length',len(variables))
    print('cont var length',len(new_input_variables_cont))
    print('cat var length',len(new_input_variables_cat))
    twodarray_list=[]
    for var in (new_input_variables_cont+new_input_variables_cat):#note that this is the var order. The cat vars will have a category order also. 
        variable=data[var]
        if variable.ndim==1:
            twodvar=variable.reshape(-1,1)#convert all arrays to 2 d and hstack them columnwise.
        else:
            twodvar=variable
        twodarray_list.append(twodvar)
    input_array=np.hstack(twodarray_list)
    X=input_array    
    print('X shape',X.shape)
    return(X)

def load_model(model_name,X):
    '''loads the model and then uses X to output y. Needs model name and X input'''
    loaded_model=joblib.load(model_name)
    y_predict=noml.predict_data(X,loaded_model)
    return(y_predict)


def predict_dates(y_predict,data):
    '''Now use the predicted durations to predict the date of discharge and print. need clean data dict also for birthtime'''
    days_predict=[]
    for num in y_predict:
        val=math.ceil(num)
        days_predict.append(val)
    date_format = "%d/%m/%Y %H:%M" 
    date_list =[datetime.strptime(date, date_format) for date in data['Birth_time']]
    pred_dates=[date + timedelta(days=duration) for date, duration in zip(date_list, days_predict)]
    new_dates_str = [date.strftime("%d/%m/%Y") for date in pred_dates]
    print(new_dates_str,data['NationalIDBabyAnon'])
    return(new_dates_str,data['NationalIDBabyAnon'])

#-------Variables------------------------------------------------------------------------------------------------
'''These are the variable I could see as relevent. This is all variables from the data pull from badger'''
variables=['BadgerUniqueID',
'NationalIDBabyAnon',
'EpisodeNumber',
'Sex',
'Readmission',
'BirthTimeBaby',
'GestationWeeks',
'GestationDays',
'Birthweight',
'BirthHeadCircumference',
'BirthOrder',
'AdmitTime',
'AdmitTemperature',
'DischTime',
'DischargeDestination',
'ResusSurfactant',
'VentilationDays',
'CPAPDays',
'ICCareDays2011',
'HDCareDays2011',
'SCCareDays2011',
'OxygenDays',#float
'MeconiumStainedLiquor',#int
'LabourOnset',#int
'SteroidsName',#int
'DrugsDuringStay',#list
'CordClampingTimeSecond',#float
'CordClampingTimeMinute',#float
'MaritalStatusMother',#str
'OffensiveLiquor',#int
'LabourPresentation',#int
'CordClamping',#int
'BloodGroupMother',#str
'AdmitHeadCircumference',#float
'SteroidsAntenatalGiven',#int
'DrugsInLabour',#list
'CordArterialpH',#float
'CordVenouspH',#float
'GPPostCode',#str
'ParenteralNutritionDays',#float
'AdmitPrincipalReason',#int
'DischargeFeeding',#list
'DischargeMilk',#list
'BirthOrder',#int
'NormalCareDays2011',
'NECDiagnosis',
'SteroidsAntenatalCourses',
'MembranerupturedDuration',
'HeadScanFirstResult',
'HeadScanLastResult',
'ProblemsMedicalMother',
'DiabetesMother',
'DrugsAbusedMother',
'SmokingMother',
'AlcoholMother',
'ProblemsPregnancyMother',
'Apgar1', 
'Apgar5', 
'LabourDelivery',
'Resuscitation',
'Pneumothorax',
'NecrotisingEnterocolitis',
'MaternalPyrexiaInLabour38c',
'AdmitBloodGlucose',
'HIEGrade']

int_variables=[
'EpisodeNumber',
'Readmission',
'BirthOrder',
'DischargeDestination',
'ResusSurfactant',
'NECDiagnosis',
'SteroidsAntenatalCourses',
'DiabetesMother',
'SmokingMother',
'AlcoholMother',
'Apgar1', 
'Apgar5', 
'LabourDelivery',
'Pneumothorax',
'NecrotisingEnterocolitis',
'MaternalPyrexiaInLabour38c',
'HIEGrade','MeconiumStainedLiquor','LabourOnset',
'SteroidsName','OffensiveLiquor',
'CordClamping','SteroidsAntenatalGiven','AdmitPrincipalReason','BirthOrder']

time_variables=['BirthTimeBaby','AdmitTime','DischTime']

float_variables=['AdmitTemperature','BirthHeadCircumference','AdmitBloodGlucose','GestationWeeks',
                 'GestationDays','Birthweight','AdmitHeadCircumference','CordArterialpH','CordVenouspH']

float_variables_zero=['VentilationDays','CPAPDays','ICCareDays2011','HDCareDays2011','SCCareDays2011','NormalCareDays2011','MembranerupturedDuration','OxygenDays','CordClampingTimeSecond',
'CordClampingTimeMinute','ParenteralNutritionDays']

list_variables=['ProblemsPregnancyMother','ProblemsMedicalMother','Resuscitation','DrugsAbusedMother','DrugsInLabour','LabourPresentation','DischargeFeeding','DischargeMilk']#list of ints

list_variables_str=['DrugsDuringStay']#had to add this as this is a list of strings not ints from a picklist. would put diagnosis in this cat

string_variables=['Sex','BadgerUniqueID','NationalIDBabyAnon','HeadScanFirstResult','HeadScanLastResult','MaritalStatusMother','BloodGroupMother','GPPostCode']

new_variables=['duration_of_stay','gestation_days']

orig_input_variables_cat=['EpisodeNumber',#these are the variables used for yje poster but they contain paameters that can only be known at discharge. 
'Readmission',
'BirthOrder',
'DischargeDestination',
'ResusSurfactant',
'NECDiagnosis',
'SteroidsAntenatalCourses',
'DiabetesMother',
'SmokingMother',
'AlcoholMother',
'Apgar1', 
'Apgar5', 
'LabourDelivery',
'Pneumothorax',
'NecrotisingEnterocolitis',
'MaternalPyrexiaInLabour38c',
'HIEGrade','MeconiumStainedLiquor','LabourOnset',
'SteroidsName','OffensiveLiquor',
'CordClamping','SteroidsAntenatalGiven','AdmitPrincipalReason','BirthOrder','ProblemsPregnancyMother',
'ProblemsMedicalMother','Resuscitation','DrugsAbusedMother','DrugsInLabour','LabourPresentation','DischargeFeeding','DischargeMilk',
'DrugsDuringStay','Sex','HeadScanFirstResult','HeadScanLastResult','MaritalStatusMother','BloodGroupMother','GPPostCode']

orig_input_variables_cont=['AdmitTemperature','BirthHeadCircumference','AdmitBloodGlucose',
'gestation_days','Birthweight','AdmitHeadCircumference','CordArterialpH','CordVenouspH','VentilationDays','CPAPDays','MembranerupturedDuration','OxygenDays','CordClampingTimeSecond',
'CordClampingTimeMinute','ParenteralNutritionDays','ICCareDays2011']#,'HDCareDays2011','SCCareDays2011','NormalCareDays2011'

new_input_variables_cat=['EpisodeNumber',#can adjust these if needed, Currently adjusted so all are nown at birth other than ICC days. These must match teh model
'Readmission',
'BirthOrder',
'ResusSurfactant',
'SteroidsAntenatalCourses',
'DiabetesMother',
'SmokingMother',
'AlcoholMother',
'Apgar1', 
'Apgar5', 
'LabourDelivery',
'MaternalPyrexiaInLabour38c',
'HIEGrade','MeconiumStainedLiquor','LabourOnset',
'SteroidsName','OffensiveLiquor',
'CordClamping','SteroidsAntenatalGiven','AdmitPrincipalReason','BirthOrder','ProblemsPregnancyMother',
'ProblemsMedicalMother','Resuscitation','DrugsAbusedMother','DrugsInLabour','LabourPresentation',
'Sex','MaritalStatusMother','BloodGroupMother','GPPostCode']

new_input_variables_cont=['AdmitTemperature','BirthHeadCircumference','AdmitBloodGlucose',
'gestation_days','Birthweight','AdmitHeadCircumference','CordArterialpH','CordVenouspH','MembranerupturedDuration','CordClampingTimeSecond',
'CordClampingTimeMinute','ICCareDays2011']#,'HDCareDays2011','SCCareDays2011','NormalCareDays2011'

verbose=False

model_name='new_hist_grad_model.joblib'

def main():
    st.title ("Discharge date Calculator. This calculator predicts the date of discharge using the trained Model.\n The input dictionary can handle one baby at the moment as a pull from Badger.\n")
    st.write("The categorical variables required are")
    for var in new_input_variables_cat:
        st.write(var)
    st.write("The continuous variables required are")  
    for var in new_input_variables_cont:
         st.write(var)
    st.write('Birthdate and Anonymous ID is also required')
    st.write('Only 27+0 to 35+6 week gestation babies can be considered')
    csv_input = st.text_area("Input your csv here as a text string):")
    if csv_input:
        try:
            csv_file = StringIO(csv_input)
            st.write("Converted csv. Now converting to dict:")
        except Exception as e:
            st.error("Error converting the input to a dictionary. Please ensure it is in the correct format.")
            st.error(e)
    data=read_input(csv_file)
    if len(data)==0:
        st.error('No valid cases were identified. Please retry')
        st.error(e)
    else:
        input_vars=data[0].keys
        cleaned_data=clean_data(data)
        processed_data=pre_process(cleaned_data,input_vars)
        X=make_input_array(processed_data)
        y_predict=load_model(model_name,X)
        predicted_dates,babyIDanon=predict_dates(y_predict,processed_data)
        for i,date in enumerate(predicted_dates):
            st.write(f'predicted discharge date for {babyIDanon[i]} is {date}') 



if __name__ == "__main__":
    main()