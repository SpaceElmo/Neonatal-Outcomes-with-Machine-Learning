'''Use this code to pull in new data and use the trained model to predict outcomes. Eventually need to embed this in Streamlit'''

import pandas as pd
import numpy as np
import math
import joblib
import os
import random as rand
import matplotlib.pyplot as plt
from matplotlib import cm 
import csv
from datetime import datetime
import Code as noml

#-------Variables------------------------------------------------------------------------------------------------
'''These are the variable I could see as relavent'''
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

input_variables_cat=['EpisodeNumber',
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
'DrugsDuringStay','Sex','HeadScanFirstResult','HeadScanLastResult','MaritalStatusMother','BloodGroupMother','GPPostCode']#actual variables used. Could extract this into a text file if

input_variables_cont=['AdmitTemperature','BirthHeadCircumference','AdmitBloodGlucose',
'gestation_days','Birthweight','AdmitHeadCircumference','CordArterialpH','CordVenouspH','VentilationDays','CPAPDays','MembranerupturedDuration','OxygenDays','CordClampingTimeSecond',
'CordClampingTimeMinute','ParenteralNutritionDays','ICCareDays2011']#,'HDCareDays2011','SCCareDays2011','NormalCareDays2011'

output_variables=['duration_of_stay']

verbose=False


file_path='../data/predict_data.csv'
data= noml.csv_to_dict(file_path) #create a list of dicts of all episodes. May contain multiple copies of baby. 
remainder=set()
all_keys=data[0].keys
for key in data[0].keys():
    if key in variables:
        continue
    else:
        remainder.add(key)

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


#print(len(exclude_list))
#print(len(readmission_list))
#print(len(data))


'''Convert the list of dicts into a dict of values'''
print('Converting to list of vals')
data_dict={}
for i,keyname in enumerate(variables):
    vals=noml.convert_to_np(data,keyname)
    data_dict.update({keyname:vals})
#print(data_dict['GestationWeeks'])    

'''Now clean the data -Convert gestation to days val. Get the discharge time. Convert to ints or floats or lists where appropriate. Replace missing vals. '''

print('Cleaning data')
clean_data_dict={}
for variable in int_variables:
    #print(variable)
    int_vals=noml.convert_to_int(data_dict[variable])
    clean_data_dict.update({variable:int_vals})
for variable in float_variables+float_variables_zero:
    #print(variable)
    float_vals=noml.convert_to_float(data_dict[variable])
    clean_data_dict.update({variable:float_vals})
for variable in list_variables:
    #print(variable)
    list_vals=noml.convert_to_list(data_dict[variable])
    clean_data_dict.update({variable:list_vals})
for variable in list_variables_str:
    #print(variable)
    list_vals=noml.convert_to_str_list(data_dict[variable])
    clean_data_dict.update({variable:list_vals})   
for variable in string_variables:
    str_vals=noml.convert_to_str(data_dict[variable])
    #print(variable)
    clean_data_dict.update({variable:str_vals})

duration_list=[]
Birthtime_list=[]
for i,time in enumerate(data_dict['BirthTimeBaby']):
    duration=noml.days_difference(time,data_dict['DischTime'][i])#Note that this is defined from birth to discharge, not admission. Later episodes will not depend on admission time
    duration_list.append(duration)
    Birthtime_list.append(time)
clean_duration=noml.convert_to_float(duration_list)    
clean_data_dict.update({'duration_of_stay':clean_duration})
clean_data_dict.update({'Birth_time':Birthtime_list})

#print(clean_data_dict['duration_of_stay'])
gestation_days=clean_data_dict['GestationWeeks']*7+clean_data_dict['GestationDays']
clean_array=noml.convert_to_float(gestation_days)
clean_data_dict.update({'gestation_days':clean_array})



#for val in clean_data_dict['DrugsDuringStay']:
    #print(val)
#print(clean_data_dict['duration_of_stay'])
#print(np.min(clean_data_dict['GestationWeeks']))

'''Now preprocess the data - one hot encoding and replace missing vals with median or 0. Store the category orders somewhere'''

print('Preprocessing data')
#print(clean_data_dict['ProblemsPregnancyMother'])
#print(encode_list(clean_data_dict['ProblemsPregnancyMother'],verbose=True))
for variable in list_variables+list_variables_str:
    print('encoding '+variable)
    encoded_array=noml.encode_list(clean_data_dict[variable],verbose=verbose)
    #print(encoded_array[0],'classes',encoded_array[1])
    clean_data_dict.update({'encoded_'+variable:encoded_array[0]})#get the values
    clean_data_dict.update({'encoded_'+variable+'_classes':encoded_array[1]})#get the one hot encoded categories too
for variable in int_variables+string_variables:
    print('encoding '+variable)
    encoded_array=noml.encode_int(clean_data_dict[variable],verbose=verbose)
    #print(encoded_array[0],'classes',encoded_array[1])
    clean_data_dict.update({'encoded_'+variable:encoded_array[0]})
    clean_data_dict.update({'encoded_'+variable+'_classes':encoded_array[1]})  

for variable in float_variables+new_variables:#replace missing values in floats with median
    print('cleaning '+variable)
    median_val=np.nanmedian(clean_data_dict[variable])
    for i,val in enumerate(clean_data_dict[variable]):
        if np.isnan(val):
            clean_data_dict[variable][i]=median_val
for variable in float_variables_zero:#replace missing values in floats with 0
    print('cleaning '+variable)
    median_val=np.nanmedian(clean_data_dict[variable])
    for i,val in enumerate(clean_data_dict[variable]):
        if np.isnan(val):
            clean_data_dict[variable][i]=0


#for variable in float_variables_zero+float_variables:
#    print(variable,clean_data_dict[variable])

'''Convert the data to an input array that the model can use to predict y'''
use_encoded_var=True#if categoricals need to be one hot encoded then set this to true
if use_encoded_var:
    for i,variable in enumerate(input_variables_cat):
        input_variables_cat[i]='encoded_'+variable#uses encoded variables

zero_d=len(clean_data_dict['NationalIDBabyAnon'])
one_d= len(input_variables_cont)+len(input_variables_cat)
cont_d=len(input_variables_cont)
cat_d=len(input_variables_cat)
print('variable length',len(variables))
print('cont var length',len(input_variables_cont))
print('cat var length',len(input_variables_cat))
twodarray_list=[]
for var in (input_variables_cont+input_variables_cat):#note that this is the var order. The cat vars will have a category order also. 
    variable=clean_data_dict[var]
    if variable.ndim==1:
        twodvar=variable.reshape(-1,1)#convert all arrays to 2 d and hstack them columnwise.
    else:
        twodvar=variable
    twodarray_list.append(twodvar)
input_array=np.hstack(twodarray_list)
X=input_array    
print('X shape',X.shape)
full_var_list=[]#label the columns in the input array with teh appropriate category for later analysis. cat names are repeated if the encoding is in the same cat
for variable in (input_variables_cont):
    full_var_list.append(variable)
for variable in (input_variables_cat):
    #print(variable)
    #print(clean_data_dict[variable+'_classes'])
    num_cat=len(clean_data_dict[variable+'_classes'][0])
    for _ in range(num_cat):
        full_var_list.append(variable)
print('length of the input var list labels',len(full_var_list))

'''Now load the model'''
model_name='random_forest_model.joblib'
loaded_model=joblib.load(model_name)
y_predict=noml.predict_data(X,loaded_model)
days_predict=[]
for num in y_predict:
    val=math.ceil(num)
    days_predict.append(val)



