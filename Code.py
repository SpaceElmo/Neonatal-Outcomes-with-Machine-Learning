'''This is the main code that generates a model based on training data. Model is stored in the local folder'''
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sklearn
import math
import joblib
import os
from sklearn import metrics
from sklearn.model_selection import cross_val_score,RepeatedKFold,train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import r2_score
import torch.optim as optim 
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer 
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from matplotlib import cm 
import requests
from supabase import create_client, Client
import csv
from datetime import datetime
import pickle
#BirthDateMother #EthnicityMother
def csv_to_dict(file_path):
    '''converts csv into a list of dicts with one dict for each contact'''
    with open(file_path, newline='',mode='r',encoding="latin-1") as file:
        csv_reader = csv.DictReader(file,dialect='excel') 
        data = [row for row in csv_reader] 
        return data

def parse_date(date_str):
    '''Converts dates in different formats'''
    formats = ["%d/%m/%Y %H:%M","%d-%m-%y","%d/%m/%Y","%Y/%m/%d %H:%M"]
    for fmt in formats:
        #print(date_str,fmt)
        try:
            #print(datetime.strptime(date_str,fmt))
            return datetime.strptime(date_str,fmt)
        except:
            #print('Error in date found', date_str,fmt)
            continue

def fix_dataset(filename,col_to_remove):
    'remove a key from a dataset as csv and save'
    df=pd.read_csv(filename)
    df.drop(columns=[col_to_remove],inplace=True)
    df.to_csv('../data/modified.csv',index=False)
    


def days_difference(date1, date2):
    '''date1 is the earlier time to generate a positive number. Must be positive. If a value is missing will generate None''' 
    #print(date1,date2)
    if date1 and date2:
        d1 = parse_date(date1)
        #print(d1,"date 1")
        d2 = parse_date(date2)
        #print(d2,"date 2")
        delta = d2 - d1 
        return float(delta.days)
    else:
        return None

def find_duplicates(input_list):
    '''input must be a list'''
    duplicates = []
    seen = set()
    for item in input_list:
        if item in seen:
            duplicates.append(item)
        else:
            seen.add(item)
    return duplicates


def convert_to_np(datasource,keyname):
    '''Using the list of dicts as datasource and a keyword, convert to an np.array of vals'''
    array=np.array([])
    for case in datasource:
        val=case[keyname]
        array=np.append(array,val)
    return(array)    

def convert_to_int(array):
    '''Missing vals are labelled 999'''
    #print("int",array)
    int_array = np.array([int(float(x)) if x else 999 for x in array])
    return int_array

def convert_to_float(array):
    '''Missing vals are labelled 999'''
    #print("float",array)
    float_array = np.array([float(x) if x else np.nan for x in array])
    return float_array

def convert_to_str(array):
    '''Missing vals are labelled None'''
    #print("str",array)
    str_array = np.array([str(x) if x else 'None' for x in array])
    str_array= np.char.replace(str_array, 'MS', 'M')
    str_array= np.char.replace(str_array, 'CH', 'C')
    return str_array    

def convert_to_list(array):
    '''Missing vals are labelled 999. converts comma seperated numbers in strings to lists of numbers. Good for picklists.'''
    #string_array = np.char.replace(array, '', '999')
    list_array=[]
    for s in array:
        #print(s)
        if s=='':
            clean_sub_array=[999]
        else:
            sub_list=s.split(',')
            #print(sub_list)#Use this to debug. Replace strings with numbers using lines below
            sub_list = np.char.replace(sub_list, 'Cannabis', '8')
            sub_list = np.char.replace(sub_list, 'MethadonePrescribed', '3')
            sub_list = np.char.replace(sub_list, 'Crack Cocaine', '5')
            sub_list = np.char.replace(sub_list, 'HaemDisOther', '88')
            sub_list = np.char.replace(sub_list, 'Maternal mental health', '88')
            sub_list = np.char.replace(sub_list, 'UVC', '99')
            sub_list = np.char.replace(sub_list, 'vertex', '2')
            sub_list = np.char.replace(sub_list, 'u', '9')
            sub_list = np.char.replace(sub_list, 'frank', '1')
            sub_list = np.char.replace(sub_list, 'footling', '1')
            sub_list = np.char.replace(sub_list, 'brow', '8')
            sub_list = np.char.replace(sub_list, 'complete', '1')
            sub_list = np.char.replace(sub_list, 'face', '8')
            sub_list = np.char.replace(sub_list, 'Antihypertensive', '18')
            #print(sub_list)
            #sub_list=np.char.replace(sub_list, '', '999')
            clean_sub_array = list(map(int, sub_list))
        list_array.append(clean_sub_array)
    return list_array

def convert_to_str_list(array):
    '''Missing vals are labelled None. converts comma seperated numbers in strings to lists of strings. Used gpt to combine drug names.'''
    #string_array = np.char.replace(array, '', '999')
    list_array=[]
    for s in array:
        #print(s)
        if s=='':
            clean_sub_array=['None']
        else:
            split_list=s.split(',')
            sub_list=[i.strip() for i in split_list]
            #print(sub_list)#Use this to debug. Replace strings using lines below
            sub_list=np.char.replace(sub_list, 'Adrenaline (Epinephrine)', 'Adrenaline')
            sub_list=np.char.replace(sub_list, 'Azithromycin Oral', 'Azithromycin')
            sub_list=np.char.replace(sub_list, 'Bactroban ointment', 'Bactroban')
            sub_list=np.char.replace(sub_list, 'Bactraban', 'Bactroban')
            sub_list=np.char.replace(sub_list, 'Caffeine Citrate', 'Caffeine')
            for char in ['Calcium', 'Calcium carbonate (tablets)', 'Calcium gluconate 10%', 'Calvive (oral calcium)', 'Calcium Sandoz']:
                sub_list=np.char.replace(sub_list, char, 'Calcium')
            for char in ['Canestan cream (Clotrimazole)', 'Clotrimazole', 'Clotrimazole (pessary)', 'Clotrimazole Cream']:
                sub_list=np.char.replace(sub_list, char, 'Clotrimazole')
            for char in ['Chloramphenicol', 'Chloramphenicol eyedrops', 'Chloramphenicol eye ointment','Chloramphenical eyedrops']:
                sub_list=np.char.replace(sub_list, char, 'Chloramphenicol')
            for char in ['Daktarin (See Miconazole)', 'Miconazole', 'Miconazole cream HC', 'Miconazole gel / cream']:
                sub_list=np.char.replace(sub_list, char, 'Miconazole')  
            for char in ['Dextrose (See Glucose 10%% or Glucose any Conc)', 'Dextrose 10%', 'Dextrose 5%','Glucose 10%', 'Glucose Gel 40% (oral)', 'Glycogel','Sucrose', 'Sucrose (oral)','Hypostop','Dextrose (oral)','Dextrose (See Dextrose or Glucose any Conc)']:
                sub_list=np.char.replace(sub_list, char, 'Dextrose')
            for char in ['Heparinized Saline','Hepsal']:
                sub_list=np.char.replace(sub_list, char, 'Heparin') 
            for char in ['Human milk fortifier', 'Human milk fortifier (Nutriprem cow and gate)', 'Human milk fortifier (SMA)', 'Carobel','Fortifier (Nutriprem cow and gate)','Fortifier (SMA)']:
                sub_list=np.char.replace(sub_list, char, 'Fortifier')
            for char in ['Joules Phosphate','Sodium Acid Phosphate','Sodium Phosphate','buffered phosphate','Polyfusor Phosphates']:
                sub_list=np.char.replace(sub_list, char, 'Phosphate')
            for char in ['Ketovite','ABIDEC', 'Multivitamins','Dalivit','Vitamins']:
                sub_list=np.char.replace(sub_list, char, 'Abidec')
            for char in ['Vitamin K', 'Vitamin K (Phytomenadione)', 'Vitamin K - 2nd Dose','Konakion (Phytomenadione)','Konakion - 2nd Dose']:
                sub_list=np.char.replace(sub_list, char, 'Konakion')
            for char in ['IV Morphine','Oral Morphine', 'Oramorph','Morphine sulphate']:
                sub_list=np.char.replace(sub_list, char, 'Morphine')
            for char in ['Poractant Alfa - Curosurf','Surfactant']:
                sub_list=np.char.replace(sub_list, char, 'Surfactant') 
            for char in ['Saline 0.9%', 'Sodium Chloride 0.9%', 'Sodium Chloride for flush','Sodium','Sodium chloride 30%','Sodium Chloride chloride 30%','Sodium Chloride Chloride','Sodium chloride','Sodium Chloride chloride','Sodium Chloride chloride 30%']:
                sub_list=np.char.replace(sub_list, char, 'Sodium Chloride')
            for char in ['Iron (Sytron)','Sodium Ferederate','Sytron - Sodium Ironedetate (Sodium Feredate)','Sytron','Ferrous Sulphate','Sodium Chloride Ferederate','Fersamal (ferrous fumarate)','Iron - Sodium Chloride Ironedetate (Sodium Chloride Feredate)']:
                sub_list=np.char.replace(sub_list, char, 'Iron') 
            for char in ['Sodium Bicarbinate 8.4%','Sodium Chloride Bicarbinate 8.4%','Sodium Chloride Bicarbonate']:
                sub_list=np.char.replace(sub_list, char, 'Sodium Bicarbonate')
            for char in ['Omerazole','Esomeprazole']:
                sub_list=np.char.replace(sub_list, char, 'Omeprazole')
            for char in ['Flamazine cream','Flaminal hydro gel']:
                sub_list=np.char.replace(sub_list, char, 'Flamazine')                    
            for char in ['DTaP/IPV/HiB/HepB (6 in 1) Infanrix hexa','Pediacel']:
                sub_list=np.char.replace(sub_list, char, 'DTaP/IPV/HiB/HepB (6 in 1)')     
            sub_list=np.char.replace(sub_list, 'Colecalciferol', 'Cholecalciferol')
            sub_list=np.char.replace(sub_list, 'Prevenar', 'Pneumococcal Vaccine')
            sub_list=np.char.replace(sub_list, 'Gaviscon infant', 'Gaviscon')
            sub_list=np.char.replace(sub_list, 'Hepatitis B immunoglobulin', 'Hepatitis B vaccine')
            sub_list=np.char.replace(sub_list, 'Insulin Actrapid', 'Insulin')
            sub_list=np.char.replace(sub_list, 'Nystatin suspension', 'Nystatin')
            sub_list=np.char.replace(sub_list, 'Phenobarbitone - loading dose', 'Phenobarbitone')
            sub_list=np.char.replace(sub_list, 'Potassium Chloride', 'Potassium')
            sub_list=np.char.replace(sub_list, 'rotavirus vaccine (Rotarix®)', 'Rotarix')
            clean_sub_array = list(sub_list)
        list_array.append(clean_sub_array)
    return list_array    

def encode_int(array,verbose=False):
    '''One hot encode categorical variables if given array of variables as a 1d int array. Will be converted to 2d and array returned. Returns array and array of classes'''
    encoder=OneHotEncoder(sparse_output=False,categories='auto')
    twodarray=np.array(array).reshape(-1,1)
    encoded_array=encoder.fit_transform(twodarray)
    if verbose:
        print('encoded shape is ',encoded_array.shape)
        print('encoded categories are ',encoder.fit(twodarray).categories_)
    classes=encoder.fit(twodarray).categories_   
    return (encoded_array,classes)    

def encode_list(array,verbose=False):
    '''One hot encode categorical variables if given array of variables as a 2d array with category values within a list.Returns a np array ov vals and an array of classes'''
    mlb=MultiLabelBinarizer()
    classes=[]
    encoded_array=mlb.fit_transform(array)
    if verbose:
        print('encoded shape is ',encoded_array.shape)
        print('encoded categories are ',mlb.fit(array).classes_)  
    classes.append(mlb.fit(array).classes_)      
    return (encoded_array,classes)
       

def evaluate_model(X,y,splits,repeats):
    'evaluate randomforest model. Returns R2_mean,R2_std,fitted_model,MAE_score,X_test,y_test,y_train,X_train as a tuple'
    hyperparam_dis={'n_estimators': [100,200,300,500,750,1000],  
                    'max_depth': [None, 10, 20, 50, 100], 
                    'min_samples_split': [2, 5, 10, 15, 20], 
                    'min_samples_leaf': [1, 2, 5, 10, 15]
                    }
    #use regressor model to generate new model from data X and outcome y. search cross validator to get best hyperparams. Train and test sample
    model=RandomForestRegressor()
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=11)
    print('got training and test sets')
    #print(X_train,y_train)
    all_datasets=[X_train,X_test]
    search=RandomizedSearchCV(estimator=model,n_iter=10,param_distributions=hyperparam_dis,random_state=11,n_jobs=-1)
    #search=GridSearchCV(estimator=model,param_grid=hyperparam_dis,n_jobs=-1)
    search.fit(X=X_train,y=y_train)
    params=search.best_params_
    MAE_score=search.score(X_test,y_test)
    #cv=RepeatedKFold(n_splits=splits,n_repeats=repeats,random_state=11)
    #print(cv)
    model=RandomForestRegressor(**params)
    fitted_model=model.fit(X=X_train,y=y_train)
    n_scores=cross_val_score(fitted_model,X=X_test,y=y_test,cv=5,n_jobs=-1,error_score='raise',scoring='r2')
    R2_mean=n_scores.mean()
    R2_std=n_scores.std()
    #print(n_scores)
    print('MAE: %.3f(%.3f)',n_scores,n_scores)
    return(R2_mean,R2_std,fitted_model,MAE_score,X_test,y_test,y_train,X_train)

def evaluate_hist_model(X,y,splits,repeats,cat_feat):
    'evaluate histgradientboostregressor model. Provide input and output for training. Provide splits and repeats for kfold cross validation. Provide the boolean array for the features that are categorical.Returns R2_mean,R2_std,fitted_model,MAE_score,X_test,y_test,y_train,X_train as a tuple. Similar kfolds and gridsearch cv as random forest'
    hyperparam_dis={'learning_rate':[0.01,0.05,0.1,0.2],
                    'max_iter':range(50,500,50),
                    'max_leaf_nodes':range(10,200,10),
                    'min_samples_leaf':range(10,100,10),
                    'max_bins':[50,100,150,255]
                    }
    #use regressor model to generate new model from data X and outcome y. search cross validator to get best hyperparams. Train and test sample
    model=HistGradientBoostingRegressor(categorical_features=cat_feat)
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=11)
    print('got training and test sets')
    #print(X_train,y_train)
    all_datasets=[X_train,X_test]
    search=RandomizedSearchCV(estimator=model,n_iter=10,param_distributions=hyperparam_dis,random_state=11,n_jobs=-1)
    #search=GridSearchCV(estimator=model,param_grid=hyperparam_dis,n_jobs=-1)
    search.fit(X=X_train,y=y_train)
    params=search.best_params_
    print('best hyperparams are ', params)
    MAE_score=search.score(X_test,y_test)
    #cv=RepeatedKFold(n_splits=splits,n_repeats=repeats,random_state=11)
    #print(cv)
    model=HistGradientBoostingRegressor(**params,categorical_features=cat_feat)
    fitted_model=model.fit(X=X_train,y=y_train)
    n_scores=cross_val_score(fitted_model,X=X_test,y=y_test,cv=5,n_jobs=-1,error_score='raise',scoring='r2')
    R2_mean=n_scores.mean()
    R2_std=n_scores.std()
    #print(n_scores)
    print('MAE: %.3f(%.3f)',n_scores,n_scores)
    return(R2_mean,R2_std,fitted_model,MAE_score,X_test,y_test,y_train,X_train)    

def predict_data(X_test,model):
    '''predict data with x_test and fitted model as arguments. Return y vals predict'''
    yhat=model.predict(X_test)
    #print(yhat)
    return(yhat)

# Define the nn training loop
def training_loop(n_epochs,lr,optimizer,model,loss_fn,X_train,X_val,y_train,y_val):
    for epoch in range(1,n_epochs+1):
        y_pred_train=model(X_train)
        loss_train=loss_fn(y_pred_train,y_train)
        y_pred_val=model(X_val)
        loss_val=loss_fn(y_pred_val,y_val)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        if epoch==1 or epoch % 100 ==0:
            print(f"Epoch {epoch} , Training loss {loss_train.item():.4f},"f"Validation loss {loss_val.item():.4f}")
        


def main():



    today = datetime.today()
    #fix_dataset('../data/Badger download.csv' ,"HRG2016_Unknown")
    #print('Data fixed')




    #-------Variables------------------------------------------------------------------------------------------------
    '''These are the variable I could see as relevent. Do not change order of variables'''
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
    'HIEGrade','BirthDateMother', 'EthnicityMother']

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

    time_variables=['BirthTimeBaby','AdmitTime','DischTime','BirthDateMother']

    float_variables=['AdmitTemperature','BirthHeadCircumference','AdmitBloodGlucose','GestationWeeks',
                    'GestationDays','Birthweight','AdmitHeadCircumference','CordArterialpH','CordVenouspH']

    float_variables_zero=['VentilationDays','CPAPDays','ICCareDays2011','HDCareDays2011','SCCareDays2011','NormalCareDays2011','MembranerupturedDuration','OxygenDays','CordClampingTimeSecond',
    'CordClampingTimeMinute','ParenteralNutritionDays']

    list_variables=['ProblemsPregnancyMother','ProblemsMedicalMother','Resuscitation','DrugsAbusedMother','DrugsInLabour','LabourPresentation','DischargeFeeding','DischargeMilk']#list of ints

    list_variables_str=['DrugsDuringStay']#had to add this as this is a list of strings not ints from a picklist. would put diagnosis in this cat

    string_variables=['Sex','BadgerUniqueID','NationalIDBabyAnon','HeadScanFirstResult','HeadScanLastResult','MaritalStatusMother','BloodGroupMother','GPPostCode','EthnicityMother']

    new_variables=['duration_of_stay','gestation_days','mat_age']

    orig_input_variables_cat=['EpisodeNumber',#these are the variables used for the poster but they contain paameters that can only be known at discharge. 
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

    new_input_variables_cat=['EpisodeNumber',#can adjust these if needed
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
    'Sex','MaritalStatusMother','BloodGroupMother','GPPostCode', 'EthnicityMother']

    new_input_variables_cont=['AdmitTemperature','BirthHeadCircumference','AdmitBloodGlucose',
    'gestation_days','Birthweight','AdmitHeadCircumference','CordArterialpH','CordVenouspH','MembranerupturedDuration','CordClampingTimeSecond',
    'CordClampingTimeMinute','ICCareDays2011','mat_age']#,'HDCareDays2011','SCCareDays2011','NormalCareDays2011'

    orig=0#set to 1 if you want the input vars used in teh bapm paper

    #in order to more easily adjust input variables of model. Remmeber to change saved model 
    if orig==1:
        input_variables_cat=orig_input_variables_cat
        input_variables_cont=orig_input_variables_cont
    else:
        input_variables_cat=new_input_variables_cat
        input_variables_cont=new_input_variables_cont


    output_variables=['duration_of_stay']

    verbose=False

    #Executables---------------------------------------------------------------------------------
    '''Get the file'''
    file_path = 'D:\\Work\\NOML\\data\\NNUEpisodeSummarysince_2010.csv'
    print('getting file and converting to dict')
    data= csv_to_dict(file_path) #create a list of dicts of all episodes. May contain multiple copies of baby. 
    #print(data)
    remainder=set()
    for key in data[0].keys():
        if key in variables:
            continue
        else:
            remainder.add(key)

    #print(remainder)
    #for key in remainder:
     #   print(key,data[0][key])




    '''Now merge babies with multiple episodes by only keeping the latest episode. Remove cases where baby died or was transferred for repatriation or is outside the min and max stay thresholds. 
    Exclude all cases >36 weeks. 
    Exclude readmited and then exclude any duplicates'''
    print('Applying exclusion criteria')
    exclude_list=set()
    exclude_crit=['3','17','99']#,death or repatriation
    for i,case in enumerate(data):
        for crit in exclude_crit:
            if case['DischargeDestination']==crit:
                exclude_HN=case['NationalIDBabyAnon']
                #print(exclude_HN)
                exclude_list.add(exclude_HN)


    for i,exclude in enumerate(exclude_list):#Now remove excluded cases from data
        for j,case in enumerate(data):
            if exclude==case['NationalIDBabyAnon']:
                del data[j]


    exclude_gest=['22','23','24','25','26','36','37','38','39','40']#exclude all episodes of these birth gestation
    for i,case in enumerate(data):
        for crit in exclude_gest:
            if case['GestationWeeks']==crit:
                exclude_HN=case['NationalIDBabyAnon']
                del data[i]


    remove_eps=['10','11','12','13','14','15','16','2']#episode where discharge was to other hospital or to the ward. Preserves the readmission
    for i,case in enumerate(data):
        for crit in remove_eps:
            if case['DischargeDestination']==crit:
                del data[i]

    max_stay=float(100)#exclude cases that have a min or max stay
    min_stay=float(1) 
    for i,case in enumerate(data):
        #print('Birthtime',case['BirthTimeBaby'],'Dischtime',case['DischTime'],case['NationalIDBabyAnon'])
        duration=days_difference(case['BirthTimeBaby'],case['DischTime'])
        #print('duration is', duration)
        if duration is None:
            del data[i]
            continue
        elif float(duration)>max_stay:
            del data[i]
        elif float(duration)<=min_stay:
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

    dups=find_duplicates(HN_list)
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
        vals=convert_to_np(data,keyname)
        data_dict.update({keyname:vals})
    #print(data_dict['GestationWeeks'])    

    #for i,val in enumerate(data_dict['NationalIDBabyAnon']):
      # if val=='D12285FD8B8B14E3C5C7D52BC255E3CFD898EA18':
     #      print('This is dat item..................................',data_dict['BirthDateMother'][i])

    '''Now clean the data -Convert gestation to days val. Get the discharge time. Convert to ints or floats or lists where appropriate. Replace missing vals. '''

    print('Cleaning data')
    clean_data_dict={}
    for variable in int_variables:
        print(variable)
        int_vals=convert_to_int(data_dict[variable])
        clean_data_dict.update({variable:int_vals})
    for variable in float_variables+float_variables_zero:
        #print(variable)
        float_vals=convert_to_float(data_dict[variable])
        clean_data_dict.update({variable:float_vals})
    for variable in list_variables:
        #print(variable)
        list_vals=convert_to_list(data_dict[variable])
        clean_data_dict.update({variable:list_vals})
    for variable in list_variables_str:
        #print(variable)
        list_vals=convert_to_str_list(data_dict[variable])
        clean_data_dict.update({variable:list_vals})   
    for variable in string_variables:
        str_vals=convert_to_str(data_dict[variable])
        #print(variable)
        clean_data_dict.update({variable:str_vals})

    duration_list=[]
    for i,time in enumerate(data_dict['BirthTimeBaby']):
        duration=days_difference(time,data_dict['DischTime'][i])#Note that this is defined from birth to discharge, not admission. Later episodes will not depend on admission time
        duration_list.append(duration)
    clean_duration=convert_to_float(duration_list)    
    clean_data_dict.update({'duration_of_stay':clean_duration})
    
    #Calculate maternale age at birth
    mat_age_list=[]
    for i,time in enumerate(data_dict['BirthDateMother']):
        if data_dict['BirthDateMother'][i]:
            #print(data_dict['NationalIDBabyAnon'][i])
            duration=days_difference(time,data_dict['BirthTimeBaby'][i])#
            mat_age_list.append(duration/365.25)#age in years
            #print(duration)
        else:
            duration=0
            mat_age_list.append(duration/365.25)   
        clean_duration=convert_to_float(mat_age_list)    
        clean_data_dict.update({'mat_age':clean_duration})

    #print(clean_data_dict['duration_of_stay'])
    gestation_days=clean_data_dict['GestationWeeks']*7+clean_data_dict['GestationDays']
    clean_array=convert_to_float(gestation_days)
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
        encoded_array=encode_list(clean_data_dict[variable],verbose=verbose)
        #print(encoded_array[0],'classes',encoded_array[1])
        clean_data_dict.update({'encoded_'+variable:encoded_array[0]})#get the values
        clean_data_dict.update({'encoded_'+variable+'_classes':encoded_array[1]})#get the one hot encoded categories too
    for variable in int_variables+string_variables:
        print('encoding '+variable)
        encoded_array=encode_int(clean_data_dict[variable],verbose=verbose)
        print(encoded_array[0],'classes',encoded_array[1])
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


    '''Now train the model. Random forest, Hist gradient regressor and unsupervised neural network.'''


    print('Now collating model input and output')


    '''Collect the input and output params in X input  and y output arrays'''
    use_encoded_var=True#if categoricals need to be one hot encoded then set this to true
    y=np.array([])
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
    y=np.append(y,clean_data_dict[output_variables[0]])    
    print('X shape',X.shape)
    print('y shape',y.shape)
    full_var_list=[]#label the columns in the input array with teh appropriate category for later analysis. cat names are repeated if the encoding is in the same cat
    for variable in (input_variables_cont):
        full_var_list.append(variable)
    for variable in (input_variables_cat):
        print(variable)
        for category in clean_data_dict[variable+'_classes']:
            for val in category:
                full_var_list.append(val)
        #print('class names are', clean_data_dict[variable+'_classes'])
        #num_cat=len(clean_data_dict[variable+'_classes'][0])
        #for _ in range(num_cat):
        #    full_var_list.append(variable)
    print('length of the input var list labels',len(full_var_list))
    #print('The list is',full_var_list)
    
    '''save clean dict locally as a json so that predict code can access it.'''
    with open(r"D:\Work\NOML\data\clean_data_dict.pkl","wb") as file:
        pickle.dump(clean_data_dict,file) 





    '''select the model'''
    rand_forest=False
    hist_grad_regressor=True
    neural_net=False
    plot=True
    simple_model=False

    if rand_forest==True:
        '''Random forest regressor model training'''
        print('Evaluating Random Forest')
        model=evaluate_model(X,y,splits=10,repeats=20)
        fitted_model=model[2]
        y_actual=model[5]
        X_test=model[4]
        y_train=model[6]
        X_train=model[7]
        score=model[0]
        score_sd=model[1]
        loss_score=model[3]
    

        '''save the model'''
        print('Saving model to file')
        model_pth=os.path.join('random_forest_model.joblib')#can add joins to define a bespoke path to store models
        joblib.dump(fitted_model,model_pth)

        '''Test the model on data and print scores'''
        print('opening model and testing')
        loaded_model=joblib.load(model_pth)
        y_predict=predict_data(X_test,loaded_model)
        y_predict=[math.ceil(num) for num in y_predict]#push the predicted value to the ceiling integer
        calculated_r2=r2_score(y_actual,y_predict)
        print(f'R2 score is {calculated_r2} with cross validation score {score} with sd {score_sd} and MAE {loss_score}') 
        residuals = y_actual - y_predict
        # Calculate standard deviation of residuals 
        n = len(y_actual)
        mean_resid=np.sqrt(np.sum(residuals**2))/n
        sd_resid=np.std(np.sqrt(residuals**2))
        std_dev_residuals = np.sqrt(np.sum(residuals**2) / (n - X.shape[1]))
        print('sd of residuals is ',std_dev_residuals)
        print('calculated sd of residuals ', sd_resid, ' with mean ',mean_resid)
        #with open('rf_result.txt', 'w') as file:
        #     file.write([score,score_sd,loss_score])

        '''find the most important variable sthat explain the model'''
        r=permutation_importance(loaded_model,X_test,y_actual,n_repeats=20,random_state=11)
        #print('mean importances',(r['importances_mean']))
        max_importance_idx=(np.argmax(r['importances_mean']))
        print('max importance idx',max_importance_idx)

        sorted_idx=np.argsort(r['importances_mean'])

        print('The most important variable is ',full_var_list[max_importance_idx])
        for i,idx in enumerate(sorted_idx[:-10-1:-1]):#get the largest values in the arg sort
            #idx=sorted_idx[i]
            #print(idx)
            print(f"The {i} most important variable is {full_var_list[idx]} with a mean of {r['importances_mean'][idx]} and sd {r['importances_std'][idx]}")
            #with open('rf_result.txt') as file:
            #    file.write('The'+ str(i) +'most important variable is '+full_var_list[idx] +'with a mean of '+str(r['importances_mean'][idx])+' and sd '+str(r['importances_std'][idx])+'\n')
        
    if neural_net==True:
        print('Now training the nn model')
        input_size = X.shape[1]
        case_num=X.shape[0]
        print(input_size,'input_size')
        # Data preprocessing - standardizing the continuous features
        print('scaling and masking cont vars')
        continuous_mask = [True if i < cont_d else False for i in range(input_size)] # mask the continuous variables 
        scaler = StandardScaler() 
        X[:, continuous_mask] = torch.tensor(scaler.fit_transform(X[:, continuous_mask]))

        # Splitting the data
        print('splitting data')
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        #print(X_train)
        X_train=torch.tensor(X_train.astype(np.float32))# nn needs them as floats
        X_val=torch.tensor(X_val.astype(np.float32))
        y_train=torch.tensor(y_train.astype(np.float32))
        y_val=torch.tensor(y_val.astype(np.float32))

        print(X_train.dtype,'data type')

        #create a sequential model
        seq_model=nn.Sequential(nn.Linear(input_size, 128),
                                nn.Linear(128, 256),
                                nn.Linear(256, 128),
                                nn.Linear(128, 64),
                                nn.Linear(64, 32),
                                nn.Linear(32, 1),
                                nn.ReLU())


        # Hyperparameters

        lr = 0.001
        n_epochs = 1000
        model = seq_model
        loss_fn = nn.MSELoss() # Loss function for continuous output 
        optimizer = optim.Adam(model.parameters(),lr)


        # Training the model 
        print('training epochs')
        training_loop(n_epochs,lr,optimizer,model,loss_fn,X_train,X_val,y_train,y_val)
        print('output', seq_model(X_val))
        print('answer',y_val)
        
            
        print("Training complete.")

        #Save the model
        torch.save(model.state_dict(), 'my_model.pth')


        # Load the model
    # from model.pth import modelClass
    # model=modelClass()
    # model.load_state_dict(torch.load('model.pth'))
        #loaded_model.eval() 

        # Using the loaded model to make predictions
        #new_data =torch.tensor(...) # Example new data 
        #with torch.no_grad():
        #    predictions = loaded_model(new_data)
        #    print("Predicted duration of stay:\n", predictions)

    if hist_grad_regressor==True:
        print('Evaluating Histogram Gradient Boost Regressor')
        cat_feat=np.zeros(X.shape[1],dtype=bool)#define an array the same size as all categories
        cat_feat[cont_d+1:-1]=True#true for cat vars
        model=evaluate_hist_model(X,y,splits=10,repeats=20,cat_feat=cat_feat)
        fitted_model=model[2]
        y_actual=model[5]
        X_test=model[4]
        y_train=model[6]
        X_train=model[7]
        score=model[0]
        score_sd=model[1]
        loss_score=model[3]
        print('X test shape ', X_test.shape)
        '''save the model'''
        print('Saving model to file')
        model_pth=os.path.join('new_hist_grad_boost_model.joblib')#can add joins to define a bespoke path to store models
        joblib.dump(fitted_model,model_pth)

        '''Test the model on data and print scores'''
        print('opening model and testing')
        loaded_model=joblib.load(model_pth)
        y_predict_hist=predict_data(X_test,loaded_model)
        y_predict_hist=[math.ceil(num) for num in y_predict_hist]#push the predicted value to the ceiling integer
        calculated_r2=r2_score(y_actual,y_predict_hist)
        print(f'R2 score is {calculated_r2} with cross validation score {score} with sd {score_sd} and MAE {loss_score}') 
        residuals = y_actual - y_predict_hist
        # Calculate standard deviation of residuals 
        n = len(y_actual)
        mean_resid=np.sqrt(np.sum(residuals**2))/n
        sd_resid=np.std(np.sqrt(residuals**2))
        std_dev_residuals = np.sqrt(np.sum(residuals**2) / (n - X.shape[1]))
        print('sd of residuals is ',std_dev_residuals)
        print('calculated sd of residuals ', sd_resid, ' with mean ',mean_resid)
        #with open('hist_result.txt', 'w') as file:
        #     file.write(score,score_sd,loss_score])
        '''find the most important variable sthat explain the model'''
        r=permutation_importance(loaded_model,X_test,y_actual,n_repeats=20,random_state=11)
        #print('mean importances',(r['importances_mean']))
        max_importance_idx=(np.argmax(r['importances_mean']))
        print('max importance idx',max_importance_idx)

        sorted_idx=np.argsort(r['importances_mean'])

        print('The most important variable is ',full_var_list[max_importance_idx])
        for i,idx in enumerate(sorted_idx[:-10-1:-1]):#get the largest values in the arg sort
            #idx=sorted_idx[i]
            #print(idx)
            print(f"The {i} most important variable is {full_var_list[idx]} with a mean of {r['importances_mean'][idx]} and sd {r['importances_std'][idx]}")
            #with open('hist_result.txt', 'w') as file:
            # file.write('The'+ str(i) +'most important variable is '+full_var_list[idx] +'with a mean of '+str(r['importances_mean'][idx])+' and sd '+str(r['importances_std'][idx])+'\n')

    if simple_model==True:#model that simply predicts discharge day as Term
        y_actual_simple=clean_data_dict['duration_of_stay']
        y_predict_simple=37*7-clean_data_dict['gestation_days']


    if plot==True:
        #Plots for information---------
        
        fig,ax=plt.subplots()
        #ax.set_ylim(0,np.max(y_actual))
        #coefficients=np.polyfit(clean_data_dict['gestation_days'],clean_data_dict['duration_of_stay'],1)
        #best_fit_line=np.polyval(coefficients,clean_data_dict['gestation_days'])
        #ax.plot(clean_data_dict['gestation_days'],best_fit_line,color='k')
        #array=np.array(clean_data_dict['duration_of_stay'])
        #ax.scatter(clean_data_dict['gestation_days']/7,array,s=4,c='b',label='Gestation vs Duration of stay')
        if rand_forest==True:
            mod_type='rand_for'
            ax.scatter(y_actual,y_predict,s=4,c='b',alpha=0.5,label='RF Model vs Actual')
        if hist_grad_regressor==True:
            mod_type='histgrad'
            data = pd.DataFrame({
            'y_actual': y_actual,
            'y_predict': y_predict_hist
            })
            #sns.regplot(data=data,y="y_actual",x='y_predict',ax=ax,line_kws={'color':'blue'},scatter_kws={'alpha':0.5,'color':'blue','s':5},ci=95,order=1,x_estimator=np.median)
            mod_type='histgrad_scatter'
            ax.set_title('Hist Grad Boost model plot')
            ax.scatter(y_predict_hist,y_actual,s=4,c='blue',alpha=0.5,label='Hist Boost Regressor Model vs Actual')
        if neural_net == True:
            mod_type='nn'
            y_actual=y_val
            y_predict=seq_model(X_val)
            y_predict=y_predict.detach().numpy()
            ax.scatter(y_actual,y_predict,s=4,c='r',alpha=0.5,label='Neural net Model vs Actual')   
        if simple_model==True:
            mod_type='simple'
            data = pd.DataFrame({
            'y_actual': y_actual_simple,
            'y_predict': y_predict_simple
            })
            sns.regplot(data=data,y="y_actual",x='y_predict',ax=ax,line_kws={'color':'darkgreen'},scatter_kws={'alpha':0.5,'color':'green','s':5},ci=95,order=1,x_estimator=np.median)
            #ax.scatter(y_actual_simple,y_predict_simple,s=4,c='green',alpha=0.5,label='Simplistic Model vs Actual') 
            ax.set_title('Gestation only model plot (95% CI)')

        ax.legend()
        x=np.linspace(0,np.max(data['y_actual']),len(data['y_actual']))
        y=x
        ax.set_xlim(0,np.max(data['y_actual']))
        ax.set_ylim(0,np.max(data['y_actual']))
        ax.plot(x,y,color='k',lw=0.5,ls='-')
        ax.set_xlabel('Predicted duration of stay (days)')
        ax.set_ylabel('Actual duration of stay (days)')
        
        #plt.xlim(22,42)
        #plt.ylim(0,300)
        #plt.tight_layout(pad=0.2)
        #plt.legend(fontsize='x-small',labelspacing=0.2,columnspacing=1)
        plt.show() 
        plt.savefig(r'D:\Work\NOML\plots\model_plot_'+mod_type+'.png',dpi=1200)

if __name__ == "__main__":
    main()


'''
-----------------Descriptives of variables below-----------------------------------------------------------------------
BadgerUnique ID is unique to episode
DischargeDestination(1 - Home ,2 - Ward ,3 - Died ,4 - Foster Care ,10 - Transferred to another hospital for continuing care / Higher medical care ,11 - Transferred to another hospital for specialist care 
12 - Transferred to another hospital for surgical care, 
13 - Transferred to another hospital for cardiac care, 
14 - Transferred to another hospital for ecmo, 
15 - Transferred to another hospital due to lack of equipment / Cot space, 
16 - Transferred to another hospital due to insufficient staffing, 
17 - Transferred to another hospital for repatriation / Closer to home, 
99 - Unknown)
Field
 Type
 Description
 Coding
 Field
 Type
 1 - Yes 
9 - Unknown 
DrugsAbusedMother
 str (255)
 Where drug misuse a problem before the start of pregnancy, a list of drugs mother used in this
 pregnancy from which the baby is likely to show withdrawal effects. Multi choice pick list
 appearing as comma delimited list
 1 - Methadone 
2 - Heroin 
3 - Other opiates 
4 - Benzodiazepines 
5 - Cocaine 
8 - Other 
9 - Unknown 
SmokingMother
 numInt
 Description Yes if recorded in maternal antenatal record that she was smoking at the time of booking in
 Coding
 Field
 Type
 this pregnancy.
 1 - Yes 
0 - No 
CigarettesMother - Got rid of this one as input was not consistent. 
 numInt
 Description If mother recorded as smoking at time of booking in this pregnancy, average number of
 cigarettes smoked a day at that time.
 Field
 Type
 AlcoholMother
 str (15)
 Description Record of maternal alcohol use this pregnancy
 0 - None 
Coding
 1 - Social 
2 - Heavy 
9 - Unknown 
ProblemsPregnancyMother
 Field
 Type
 str (65)
 Description List of problems encountered relating to this pregnancy. Multi choice pick list
 00 - None 
Coding
 Field
 Type
 10 - Intrauterine growth restriction 
11 - Poor biophysical profile 
12 - Reduced fetal movements 
13 - Oligohydramnios 
14 - Polyhydramnios 
15 - Fetal abnormality 
16 - Cord problems 
17 - Twin to twin transfusion 
18 - Chorioamnionitis 
19 - Preterm rupture of membranes 
20 - Prolonged rupture membranes 
21 - Cervical suture 
22 - Maternal Gp B Strep 
23 - Maternal UTI 
24 - Other infection 
25 - Rhesus 
26 - Other haemolytic disease 
27 - Placental abruption 
28 - Placenta praevia 
29 - Other antepartum haemorrhage 
30 - Pregnancy induced hypertension 
31 - Pre-eclampsia 
32 - Maternal HELLP 
33 - Gestational diabetes 
34 - Cholestasis of pregnancy 
88 - Other

LabourDelivery
 Field
 Type
 Coding
 numInt
 Description Mode of delivery (single choice pick list)
 1 - Emergency caesarean section - not in labour 
2 - Emergency caesarean section - in labour 
3 - Elective section - not in labour 
4 - Elective section - in labour 
5 - Vaginal - forceps assisted 
6 - Vaginal- spontaneous 
7 - Vaginal - ventouse assisted 
9 - Unknown

Resuscitation
 str (30)
 00 - None 
10 - Stimulation 
11 - Positioning managing airways 
12 - Oxygen 
13 - Suction 
14 - Bag and mask IPPV 
15 - Intubation 
16 - Cardiac massage 
17 - Adrenaline 
18 - Tracheal suction for meconium 
19 - Face Mask CPAP 
88 - Other drugs 
99 - Unknown

NEC
 numInt
 Description Necrotising enterocolitis was suspected or confirmed on any day in this episode.
 0 - None 
Coding
 1 - Confirmed 
'''

