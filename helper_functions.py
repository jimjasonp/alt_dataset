import pandas as pd
import statistics
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
feature_list = ['max','mean','stdev','median_high']



def feature_csv_assign(feature,sensor_fft_df,sensor_max,sensor_mean,sensor_stdev,sensor_median_high):

    if feature == 'max':
        sensor_max = sensor_max.assign(**sensor_fft_df)
        sensor_max.to_csv(r'C:\Users\jimja\Desktop\thesis\feature_csvs\sensor_max.csv')
    elif feature =='mean':
        sensor_mean = sensor_mean.assign(**sensor_fft_df)
        sensor_mean.to_csv(r'C:\Users\jimja\Desktop\thesis\feature_csvs\sensor_mean.csv')
    elif feature =='stdev':
        sensor_stdev = sensor_stdev.assign(**sensor_fft_df)
        sensor_stdev.to_csv(r'C:\Users\jimja\Desktop\thesis\feature_csvs\sensor_stdev.csv')
    elif feature =='median_high':
        sensor_median_high = sensor_median_high.assign(**sensor_fft_df)
        sensor_median_high.to_csv(r'C:\Users\jimja\Desktop\thesis\feature_csvs\sensor_median_high.csv')


#epilegw analogws to montelo pou thelw 
def model_choice(model,X_train,y_train,X_test):
    ###########
    #classification
    ###########
    if str(model) == 'knn':
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)


    if str(model) =='svm':
        svm = SVC()
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)



    if str(model) =='DT':
        DT = DecisionTreeClassifier()
        DT.fit(X_train, y_train)
        y_pred = DT.predict(X_test)


    if str(model) =='dummy':
        dummy = DummyClassifier(strategy='uniform')
        dummy.fit(X_train, y_train)
        y_pred = dummy.predict(X_test)


    ###########
    #regression
    ###########


    if str(model) =='xgb':
        xgb = xg.XGBRegressor(objective ='reg:linear', 
                  n_estimators = 10, seed = 123)
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)


    if str(model) =='linear_regression':
        linear_regression = LinearRegression()
        linear_regression.fit(X_train, y_train)
        y_pred = linear_regression.predict(X_test)



    if str(model) =='RF':
        RF = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
        RF.fit(X_train, y_train)
        y_pred = RF.predict(X_test)
    return y_pred