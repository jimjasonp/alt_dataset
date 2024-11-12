from y_set_creator_dmg_percentage import y_set_creator
from x_set_creator import sensor_mean,sensor_max,sensor_median_high,sensor_stdev,feature_vector
from helper_functions import model_choice
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,root_mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import time

start = time.time()
mode = 'regression'  #classification / regression
sensor_list = ['s2','s3','s4']      #s2,s3,s4
feature = sensor_mean   #sensor_median_high,sensor_max,sensor_mean,sensor_stdev
damage_index = 'Damage_percentage' # ['Damage_percentage', 'DamageLayer1', 'DamageLayer2', 'DamageLayer3', 'DamageLayer4', 'DamageLayer5']

X = feature[sensor_list]
X = feature.iloc[:,:]

y = y_set_creator(damage_index,mode)
y = y.iloc[:,:]

data_percentage = 0.33  # 0 ---> 1

if data_percentage == 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True)
else:
    X, X_drop, y, y_drop = train_test_split(X, y, test_size=1-data_percentage,shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,shuffle=True)

samples = len(X_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
end = time.time()


def reg_models(model):
    start_model = time.time()
    y_pred = model_choice(model,X_train,y_train,X_test)
    mae = mean_absolute_error(y_test,y_pred)
    end_model = time.time()
    training_time = end_model-start_model
    return [mae,training_time]


def bar_charts_mae():

    lr = reg_models('linear_regression')
    mae_lr = lr[0]
    time_lr = lr[1]*1000

    rf = reg_models('RF')
    mae_rf = rf[0]
    time_rf = rf[1]*1000
    
    xgb = reg_models('xgb')
    mae_xgb = xgb[0]
    time_xgb = xgb[1]*1000


    #mae bar plot
    
    # creating the dataset
    data = {'Linear Regression':mae_lr, 
            'Random Forest':mae_rf,
            'XGB':mae_xgb}

    model_names  = list(data.keys())
    mae = list(data.values())
 
    fig = plt.figure(figsize = (10, 5))

    # creating the bar plot
    plt.bar(model_names, mae, color ='maroon', 
            width = 0.4)

    plt.xlabel("Models")
    plt.ylabel("Mean absolute error")
    plt.title(f"Mean absolute error of models with number of samples used = {samples}")
    plt.show()


    #time bar plot
    
    # creating the dataset
    data = {'Linear Regression':time_lr, 
            'Random Forest':time_rf,
            'XGB':time_xgb}

    model_names  = list(data.keys())
    time = list(data.values())
 
    fig = plt.figure(figsize = (10, 5))

    # creating the bar plot
    plt.bar(model_names, time, color ='maroon', 
            width = 0.4)

    plt.xlabel("Models")
    plt.ylabel("Training time")
    plt.title(f"Training time of models with number of samples used = {samples} in ms")
    plt.show()



def scatter_x_y (model):
    y_pred = model_choice(model,X_train,y_train,X_test)
    plt.plot(range(len(y_test)),y_test)
    plt.plot(range(len(y_test)),y_pred,linestyle='dashed')
    plt.xlabel("sample")
    plt.ylabel("y value")
    plt.title(f"Comparison of predicted and y test for each datapoint using {model}")
    plt.legend(["y_test", "y_pred"], loc="lower right")
    plt.show()



model_list = ['linear_regression','RF','xgb']


for model in model_list:
    scatter_x_y(model)

    
bar_charts_mae()
