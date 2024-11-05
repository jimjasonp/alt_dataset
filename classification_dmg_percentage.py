#from y_set_creator import damage_data_df
from y_set_creator_dmg_percentage import y_set_creator
from x_set_creator import sensor_mean,sensor_max,sensor_median_high,sensor_stdev
from helper_functions import model_choice
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix


# to programma trabaei ta dedomena me duo tropous
# o prwtos tropos einai na trabaei to feature apo to data processing
# o deuteros einai apo to csv pou kanei save to main


# -------------------user input-------------------
#############################################
mode = 'classification'  #classification
sensor_list = ['s2','s3','s4']      #s2,s3,s4
feature = sensor_max   #sensor_median_high,sensor_max,sensor_mean,sensor_stdev
damage_index = 'Damage_percentage' # ['Damage_percentage', 'DamageLayer1', 'DamageLayer2', 'DamageLayer3', 'DamageLayer4', 'DamageLayer5']
model = 'svm'                   #knn,svm,DT,dummy,   xgb,linear_regression,RF
data_percentage = 1 # 0-> no data .... 1 -> full dataset (150 samples)
#odhgies gia user:
#------gia na kanw tune to montelo pou thelo peirazw to arxeio training params
#----- an thelo na allaksw ton arithmo twn samples phgainw stis train test split
#------an thelo na treksw arxeio apo data_processing tote energopoiw tis duo parakatw grammes kai tis ftiaxnw opws thelw kai sbhnw to X apo pio katw
#from data_processing import sensor_median_high,sensor_max,sensor_stdev,sensor_mean
#X = sensor_mean.iloc[:,:]
X = feature[sensor_list]
X = feature.iloc[:,:]
#############################################


y = y_set_creator(damage_index,mode)
y = y.iloc[:,:]



if data_percentage == 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True)
else:
    X, X_drop, y, y_drop = train_test_split(X, y, test_size=1-data_percentage,shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,shuffle=True)
# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_pred = model_choice(model,X_train,y_train,X_test)


CM = confusion_matrix(y_test,y_pred)
print(CM)

accuracy = accuracy_score(y_test, y_pred)
print('===============')
print("Accuracy:", accuracy)





