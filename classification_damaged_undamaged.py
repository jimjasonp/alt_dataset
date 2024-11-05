from y_set_creator_dmg_percentage import DL1,DL2,DL3,DL4,DL5,DL1_total,DL3_total,DL5_total
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
import pandas as pd
from x_set_creator import sensor_mean


X = sensor_mean
y = DL2['dd']

counter =0 
for i in y:
    if i =='undamaged':
        counter +=1

j =6 

while j>5:
    train_counter =0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True)
    for i in y_train:
        if i =='undamaged':
            train_counter +=1
    if train_counter >0.5*counter:
        j=0

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

LR = LogisticRegression(
                tol=0.00000000001,
                C=0.7,
                class_weight={'damaged':1.9,'undamaged':4},
                max_iter=100000,
                warm_start=True)
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)

CM = confusion_matrix(y_test,y_pred)
print(CM)
accuracy = accuracy_score(y_test, y_pred)
print('===============')
print("Accuracy:", accuracy)
