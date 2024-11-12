from y_set_df_dm_dd import layer_damage,dm_df_dd_list
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from x_set_creator import sensor_mean,sensor_max,sensor_median_high,sensor_stdev


X = sensor_stdev
y = layer_damage['Layer_3']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#LR = LogisticRegression(
#                tol=0.00000001,
#                C=0.7,
#                class_weight={'df&dm':2,'df':2,'dm':1,'clean':1},
#                max_iter=100000,
#                warm_start=True
#)

#LR.fit(X_train, y_train)
#y_pred = LR.predict(X_test)

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)


CM = confusion_matrix(y_test,y_pred)
print(CM)
accuracy = accuracy_score(y_test, y_pred)
print('===============')
print("Accuracy:", accuracy)





