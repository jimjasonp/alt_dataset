from y_set_df_dm_dd import layer_damage
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from x_set_creator import sensor_mean


X = sensor_mean

y = layer_damage['Layer_1']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

CM = confusion_matrix(y_test,y_pred)
print(CM)
accuracy = accuracy_score(y_test, y_pred)
print('===============')
print("Accuracy:", accuracy)
