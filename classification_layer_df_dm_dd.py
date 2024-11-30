from y_set_df_dm_dd import dm_df_dd_list
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from feature_vector_test import sensor2_vector,sensor3_vector,sensor4_vector
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ExpSineSquared


from x_y_set_dm_df_dd import X,y




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#################     PCA     #####################

from sklearn.decomposition import PCA

wine_pca = PCA(n_components=0.9, random_state = 42)
wine_pca.fit(X_train)
X_train = wine_pca.transform(X_train)
X_train = pd.DataFrame(X_train)
X_test = wine_pca.transform(X_test)

###############################################
#LR = LogisticRegression(
#                tol=0.00000001,
#                C=0.7,
#                class_weight={'df&dm':2,'df':2,'dm':2,'clean':2,'df&dm&dd':1},
#                max_iter=100000,
#                warm_start=True
#)

#LR.fit(X_train, y_train)
#y_pred = LR.predict(X_test)
gpc = GaussianProcessClassifier()




svm = SVC(
          C = 1 , 
          kernel= 'sigmoid',
          tol = 1e-12,
          gamma=1,
          coef0=1,
          probability=True,
          shrinking=True,
          cache_size=5000,
          max_iter=-1
          )


svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

#from sklearn.model_selection import cross_val_score,LeaveOneOut
#y_pred = cross_val_score(svm,X,y,cv=LeaveOneOut())


CM = confusion_matrix(y_test,y_pred)
print(CM)
accuracy = accuracy_score(y_test, y_pred)
print('===============')
print("Accuracy:", accuracy)


