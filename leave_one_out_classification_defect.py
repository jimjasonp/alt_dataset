from sklearn.model_selection import LeaveOneOut,cross_val_score,StratifiedKFold,cross_val_predict
from sklearn.svm import SVC
from y_set_single_defect import data
from numpy import mean,absolute,std
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score


y = data['defects']
X = data.loc[:, data.columns != 'defects']

#from x_y_set_dm_df_dd import X,y


cv = LeaveOneOut()

model  = SVC(
          C = 10 , 
          kernel='sigmoid',
          tol = 1e-12,
          gamma=1,
          coef0=1,
          probability=True,
          shrinking=True,
          cache_size=5000,
          max_iter=-1
          )




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


model.fit(X_train,y_train)

pred = cross_val_predict(model, X_test, y_test,
                         cv=cv, n_jobs=-1)
cm = confusion_matrix(y_test,pred)
acc = accuracy_score(y_test,pred)
print(acc)
print(cm)