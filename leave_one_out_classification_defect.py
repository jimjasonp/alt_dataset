from sklearn.model_selection import LeaveOneOut,cross_val_score,StratifiedKFold
from sklearn.svm import SVC
from y_set_single_defect import data
from numpy import mean,absolute,std
y = data['defects']
X = data.loc[:, data.columns != 'defects']
#cv = LeaveOneOut()
cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
model  = SVC(
          C = 1 , 
          kernel='rbf',
          tol = 1e-12,
          gamma=1,
          coef0=1,
          probability=True,
          shrinking=True,
          cache_size=5000,
          max_iter=-1
          )
scores = cross_val_score(model, X, y, scoring='accuracy',
                         cv=cv, n_jobs=-1)

print('mean of all the accuracies')
print(mean(absolute(scores)))
print('standard deviation of all the accuracies is')
print(std(absolute(scores)))