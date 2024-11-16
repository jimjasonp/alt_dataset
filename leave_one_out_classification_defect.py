from sklearn.model_selection import LeaveOneOut,cross_val_score
from sklearn.svm import SVC
from y_set_single_defect import data
from numpy import mean,absolute,std,median

y = data['defects']
X = data.loc[:, data.columns != 'defects']
cv = LeaveOneOut()

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

#view mean absolute error
print('mean of all the accuracies')
print(mean(absolute(scores)))
print('standard deviation of all the accuracies is')
print(std(absolute(scores)))