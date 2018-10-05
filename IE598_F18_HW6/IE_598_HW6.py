from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
    
iris=datasets.load_iris()
X=iris.data[:,[2,3]]
y=iris.target
#Class labels:[0 1 2]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,stratify=y)
tree=DecisionTreeClassifier()
param_grid={'max_depth':[1,2,3,4,5],
        'criterion':['gini','entropy']}
tree_cv = GridSearchCV(tree, param_grid,cv=10)
tree_cv.fit(X,y)
print(tree_cv.best_params_)
print('\n')

treecv=DecisionTreeClassifier(criterion='gini',max_depth=3)
scorecv=cross_val_score(treecv,X_train,y_train,cv=10)
treecv.fit(X_train,y_train)
print(scorecv)
print(scorecv.mean())
print(scorecv.std())
print(treecv.score(X_test,y_test))
print('\n')

score=[0,0,0,0,0,0,0,0,0,0]
scoretrain=[0,0,0,0,0,0,0,0,0,0]
for i in range(1,11):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i,stratify=y)
    tree=DecisionTreeClassifier(criterion='gini',max_depth=3)
    tree.fit(X_train,y_train)
    score[i-1]=tree.score(X_test,y_test).copy()
    scoretrain[i-1]=tree.score(X_train,y_train).copy()

print(score)
score=np.asarray(score)
print(score.mean())
print(score.std())
print('\n')

print(scoretrain)
scoretrain=np.asarray(scoretrain)
print(scoretrain.mean())
print(scoretrain.std())
print('\n')

print("My name is Guanhua Sun")
print("My NetID is: guanhua4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.") 