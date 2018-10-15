import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']
feat_labels = df_wine.columns[1:]
X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

bscore=0
bi=0

for i in [25,50,75,100,200,300,400,500]:
    forest = RandomForestClassifier(n_estimators=i,criterion='gini',max_depth=1,random_state=1)
    score = cross_val_score(forest,X_train,y_train,cv=10)
    print(i)
    print(score)
    print(score.mean())
    if(score.mean()>bscore):
        bscore=score.mean()
        bmodel=forest
        bi=i
print(bi)
print(bscore)
bmodel.fit(X_train,y_train)
print(bmodel.score(X_test,y_test))
importances = bmodel.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

print("My name is Guanhua Sun")
print("My NetID is: guanhua4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.") 