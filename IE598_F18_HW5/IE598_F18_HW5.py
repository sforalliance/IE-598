import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA as KPCA

#read the data from file
df_wine=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None,names=["Class","Alcohol","Malic Acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"])

#EDA analysis
print(df_wine.head())
print(df_wine.describe())
print(df_wine.corr())
sns.set(font_scale=1.5)
sns.heatmap(df_wine.corr(),cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':15},yticklabels=df_wine.columns,xticklabels=df_wine.columns)
plt.show()

#standardize the features and split the set
X,y=df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
sc=StandardScaler()
X_std=sc.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_std,y,test_size=0.2,stratify=y,random_state=42)

#logisctic regression and SVM classification
def classifier(X1,X2,y1,y2):
    lr=LogisticRegression()
    lr.fit(X1,y1)
    print(lr.score(X1,y1))
    print(lr.score(X2,y2))
    svm=SVC(kernel="linear")
    svm.fit(X1,y1)
    print(svm.score(X1,y1))
    print(svm.score(X2,y2))
    print("\n")

classifier(X_train,X_test,y_train,y_test)

#PCA decomponent
pca=PCA(n_components=2)
X_train_pca=pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)
classifier(X_train_pca,X_test_pca,y_train,y_test)

#LDA decomponent
lda = LDA(n_components=2)
X_train_lda=lda.fit_transform(X_train,y_train)
X_test_lda=lda.transform(X_test)
classifier(X_train_lda,X_test_lda,y_train,y_test)

#kpca decomponent
kpca=KPCA(n_components=2, kernel='rbf')
X_train_kpca=kpca.fit_transform(X_train)
X_test_kpca=kpca.transform(X_test)
classifier(X_train_kpca,X_test_kpca,y_train,y_test)

for i in [0.2,0.4,0.6,0.8,1]:
    kpca=KPCA(n_components=2, kernel='rbf',gamma=i)
    X_train_kpca=kpca.fit_transform(X_train)
    X_test_kpca=kpca.transform(X_test)
    print(i)
    classifier(X_train_kpca,X_test_kpca,y_train,y_test)

print("My name is Guanhua Sun")
print("My NetID is: guanhua4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")