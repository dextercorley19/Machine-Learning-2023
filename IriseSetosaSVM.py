from sklearn import svm
from sklearn import metrics
from sklearn.utils import resample
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from matplotlib import cm
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

#import and look at dataset
iris_setosa = pd.read_csv('/Users/dcmac14/Documents/GitHub/Machine-Learning-2023/Datasets/iris-1.csv')
iris_setosa

#Make a column that holds whether the flower is iris_setosa or not in binary
print(iris_setosa["Species"])
iris_setosa['Iris-setosa'] = np.where(iris_setosa['Species'] == 'Iris-setosa', 1, 0)

#make sure all the columns have proper datatypes
iris_setosa.dtypes
iris_setosa['Iris-setosa'].unique() #checking that there are only 1's and 0's in the new column
iris_setosa.drop('Species', axis=1, inplace=True) #drop old species column (no longer relevent)
iris_setosa

#prepping for SVM by created seperate dataframes for the predictors and response variables
X = iris_setosa.drop('Iris-setosa', axis=1).copy() #predictor variable set without iris setosa column
X.head
y = iris_setosa['Iris-setosa'].copy() #response variable just iris setosa column
y.head

#splitting the data into 30% test and 70% train, then scaling data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30) 
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

clf_svm = SVC() #classifier support vector machine
clf_svm.fit(X_train_scaled, y_train) #fitting the scaled x train, y train data to the SVM

#confusion matrix - create y predicted values then make confusion matrix for the test data result
y_pred = clf_svm.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Not Iris Setosa", 'Iris Setosa'])
disp.plot()

