import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping

sns.set_style('darkgrid')

#import dataset
credit = pd.read_csv('../datasets/default of credit card clients.csv')

print(credit)

#exploratory data analysis
credit.head()

credit.info()

credit.describe()

#drop id as its not needed in analysis
credit = credit.drop('ID', axis = 1)

#check for duplicates and drop if there are any
credit.duplicated().sum()
credit = credit.drop_duplicates()

#check distribution for normalization (not normalized and will need to be scaled for accuracy)
sns.displot(credit['AGE'], bins = 20, kde = True)

sns.displot(credit['BILL_AMT4'], bins = 30, kde = True)

sns.displot(credit['PAY_AMT6'], bins = 30, kde = True)

#correlation heatmap to check for relationships between variables
plt.figure(figsize = (12,8))
sns.heatmap(credit.corr(), cmap = 'viridis')

#histograms for distribution between categorical variables
sns.countplot(x = 'SEX', data= credit, palette = 'viridis')

credit['SEX'].value_counts() #count printout

sns.countplot(x = 'SEX', data = credit, palette = 'viridis', hue = 'dpnm') #hue of default or not

credit['EDUCATION'].value_counts()

sns.countplot(x ='EDUCATION', data = credit, palette = 'bright')

sns.countplot(x = 'EDUCATION', data = credit, palette = 'bright', hue = 'dpnm')

credit['MARRIAGE'].value_counts()

sns.countplot(x = 'MARRIAGE', data = credit, palette = 'rocket')

sns.countplot(x = 'MARRIAGE', data = credit, palette = 'rocket', hue = 'dpnm')

#OUTLIERS looking into through boxplots
sns.boxplot(x = 'LIMIT_BAL', data = credit)

sns.boxplot(x = 'PAY_AMT6', data = credit)

credit = credit[(credit['LIMIT_BAL'] <= 550000) & (credit['PAY_AMT6'] <= 50000)]

credit.info()

sns.boxplot(x = 'LIMIT_BAL', data = credit)

sns.boxplot(x = 'PAY_AMT6', data = credit)


#SCALING AND SPLITTING DATA
scaler = StandardScaler()
scaler.fit(credit.drop('dpnm', axis = 1))

scaled_data = scaler.transform(credit.drop('dpnm', axis = 1))

credit_feat = pd.DataFrame(scaled_data, columns = credit.columns[:-1])
credit_feat.head()

#TTS

X_train, X_test, y_train, y_test = train_test_split(scaled_data, credit['dpnm'], test_size = 0.30, random_state=42)


# NEURAL NETWORK MODEL

model = Sequential()
# input layer
model.add(Dense(50,  activation='relu'))
model.add(Dropout(0.4))

# hidden layer
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.4))

# hidden layer
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.4))

# output layer
model.add(Dense(units=1,activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')

#prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model.fit(x=X_train, 
          y=y_train, 
          epochs=50,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )




neural_predict = (model.predict(X_test) > 0.5).astype("int32")

print(confusion_matrix(y_test, neural_predict))
print('\n')
print(classification_report(y_test, neural_predict))





