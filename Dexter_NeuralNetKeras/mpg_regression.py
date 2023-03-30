from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mpg = pd.read_csv('../Datasets/auto-mpg.csv')

mpg.head()

mpg.info()

mpg.describe()

mpg = mpg.drop('car name', axis = 1) #dont need for analysis
#remove string characters from horsepower column
mpg['horsepower'] = mpg['horsepower'].str.extract('(\d+)', expand=False)

mpg['cylinders'] = mpg['cylinders'].astype(float)
mpg['horsepower'] = mpg['horsepower'].astype(float)
mpg['weight'] = mpg['weight'].astype(float)
mpg['model year'] = mpg['model year'].astype(float)
mpg['origin'] = mpg['origin'].astype(float)



mpg.duplicated().sum() #0 duplicates

sns.displot(mpg['acceleration'], bins = 20, kde = True)
sns.displot(mpg['cylinders'], bins = 20, kde = True)
sns.displot(mpg['displacement'], bins = 20, kde = True)
sns.displot(mpg['horsepower'], bins = 20, kde = True)
sns.displot(mpg['weight'], bins = 20, kde = True)
sns.displot(mpg['model year'], bins = 20, kde = True)
sns.displot(mpg['origin'], bins = 20, kde = True)


scaler = StandardScaler()
scaler.fit(mpg.drop('mpg', axis = 1))

scaled_data = scaler.transform(mpg.drop('mpg', axis = 1))

mpg_feat = pd.DataFrame(scaled_data, columns = mpg.columns[:-1])
mpg_feat.head()
print(mpg.info())


X_train, X_test, y_train, y_test = train_test_split(scaled_data, mpg['mpg'], test_size = 0.30, random_state=42)

#build model
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model

#k-fold cross validation
k = 4
num_val_samples = len(X_train) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [X_train[:i * num_val_samples],
         X_train[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [y_train[:i * num_val_samples],
         y_train[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=16, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)


all_scores
np.mean(all_scores)

#Saving the validation logs at each fold

num_epochs = 500
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [X_train[:i * num_val_samples],
         X_train[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [y_train[:i * num_val_samples],
         y_train[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16, verbose=0)
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)

    #Building the history of successive mean K-fold validation scores
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

#Plotting validation scores

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

#Plotting validation scores, excluding the first 10 data points
truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

#Training the final model
model = build_model()
model.fit(X_train, y_train,
          epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(X_test, y_test)

test_mae_score


#Generating predictions on new data
predictions = model.predict(X_test)
predictions[0:10]
