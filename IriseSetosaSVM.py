from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from matplotlib import cm
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
import glob


iris_setosa = pd.read_csv('/Users/dcmac14/Documents/GitHub/Machine-Learning-2023/IriseSetosaSVM.py')

print("Features: ", iris_setosa.data.describe)