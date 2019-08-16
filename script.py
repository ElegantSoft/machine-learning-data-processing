#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

#import dataset
dataset = pd.read_csv("Data.csv")

X  = dataset.iloc[:,:-1].values
#X2 = dataset[["Country","Age","Salary"]]
y  = dataset.iloc[:,3].values 



imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X[:,1:3])
X[:,1:3] = imp.transform(X[:,1:3])