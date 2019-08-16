#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, LabelEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


#import dataset
dataset = pd.read_csv("Data.csv")

X  = dataset.iloc[:,:-1].values
#X2 = dataset[["Country","Age","Salary"]]
y  = dataset.iloc[:,3].values 


# Imputation for missing values
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X[:,1:3])
X[:,1:3] = imp.transform(X[:,1:3])

# Encoding categorical data

transformer_X = ColumnTransformer(transformers=[("OneHot",OneHotEncoder(),[0])],remainder="passthrough")
X = transformer_X.fit_transform(X)

transformer_y = LabelEncoder()
y = transformer_y.fit_transform(y)

# Split data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size = 0.2, random_state = 0)


# Feateure Scaling
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test)